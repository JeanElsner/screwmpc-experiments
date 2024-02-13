from __future__ import annotations

import contextlib
import logging
import pathlib

import dm_env
import dm_robotics.panda
import dqrobotics
import numpy as np
import panda_py
import roboticstoolbox as rtb
import spatialmath
from dm_control import mjcf
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
from dm_robotics.geometry import pose_distribution
from dm_robotics.moma import effector, prop
from dm_robotics.transformations import transformations as tr
from dqrobotics import robots
from screwmpcpy import pandamg, screwmpc

panda_model = rtb.models.Panda()
logger = logging.getLogger("screwmpc")


def pose_to_dq(pose: tuple[np.ndarray, np.ndarray]) -> dqrobotics.DQ:
    """Computes a unit dual quaternion from a pose.

    Args:
      pose: The pose given as a tuple consisting of a 3-vector
        and a unit quaternion.
    """
    se3 = (
        spatialmath.SE3(*pose[0])
        * spatialmath.UnitQuaternion(pose[1][0], pose[1][1:]).SE3()
    )
    return dqrobotics.DQ(spatialmath.UnitDualQuaternion(se3).vec)  # pylint: disable=no-member


def generate_random_poses(
    n: int,
    min_pose_bounds: np.ndarray,
    max_pose_bounds: np.ndarray,
    random_state: np.random.RandomState,  # pylint: disable=no-member
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Generate random poses within the given bounds.

    Compute random poses within the given bounds. The poses
    are checked using inverse kinematics.
    """
    gripper_pose_dist = pose_distribution.UniformPoseDistribution(
        min_pose_bounds=min_pose_bounds,
        max_pose_bounds=max_pose_bounds,
    )
    poses: list[np.ndarray] = []
    while len(poses) < n:
        pose = gripper_pose_dist.sample_pose(random_state)
        q = panda_py.ik(pose[0], pose[1])
        if not np.any(np.isnan(q)):
            poses.append(pose)
    return poses


class ScrewMPCAgent:
    """Basic dm-robotics agent that uses a screwmpc motion generator."""

    def __init__(self, spec: specs.BoundedArray, goal_tolerance: float) -> None:
        self._spec = spec
        self._goal_tolerance = goal_tolerance
        self._waypoints: list[tuple[np.ndarray, np.ndarray]] = []
        self._goal: dqrobotics.DQ | None = None
        self._x_goal: tuple[np.ndarray, np.ndarray] | None = None
        self.init_screwmpc()

    def add_waypoint(self, waypoint: tuple[np.ndarray, np.ndarray]) -> None:
        """Adds a waypoint to the buffer.

        Waypoints will be moved through consecutively using the
        screwmpc motion generator.

        Args:
          waypoint: Waypoint to add to buffer representing a goal pose
            given as a tuple containing a 3-vector and a quaternion.
        """
        self._waypoints.append(waypoint)

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        """Computes an action given a timestep observation.

        The action is computed with the screwmpc motion generator to reach
        the current goal. Once the goal has been reached, the next waypoint
        added with :py:func:`add_waypoint` becomes the goal.
        """
        action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
        if self._goal is not None and self._x_goal is not None:
            action[:7] = self.motion_generator.step(
                timestep.observation["panda_joint_pos"], self._goal
            )
            action[-7:-4] = self._x_goal[0]
            action[-4:] = self._x_goal[1]
        if self.at_goal(timestep):
            logger.info("Goal reached.")
            self._goal = None
        if self._goal is None:
            with contextlib.suppress(IndexError):
                self._x_goal = self._waypoints.pop(0)
                self._goal = pose_to_dq(self._x_goal)
                logger.info("Tracking new goal: %s", self._goal)

        return action

    def at_goal(self, timestep: dm_env.TimeStep) -> bool:
        """Checks whether the agent has reached the current goal."""
        if self._goal is None:
            return False
        return bool(
            np.linalg.norm(
                (
                    self.kinematics.fkm(timestep.observation["panda_joint_pos"])
                    - self._goal
                )
                .norm()
                .vec8()
            )
            < self._goal_tolerance
        )

    def init_screwmpc(self) -> None:
        """Initializes the screwmpc motion generator."""
        n_p = 50  # prediction horizon, can be tuned;
        n_c = 10  # control horizon, can be tuned
        R = 10e-3  # weight matirix
        Q = 10e9  # weight matrix

        ub_jerk = np.array([8500.0, 8500.0, 8500.0, 4500.0, 4500.0, 4500.0])
        lb_jerk = -ub_jerk.copy()

        ub_acc = np.array([17.0, 17.0, 17.0, 9.0, 9.0, 9.0])
        lb_acc = -ub_acc.copy()

        ub_v = np.array([2.5, 2.5, 2.5, 3.0, 3.0, 3.0])
        lb_v = -ub_v.copy()

        jerk_bound = screwmpc.BOUND(lb_jerk, ub_jerk)
        acc_bound = screwmpc.BOUND(lb_acc, ub_acc)
        vel_bound = screwmpc.BOUND(lb_v, ub_v)

        self.motion_generator = pandamg.PandaScrewMotionGenerator(
            n_p, n_c, Q, R, vel_bound, acc_bound, jerk_bound
        )
        self.kinematics = robots.FrankaEmikaPandaRobot.kinematics()  # pylint: disable=no-member


class Goal(prop.Prop):  # type: ignore[misc]
    """Intangible prop representing the goal pose."""

    def _build(self) -> None:  # pylint: disable=arguments-differ
        xml_path = (
            pathlib.Path(pathlib.Path(dm_robotics.panda.__file__).parent)
            / "assets"
            / "panda"
            / "panda_hand.xml"
        )
        mjcf_root = mjcf.from_path(xml_path)
        for geom in mjcf_root.find_all("geom"):
            geom.rgba = (1, 0, 0, 0.3)
            geom.conaffinity = 0
            geom.contype = 0
        super()._build("goal", mjcf_root, "panda_hand")


class SceneEffector(effector.Effector):  # type: ignore[misc]
    """
    Effector used to update the state of the scene.
    """

    def __init__(self, goal: Goal) -> None:
        self._goal = goal
        self._spec = None

    def close(self) -> None:
        pass

    def initialize_episode(
        self, physics: mjcf.Physics, random_state: np.random.RandomState
    ) -> None:
        pass

    def action_spec(self, physics: mjcf.Physics) -> specs.BoundedArray:
        del physics
        if self._spec is None:
            self._spec = specs.BoundedArray(
                (7,),
                np.float32,
                np.full((7,), -10, dtype=np.float32),
                np.full((7,), 10, dtype=np.float32),
                "\t".join(
                    [
                        f"{self.prefix}_{n}"
                        for n in [
                            "goal_x",
                            "goal_y",
                            "goal_z",
                            "goal_qw",
                            "goal_qx",
                            "goal_qy",
                            "goal_qz",
                        ]
                    ]
                ),
            )
        return self._spec

    @property
    def prefix(self) -> str:
        return "scene"

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        minus45deg = tr.euler_to_quat([0, 0, -np.pi / 4])
        self._goal.set_pose(physics, command[:3], tr.quat_mul(command[3:], minus45deg))


def goal_reward(observation: spec_utils.ObservationValue) -> float:
    """Computes a reward based on distance between end-effector and goal."""
    pos_distance = np.linalg.norm(observation["goal_pos"] - observation["flange_pos"])
    rot_dist = tr.quat_dist(observation["goal_quat"], observation["flange_quat"])
    return float(-pos_distance - rot_dist / np.pi)


def manipulability(timestep: timestep_preprocessor.PreprocessorTimestep) -> np.ndarray:
    """Computes manipulability observable."""
    observation = timestep.observation
    return np.array(
        [panda_model.manipulability(observation["panda_joint_pos"])], dtype=np.float32
    )
