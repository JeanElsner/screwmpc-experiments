from __future__ import annotations

import contextlib
import logging

import dm_env
import dqrobotics
import numpy as np
import panda_py
import spatialmath
from dm_env import specs
from dm_robotics.geometry import pose_distribution
from dqrobotics import robots
from screwmpcpy import pandamg, screwmpc

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

    def __init__(self, spec: specs.BoundedArray) -> None:
        self._spec = spec
        self._waypoints: list[tuple[np.ndarray, np.ndarray]] = []
        self._goal: dqrobotics.DQ | None = None
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
        if self._goal is not None:
            action[:7] = self.motion_generator.step(
                timestep.observation["panda_joint_pos"], self._goal
            )
        if self.at_goal(timestep):
            logger.info("Goal reached.")
            self._goal = None
        if self._goal is None:
            with contextlib.suppress(IndexError):
                self._goal = pose_to_dq(self._waypoints.pop(0))
                logger.info("Tracking new goal.")

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
            < 0.05
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
