from __future__ import annotations

import csv
import logging
import pathlib
import threading
import time
import typing
from collections import deque
from xmlrpc import server

import dm_env
import dm_robotics.panda
import dqrobotics
import mujoco
import numpy as np
import panda_py
import roboticstoolbox as rtb
import spatialmath
from dm_control import mjcf
from dm_control.viewer import application, renderer, user_input
from dm_env import specs
from dm_robotics.agentflow import spec_utils
from dm_robotics.agentflow.preprocessors import timestep_preprocessor
from dm_robotics.geometry import pose_distribution
from dm_robotics.moma import effector, prop, subtask_env
from dm_robotics.panda import utils
from dm_robotics.transformations import transformations as tr
from panda_py import constants
from screwmpcpy import dqutil, pandamg, screwmpc

panda_model = rtb.models.Panda()
logger = logging.getLogger("screwmpc")
T_F_EE: spatialmath.SE3 = spatialmath.SE3(0, 0, 0.1034) * spatialmath.SE3.Rz(
    -45, unit="deg"
)


def pose_to_se3(pose: tuple[np.ndarray, np.ndarray]) -> spatialmath.SE3:
    """Transforms pose consisting of 3-vector and unit quaternion to spatialmath.SE3."""
    return spatialmath.SE3(pose[0]) * spatialmath.UnitQuaternion(pose[1]).SE3()


def se3_to_pose(se3: spatialmath.SE3) -> tuple[np.ndarray, np.ndarray]:
    """Transforms spatialmath.SE3 to pose consisting of 3-vector and unit quaternion."""
    return (se3.t, spatialmath.UnitQuaternion(se3).vec)


def pose_to_dq(pose: tuple[np.ndarray, np.ndarray, int]) -> dqrobotics.DQ:
    """Computes a unit dual quaternion from a pose.

    Args:
      pose: The pose given as a tuple consisting of a 3-vector
        and a unit quaternion.
    """
    se3 = pose_to_se3((pose[0], pose[1]))
    se3 = spatialmath.SE3(0.041, 0, 0) * se3
    return dqrobotics.DQ(spatialmath.UnitDualQuaternion(se3).vec)  # pylint: disable=no-member


def generate_random_poses(
    n: int,
    q_init: np.ndarray,
    min_pose_bounds: np.ndarray,
    max_pose_bounds: np.ndarray,
    random_state: np.random.RandomState,  # pylint: disable=no-member
) -> list[tuple[np.ndarray, np.ndarray, int]]:
    """
    Generate random poses within the given bounds.

    Compute random poses within the given bounds. The poses
    are checked using case consistent inverse kinematics.
    """
    gripper_pose_dist = pose_distribution.UniformPoseDistribution(
        min_pose_bounds=min_pose_bounds,
        max_pose_bounds=max_pose_bounds,
    )
    poses: list[np.ndarray] = []
    while len(poses) < n:
        pose = gripper_pose_dist.sample_pose(random_state)
        se3 = pose_to_se3(pose)
        se3 *= T_F_EE
        q = panda_py.ik(se3, q_init)
        if not np.any(np.isnan(q)):
            poses.append((pose[0], pose[1], 1))
            q_init = q.copy()
    return poses


def get_screw(
    start: dqrobotics.DQ, goal: dqrobotics.DQ
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute Plücker coordinates for motion between two points."""
    screw = [*dqutil.dq_to_plucker(dqutil.delta_dq(start, goal))]
    screw[:2] = normalize(screw[0]), normalize(screw[1])
    return tuple(screw)


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if 0 < norm < np.inf:
        return vec / norm
    return vec


class ScrewMPCAgent:
    """Basic dm-robotics agent that uses a screwmpc motion generator."""

    def __init__(
        self,
        spec: specs.BoundedArray,
        goal_tolerance: float,
        sclerp: float,
        use_mp: bool = False,
        output_file: str = "obs.csv",
        grasp_time: float = 2.0,
    ) -> None:
        self._spec = spec
        self._goal_tolerance = goal_tolerance
        self._waypoints: list[tuple[np.ndarray, np.ndarray, int]] = []
        self._intermediate: list[tuple[np.ndarray, np.ndarray, int]] | None = None
        self._start: dqrobotics.DQ | None = None
        self._goal: dqrobotics.DQ | None = None
        self._plucker = np.zeros(8)
        self._plucker_des = np.zeros(8)
        self._x_goal: tuple[np.ndarray, np.ndarray, float] | None = None
        self._obs: list[dict[str, np.ndarray]] = []
        self._output_file = output_file
        self._finished = False
        self._dead_time = 0.0
        self._grasp_time = grasp_time
        self.motion_generator = create_screwmpc(sclerp, use_mp)
        self.sclerp = sclerp
        self.use_mp = use_mp

    def get_plucker_observation(
        self, timestep: timestep_preprocessor.PreprocessorTimestep
    ) -> np.ndarray:
        del timestep
        return self._plucker

    def get_plucker_desired_observation(
        self, timestep: timestep_preprocessor.PreprocessorTimestep
    ) -> np.ndarray:
        del timestep
        return self._plucker_des

    def step(self, timestep: dm_env.TimeStep) -> np.ndarray:
        """Computes an action given a timestep observation.

        The action is computed with the screwmpc motion generator to reach
        the current goal. Once the goal has been reached, the next waypoint
        added with :py:func:`add_waypoint` becomes the goal.
        """
        action = np.zeros(shape=self._spec.shape, dtype=self._spec.dtype)
        # set gripper action
        if self._goal is not None:
            # set joint velocities to mpc output
            action[:7] = self.motion_generator.step(
                timestep.observation["panda_joint_pos"], self._goal
            )

            start = self.motion_generator._kin.fkm(
                timestep.observation["panda_joint_pos"]
            )
            screw = get_screw(start, self._goal)
            self._plucker = np.hstack(screw)

        if self._x_goal is not None:
            # set goal object pose
            action[8:16] = np.r_[self._x_goal[0], self._x_goal[1], self._x_goal[2]]
            action[7] = self._x_goal[2]

            if self._intermediate is not None:
                for i in range(10):
                    action[i * 7 + 16 : i * 7 + 16 + 7] = np.r_[
                        self._intermediate[i][0], self._intermediate[i][1].vec
                    ]

            # joint_positions = panda_py.fk(timestep.observation["panda_joint_pos"])
            # start = spatialmath.SE3(joint_positions)
            # start *= T_F_EE.inv()

            # intermediate = dqutil.interpolate_waypoints(
            #     [
            #         (start.t, spatialmath.UnitQuaternion(start)),
            #         (self._x_goal[0], spatialmath.UnitQuaternion(self._x_goal[1])),
            #     ],
            #     10,
            # )[1:-1]

            # for i in range(10):
            #     action[i * 7 + 16 : i * 7 + 16 + 7] = np.r_[
            #         intermediate[i][0], intermediate[i][1].vec
            #     ]
        if self.at_goal(timestep):
            logger.info("Goal reached.")
            self._goal = None
            self.motion_generator.reset()
        if not self._finished:
            self._obs.append(timestep.observation)
        if (
            not self._finished
            and self._goal is None
            and self._dead_time < timestep.observation["time"][0]
        ):
            try:
                x_goal = self._waypoints.pop(0)
                self._goal = pose_to_dq(x_goal)

                if (
                    self._x_goal is not None
                    and np.all(x_goal[0] == self._x_goal[0])
                    and np.all(x_goal[1] == self._x_goal[1])
                    and x_goal[2] != self._x_goal[2]
                ):
                    logger.info("Grasping")
                    self._dead_time = timestep.observation["time"][0] + self._grasp_time
                else:
                    logger.info("Tracking new goal: %s", self._goal)

                    self._start = self.motion_generator._kin.fkm(
                        timestep.observation["panda_joint_pos"]
                    )
                    start = (
                        spatialmath.SE3(-0.041, 0, 0)
                        * spatialmath.UnitDualQuaternion(self._start.vec8()).SE3()
                    )
                    self._intermediate = dqutil.interpolate_waypoints(
                        [
                            (start.t, start.UnitQuaternion()),
                            (
                                x_goal[0],
                                spatialmath.UnitQuaternion(x_goal[1]),
                            ),
                        ],
                        10,
                    )[1:-1]
                    screw = get_screw(self._start, self._goal)
                    self._plucker_des = self._plucker = np.hstack(screw)

                self._x_goal = x_goal
                action[8:16] = np.r_[self._x_goal[0], self._x_goal[1], self._x_goal[2]]

            except IndexError:
                self._finished = True
                self._dead_time = 0
                logger.info("Saving observations to %s", self._output_file)
                save_obs(self._obs, self._output_file)
        return action

    def at_goal(self, timestep: dm_env.TimeStep) -> bool:
        """Checks whether the agent has reached the current goal."""
        if self._goal is None or timestep.reward is None:
            return False
        return float(timestep.reward[0]) > -self._goal_tolerance

    def get_observation(self) -> dict[str, float | list[float]]:
        """Get the last environment observation."""
        obs = self._obs[-1]
        for k, v in obs.items():
            if isinstance(v, np.ndarray):
                obs[k] = v.tolist()
        return obs

    def add_waypoints(
        self, waypoints: list[tuple[np.ndarray, np.ndarray, int]] | None = None
    ) -> None:
        """Adds a waypoint to the buffer.

        Waypoints will be moved through consecutively using the
        screwmpc motion generator.

        Args:
          waypoints: List of waypoints to buffer representing goal poses. Waypoints are
            given as a tuple containing a 3-vector, a unit quaternion and a float.
        """
        if waypoints is None:
            return
        self._waypoints.extend(waypoints)
        logger.info("Added %d new waypoints to buffer", len(waypoints))
        self._finished = False

    def get_u_state_observation(
        self, timestep: timestep_preprocessor.PreprocessorTimestep
    ) -> np.ndarray:
        """Retrieves the motion generator's internal `u` state observation."""
        del timestep
        return self.motion_generator.u_state

    def get_mpc_state_observation(
        self, timestep: timestep_preprocessor.PreprocessorTimestep
    ) -> np.ndarray:
        """Retrieves the motion generator's internal `mpc` state observation."""
        del timestep
        return self.motion_generator.mpc_state


class RPCInterface:
    """Remote procedure call interface to interact with the simulation."""

    def __init__(
        self,
        agent: ScrewMPCAgent,
        env: subtask_env.SubTaskEnvironment,
        collision_env: subtask_env.SubTaskEnvironment,
        gui: ScrewMPCApp | None = None,
    ) -> None:
        self._agent = agent
        self._env = env
        self._collision_env = collision_env
        self._gui = gui
        self.init_xmlrpc()

    def _check_collision(
        self, q: np.ndarray, check_box: bool = True
    ) -> tuple[bool, str]:
        physics = self._collision_env.physics
        robot = next(iter(self._collision_env.task.robots))
        robot.position_arm_joints(physics, q)
        robot.gripper.set_joint_positions(physics, [0.04, 0.04])
        physics.forward()

        for contact in physics.data.contact:
            geom1_name: str = physics.model.id2name(contact.geom1, "geom")
            geom2_name: str = physics.model.id2name(contact.geom2, "geom")
            if (
                geom1_name.startswith("panda/")
                or geom2_name.startswith("panda/")
                and not (
                    geom1_name == "panda/panda_gripper/panda_leftfinger_collision"
                    and geom2_name == "panda/panda_gripper/panda_rightfinger_collision"
                )
                and not (
                    geom2_name == "panda/panda_gripper/panda_leftfinger_collision"
                    and geom1_name == "panda/panda_gripper/panda_rightfinger_collision"
                )
            ) and (check_box ^ ("box/body" not in [geom1_name, geom2_name])):
                return False, f"{geom1_name} in collision with {geom2_name}"
        return True, ""

    def check_box_collision(
        self,
        pose: tuple[np.ndarray, np.ndarray, int] | None = None,
        n_subsamples: int = 16,
    ) -> tuple[bool, str, list[float]]:
        """Checks for collision between the robot and the bounding box."""
        if pose is None:
            return True, "", []
        tic = time.time()
        result = self._check_ik(pose, n_subsamples)
        toc = time.time() - tic
        if result[0]:
            logger.info("found collision free ik solution after %.3fs", toc)
        else:
            logger.error("did not find collision free ik solution after %.3fs", toc)
        return result

    def _check_ik(
        self,
        pose: tuple[np.ndarray, np.ndarray, int],
        n_subsample: int,
    ) -> tuple[bool, str, list[float]]:
        se3 = pose_to_se3((pose[0], pose[1]))
        se3 *= T_F_EE
        qs_7 = np.linspace(
            constants.JOINT_LIMITS_LOWER[6],
            constants.JOINT_LIMITS_UPPER[6],
            n_subsample,
        )
        collision_result = (True, "")
        for q_7 in qs_7:
            qs = panda_py.ik_full(se3, q_7=q_7)
            for q in qs:
                if not np.any(np.isnan(q)):
                    collision_result = self._check_collision(q, True)
                    if collision_result[0]:
                        return *collision_result, q.tolist()
        if not collision_result[0]:
            return *collision_result, []
        return False, f"no IK solution found for pose {pose}", []

    def check_feasibility(
        self,
        waypoints: list[tuple[list[float], list[float], int]] | None = None,
        q_init: list[float] | None = None,
        n_points: int = 3,
    ) -> tuple[bool, str]:
        """Check inverse kinematics, collisions amnd reachability for a list of waypoints."""
        result = True, ""
        if waypoints is None:
            return result
        if q_init is None:
            result = False, "initial joint positions required"
        tic = time.time()
        preprocessed = []
        for i, wp in enumerate(waypoints):
            if i > 0 and wp[0] == waypoints[i - 1][0] and wp[1] == waypoints[i - 1][1]:
                continue
            preprocessed.append(
                (np.array(wp[0]), spatialmath.UnitQuaternion(wp[1]), wp[2])
            )
        motion_generator = create_screwmpc(0.5, False)
        stop = False
        for p in preprocessed:
            if stop:
                break
            traj, success = compute_trajectory(
                motion_generator, q_init, pose_to_dq(p), dt=0.2
            )
            if not success:
                result = False, f"screwmpc failed to reach waypoint {p}"
                break
            for i in np.round(np.linspace(0, len(traj) - 1, n_points + 2)).astype(
                np.int8
            )[1:]:
                collision_result = self._check_collision(traj[i], False)
                if not collision_result[0]:
                    result = collision_result
                    stop = True
                    break
            q_init = traj[-1]
        if result[0]:
            logger.info("feasibility check successful after %.3fs", time.time() - tic)
        else:
            logger.error("feasibility check failed after %.3fs", time.time() - tic)
        return result

    def reload_box(
        self,
        pose: tuple[np.ndarray, np.ndarray] | None = None,
        size: np.ndarray | None = None,
    ) -> None:
        """Resize and reposition the bounding box object (requires resetting the simulation)."""
        logger.info("Updating bounding box object")
        self._reload_box(self._env, pose, size)
        if self._gui is not None:
            self._gui._restart_runtime()  # pylint: disable=protected-access
        self._agent.motion_generator = create_screwmpc(
            self._agent.sclerp, self._agent.use_mp
        )
        self._reload_box(self._collision_env, pose, size)
        self._collision_env.reset()

    def _reload_box(
        self,
        env: subtask_env.SubTaskEnvironment,
        pose: tuple[np.ndarray, np.ndarray] | None = None,
        size: np.ndarray | None = None,
    ) -> None:
        """Reload the bounding box object with new size and pose."""
        if pose is None or size is None:
            return
        env.task.arena.mjcf_model.find("geom", "box/body").size[:] = size
        env.task.arena.mjcf_model.find("geom", "box/body").pos[:] = pose[0]
        env.task.arena.mjcf_model.find("geom", "box/body").quat[:] = pose[1]

    def init_xmlrpc(self) -> None:
        """Initiate server to handle remote procedure calls."""
        self.server = server.SimpleXMLRPCServer(
            ("0.0.0.0", 9000), allow_none=True, logRequests=False
        )
        self.server.register_function(self._agent.add_waypoints)
        self.server.register_function(self._agent.get_observation)
        self.server.register_function(self.check_feasibility)
        self.server.register_function(self.reload_box)
        self.server.register_function(self.check_box_collision)
        self._thread = threading.Thread(target=self.server.serve_forever)
        self._thread.start()

    def shutdown(self) -> None:
        """Shut down the server and close any open connections."""
        self.server.shutdown()
        self.server.socket.close()
        self._thread.join()
        self._collision_env.close()


class Goal(prop.Prop):  # type: ignore[misc]
    """Intangible prop representing the goal pose."""

    def _build(self, color: tuple[float, float, float, float] = (1, 0, 0, 0.3)) -> None:  # pylint: disable=arguments-differ
        xml_path = (
            pathlib.Path(pathlib.Path(dm_robotics.panda.__file__).parent)
            / "assets"
            / "panda"
            / "panda_hand.xml"
        )
        mjcf_root = mjcf.from_path(xml_path)
        for geom in mjcf_root.find_all("geom"):
            geom.rgba = color
            geom.conaffinity = 0
            geom.contype = 0
        rotated_root = mjcf.RootElement()
        rotated_root.worldbody.add("body", name="rotated_root").add(
            "site", euler=[0, 0, -0.7854]
        ).attach(mjcf_root)
        super()._build("goal", rotated_root, "rotated_root")


class SceneEffector(effector.Effector):  # type: ignore[misc]
    """
    Effector used to update the state of the scene.
    """

    def __init__(self, goal: Goal, intermediate: list[Goal]) -> None:
        self._goal = goal
        self._intermediate = intermediate
        self._actuator = goal.mjcf_model.find(
            "actuator", "panda_hand/panda_hand_actuator"
        )
        self._intermediate_actuator = []
        for i in self._intermediate:
            self._intermediate_actuator.append(
                i.mjcf_model.find("actuator", "panda_hand/panda_hand_actuator")
            )
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
                (78,),
                np.float32,
                np.full((78,), -10, dtype=np.float32),
                np.full((78,), 10, dtype=np.float32),
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
                            "goal_grasp",
                        ]
                        + [f"intermediate_{i}" for i in range(70)]
                    ]
                ),
            )
        return self._spec

    @property
    def prefix(self) -> str:
        return "scene"

    def set_control(self, physics: mjcf.Physics, command: np.ndarray) -> None:
        pos = command[:3]
        self._goal.set_pose(physics, pos, command[3:7])
        physics.bind(self._actuator).ctrl = command[7]

        for i, __ in enumerate(self._intermediate):
            subcommand = command[(i + 1) * 7 + 1 : (i + 1) * 7 + 8]
            self._intermediate[i].set_pose(physics, subcommand[:3], subcommand[3:7])
            physics.bind(self._intermediate_actuator).ctrl = command[7]


def goal_reward(observation: spec_utils.ObservationValue) -> np.ndarray:
    """Computes a reward based on distance between end-effector and goal."""
    pos_distance = np.linalg.norm(observation["goal_pos"] - observation["flange_pos"])
    rot_dist = tr.quat_dist(observation["goal_quat"], observation["flange_quat"])

    if (
        np.linalg.norm(observation["plucker_des"][3:6]) < np.inf
        and observation["plucker_des"][6] > 0.02
    ):
        l_error = -1 + np.dot(
            observation["plucker"][:3], observation["plucker_des"][:3]
        )
        m_error = -1 + np.dot(
            observation["plucker"][3:6], observation["plucker_des"][3:6]
        )
    else:
        l_error = 0
        m_error = 0
    return np.array([float(-pos_distance - rot_dist / np.pi), l_error, m_error])


def manipulability(timestep: timestep_preprocessor.PreprocessorTimestep) -> np.ndarray:
    """Computes manipulability observable."""
    observation = timestep.observation
    return np.array(
        [panda_model.manipulability(observation["panda_joint_pos"])], dtype=np.float32
    )


def save_obs(obs: list[dict[str, np.ndarray]], output_file: str) -> None:
    """Saves a list of observations to a CSV file."""
    # Extracting field names from the first row
    first_row = obs[0]
    fieldnames = []
    for key, value in first_row.items():
        if np.ndim(value) > 0:
            if len(value) > 1:
                fieldnames.extend([f"{key}_{i+1}" for i in range(len(value))])
            else:
                fieldnames.append(key)
        else:
            fieldnames.append(key)
    # Writing to CSV
    with pathlib.Path(output_file).open("w", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in obs:
            new_row = {}
            for key, value in row.items():
                if np.ndim(value) > 0:
                    if len(value) > 1:
                        for i, v in enumerate(value):
                            new_row[f"{key}_{i+1}"] = v
                    else:
                        new_row[key] = value[0]
                else:
                    new_row[key] = value.item()  # Convert 0-d array to scalar
            writer.writerow(new_row)


class ScrewMPCActionPlot(utils.ActionPlot):  # type: ignore[misc]
    """A plotting component for :py:class:`dm_control.viewer.application.Application`
    that plots the agent's actions in a screwmpc experiment.
    """

    def _init_buffer(self) -> None:
        self.maxlines = 8
        for _1 in range(self.maxlines):
            self.y.append(deque(maxlen=self.maxlen))
        self.reset_data()

    def render(self, context: mujoco.MjrContext, viewport: renderer.Viewport) -> None:
        if self._rt._time_step is None or self._rt.last_action is None:  # pylint: disable=protected-access
            return
        for i, a in enumerate(self._rt.last_action):
            if i > 7:
                break
            self.fig.linepnt[i] = self.maxlen
            self.y[i].append(a)
            self.fig.linedata[i][: self.maxlen * 2] = np.array(
                [self.x, self.y[i]]
            ).T.reshape((-1,))
        pos = mujoco.MjrRect(300 + 5, viewport.height - 200 - 5, 300, 200)  # pylint: disable=no-member
        mujoco.mjr_figure(pos, self.fig, context.ptr)  # pylint: disable=no-member


class ScrewMPCApp(utils.ApplicationWithPlot):  # type: ignore[misc]
    """Extends the GUI application with RPC functionality."""

    def __init__(
        self,
        title: str = "ScrewMPC Experiment",
        width: int = 1024,
        height: int = 768,
    ):
        super().__init__(title, width, height)
        self._viewer.render_settings.toggle_rendering_flag(0)
        self._viewer.render_settings.toggle_rendering_flag(2)

    def _perform_deferred_reload(self, params: application.ReloadParams) -> None:
        application.Application._perform_deferred_reload(self, params)  # pylint: disable=protected-access
        cmp = utils.ObservationPlot(self._runtime)
        self._renderer.components += cmp
        self._renderer.components += ScrewMPCActionPlot(self._runtime)
        self._renderer.components += utils.RewardPlot(self._runtime)
        self._input_map.bind(cmp.next_obs, user_input.KEY_F4)
        self._input_map.bind(cmp.prev_obs, user_input.KEY_F3)


class Box(prop.Prop):  # type: ignore[misc]
    """The bounding box object."""

    def _build(  # pylint:disable=arguments-renamed
        self,
        name: str = "box",
        size: typing.Iterable[float] = (0.04, 0.04, 0.04),
        pos: typing.Iterable[float] = (0, 0, 0),
        quat: typing.Iterable[float] = (1, 0, 0, 0),
    ) -> None:
        mjcf_root = _make_block_model(name, size, pos, quat)
        super()._build(name, mjcf_root, "prop_root")


def _make_block_model(
    name: str,
    size: typing.Iterable[float],
    pos: typing.Iterable[float],
    quat: typing.Iterable[float],
    color: typing.Iterable[float] = (1, 0, 0, 1),
    solimp: typing.Iterable[float] = (0.95, 0.995, 0.001),
    solref: typing.Iterable[float] = (0.002, 0.7),
) -> mjcf.RootElement:
    mjcf_root = mjcf.element.RootElement(model=name)
    prop_root = mjcf_root.worldbody.add("body", name="prop_root")
    box = prop_root.add(
        "geom",
        name="body",
        type="box",
        pos=pos,
        quat=quat,
        size=size,
        mass=0.01,
        solref=solref,
        solimp=solimp,
        condim=1,
        rgba=color,
    )
    del box
    return mjcf_root


def compute_trajectory(
    motion_generator: pandamg.PandaScrewMotionGenerator,
    q_init: np.ndarray,
    x_d: dqrobotics.DQ,
    dt: float = 0.001,
    linear_threshold: float = 0.05,
    angular_threshold: float = 5,
    max_steps: int = 1000,
) -> tuple[list[np.ndarray], bool]:
    """Offline computation of trajectory from `q_init` to `x_d`."""
    q_robot: np.ndarray = q_init.copy()
    joint_angles: list[np.ndarray] = [q_init]
    step: int = 0
    done: bool = False
    goal_pose = dqutil.dq_to_pose(x_d)

    while not done:
        try:
            dq = motion_generator.step(q_robot, x_d)
            q_robot += dq * dt
        except ValueError as e:
            logger.error(e)
            return joint_angles, False
        joint_angles.append(q_robot.copy())
        step += 1

        pose = panda_py.fk(q_robot)
        se3 = spatialmath.SE3(pose)
        se3 = spatialmath.SE3(0.041, 0, 0) * se3 * T_F_EE.inv()

        linear_error = np.sqrt(np.linalg.norm(goal_pose[0] - se3.t))
        angular_error = np.rad2deg(
            spatialmath.SO3.angdist(
                spatialmath.UnitQuaternion(goal_pose[1]).SO3(), se3.R
            )
        )
        done = (
            linear_error <= linear_threshold and angular_error < angular_threshold
        ) or step >= max_steps
    return (
        joint_angles,
        linear_error <= linear_threshold and angular_error < angular_threshold,
    )


def create_screwmpc(sclerp: float, use_mp: bool) -> pandamg.PandaScrewMotionGenerator:
    """Creates a screwmpc motion generator."""
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

    generator = (
        pandamg.PandaScrewMpRGMotionGenerator
        if use_mp
        else pandamg.PandaScrewMotionGenerator
    )
    return generator(
        n_p,
        n_c,
        Q,
        R,
        vel_bound,
        acc_bound,
        jerk_bound,
        sclerp=sclerp,
    )
