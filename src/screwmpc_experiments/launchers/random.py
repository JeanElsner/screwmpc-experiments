"""
Launcher for a dm_robotics_panda experiment that
generates motion between random poses.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
from dm_robotics.moma.models.arenas import empty
from dm_robotics.panda import arm_constants, environment, run_loop, utils
from dm_robotics.panda import parameters as params

from ..experiments import screwmpc

logging.basicConfig(level=logging.INFO, force=True)


def main() -> None:
    """
    Main function of this experiment.
    Run from the terminal by executing `screwmpc-random`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-ip", default=None, type=str, help="Robot IP or hostname for HIL."
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Deactivate visualization."
    )
    parser.add_argument(
        "--seed", "-s", type=int, help="Set the random seed.", default=2
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    robot_params = params.RobotParams(
        robot_ip=args.robot_ip, actuation=arm_constants.Actuation.JOINT_VELOCITY
    )

    goal = screwmpc.Goal()
    arena = empty.Arena()
    arena.attach(goal)

    panda_env = environment.PandaEnvironment(robot_params, arena, control_timestep=0.01)
    panda_env.add_extra_effectors([screwmpc.SceneEffector(goal)])

    with panda_env.build_task_environment() as env:
        rng = np.random.RandomState(seed=args.seed)  # pylint: disable=no-member

        # Generate 10 random poses within these bounds
        # given as x, y, z (linear) and X, Y, Z (angular, euler angles)
        min_pose_bounds = np.array(
            [0.5, -0.3, 0.7, 0.75 * np.pi, -0.25 * np.pi, -0.25 * np.pi]
        )
        max_pose_bounds = np.array(
            [0.1, 0.3, 0.1, 1.25 * np.pi, 0.25 * np.pi / 2, 0.25 * np.pi]
        )
        poses = screwmpc.generate_random_poses(
            10, min_pose_bounds, max_pose_bounds, rng
        )
        agent = screwmpc.ScrewMPCAgent(env.action_spec())
        for p in poses:
            agent.add_waypoint(p)

        # Run the environment and agent either in headless mode or inside the GUI.
        if not args.no_gui:
            app = utils.ApplicationWithPlot()
            app.launch(env, policy=agent.step)
        else:
            run_loop.run(env, agent, [], max_steps=1000, real_time=True)
