"""
Launcher for a dm_robotics_panda experiment that
generates motion between random poses.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
from dm_robotics.panda import arm_constants, environment, run_loop, utils
from dm_robotics.panda import parameters as params

from ..experiments import screwmpc

logging.basicConfig(level=logging.INFO, force=True)


def main() -> None:
    """
    Main function of this experiment.
    Run from the terminal by executing `screwmpc-random`.
    """
    logging.basicConfig(level=logging.INFO, force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-ip", default=None, type=str, help="Robot IP or hostname for HIL"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Deactivate visualization"
    )
    args = parser.parse_args()

    robot_params = params.RobotParams(
        robot_ip=args.robot_ip, actuation=arm_constants.Actuation.JOINT_VELOCITY
    )
    panda_env = environment.PandaEnvironment(robot_params, control_timestep=0.01)

    with panda_env.build_task_environment() as env:
        # Use a fixed seed for reproducibility
        rng = np.random.RandomState(seed=1)  # pylint: disable=no-member
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
