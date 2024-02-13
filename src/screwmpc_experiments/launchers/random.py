"""
Launcher for a dm_robotics_panda experiment that
generates motion between random poses.
"""
from __future__ import annotations

import argparse
import logging

from dm_robotics.panda import environment, run_loop, utils
from dm_robotics.panda import parameters as params

from ..experiments import screwmpc

logging.basicConfig(level=logging.INFO)
logger = logging.Logger("screwmpc-random")


def main() -> None:
    """
    Main function of this experiment.
    Run from the terminal by executing `screwmpc-random`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-ip", default=None, type=str, help="Robot IP or hostname for HIL"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Deactivate visualization"
    )
    args = parser.parse_args()

    robot_params = params.RobotParams(robot_ip=args.robot_ip)
    panda_env = environment.PandaEnvironment(robot_params)

    with panda_env.build_task_environment() as env:
        # Print the full action, observation and reward specification.
        # utils.full_spec(env)
        # Initialize the agent.
        agent = screwmpc.ScrewMPCAgent(env.action_spec())
        # Run the environment and agent either in headless mode or inside the GUI.
        if not args.no_gui:
            app = utils.ApplicationWithPlot()
            app.launch(env, policy=agent.step)
        else:
            run_loop.run(env, agent, [], max_steps=1000, real_time=True)
