"""
Launcher for a dm_robotics_panda experiment that
generates motion between random poses.
"""
from __future__ import annotations

import logging
import pathlib

from dm_robotics.panda import run_loop

from ..experiments import common, screwmpc

logging.basicConfig(level=logging.INFO, force=True)


def main() -> None:
    """
    Main function of this experiment.
    Run from the terminal by executing `screwmpc-pivoting`.
    """
    args = common.create_argparser().parse_args()
    xml_path = pathlib.Path(__file__).parent / ".." / "assets" / "pivoting.xml"
    panda_env = common.create_environment(xml_path, args)

    box = screwmpc.Box(pos=[0.3, 0, 0.3])
    panda_env.add_props([box])

    with panda_env.build_task_environment() as env:
        agent = common.create_agent(env, args)
        # Run the environment and agent either in headless mode or inside the GUI.
        if not args.no_gui:
            app = screwmpc.ScrewMPCApp("Screw MPC Pivoting Experiment", box=box)
            app.launch(env, policy=agent.step)
        else:
            run_loop.run(env, agent, [], max_steps=1000, real_time=True)

    agent.shutdown()
