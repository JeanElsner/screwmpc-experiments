"""
Launcher for a dm_robotics_panda experiment that
generates motion between random poses.
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np
from dm_control import composer
from dm_env import specs
from dm_robotics.agentflow.preprocessors import (
    observation_transforms,
    rewards,
)
from dm_robotics.moma.sensors import site_sensor
from dm_robotics.panda import arm_constants, environment, run_loop
from dm_robotics.panda import parameters as params

from ..experiments import screwmpc

logging.basicConfig(level=logging.INFO, force=True)


def main() -> None:
    """
    Main function of this experiment.
    Run from the terminal by executing `screwmpc-pivoting`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-ip", default=None, type=str, help="robot IP or hostname for HIL"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="deactivate visualization"
    )
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        help="norm between two dual quaternion poses to consider a goal reached (default: 0.01)",
        default=0.01,
    )
    parser.add_argument(
        "-m",
        "--manipulability",
        action="store_true",
        help="use manipulability maximizing",
    )
    parser.add_argument(
        "--sclerp", type=float, help="sclerp coefficient (default: 0.1)", default=0.1
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="filename of the csv output (default: obs.csv)",
        default="obs.csv",
    )
    parser.add_argument(
        "--rt",
        "--realtime-priority",
        action="store_true",
        help="set the robot control thread priority to realtime",
    )
    args = parser.parse_args()

    robot_params = params.RobotParams(
        robot_ip=args.robot_ip,
        actuation=arm_constants.Actuation.JOINT_VELOCITY,
        enforce_realtime=args.rt,
    )

    goal = screwmpc.Goal()
    xml_path = pathlib.Path(__file__).parent / ".." / "assets" / "scene.xml"
    arena = composer.Arena(xml_path=xml_path)
    arena.attach(goal)

    panda_env = environment.PandaEnvironment(robot_params, arena, control_timestep=0.02)

    box = screwmpc.Box(pos=[0.3, 0, 0.3])
    panda_env.add_props([box])

    # Add extra sensors for flange and goal reference sites
    # to make them observable to the agent and preprocessors.
    flange_sensor = site_sensor.SiteSensor(
        panda_env.robots["panda"].arm.mjcf_model.find("site", "real_aligned_tcp"),
        "flange",
    )
    goal_sensor = site_sensor.SiteSensor(
        goal.mjcf_model.find("site", "panda_hand/wrist_site"), "goal"
    )

    panda_env.add_extra_sensors([flange_sensor, goal_sensor])
    panda_env.add_extra_effectors([screwmpc.SceneEffector(goal)])
    panda_env.add_timestep_preprocessors(
        [
            observation_transforms.AddObservation(
                "manipulability",
                screwmpc.manipulability,
                specs.Array((1,), dtype=np.float32),
            ),
            rewards.ComputeReward(screwmpc.goal_reward),
            observation_transforms.RetainObservations(
                ["time", "manipulability", "panda_joint_pos", "panda_tcp_pos"]
            ),
        ]
    )

    with panda_env.build_task_environment() as env:
        agent = screwmpc.ScrewMPCAgent(
            env.action_spec(),
            args.goal_tolerance,
            args.sclerp,
            args.manipulability,
            args.output,
        )
        # Run the environment and agent either in headless mode or inside the GUI.
        if not args.no_gui:
            app = screwmpc.ScrewMPCApp("Screw MPC Pivoting Experiment", box=box)
            app.launch(env, policy=agent.step)
            app.shutdown()
        else:
            run_loop.run(env, agent, [], max_steps=1000, real_time=True)

    agent.shutdown()
