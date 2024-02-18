from __future__ import annotations

import argparse
import pathlib

import numpy as np
from dm_control import composer
from dm_env import specs
from dm_robotics.agentflow.preprocessors import observation_transforms, rewards
from dm_robotics.moma import subtask_env
from dm_robotics.moma.sensors import site_sensor
from dm_robotics.panda import arm_constants, environment
from dm_robotics.panda import parameters as params

from . import screwmpc


def create_environment(
    xml_path: pathlib.Path, args: argparse.Namespace
) -> environment.PandaEnvironment:
    """Creates the basic environment for the experiments."""
    robot_params = params.RobotParams(
        robot_ip=args.robot_ip,
        actuation=arm_constants.Actuation.JOINT_VELOCITY,
        enforce_realtime=args.realtime_priority,
    )
    goal = screwmpc.Goal()
    arena = composer.Arena(xml_path=xml_path)
    arena.attach(goal)

    panda_env = environment.PandaEnvironment(robot_params, arena, control_timestep=0.02)

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
                ["time", "manipulability", "panda_joint_pos"]
            ),
        ]
    )
    return panda_env


def create_argparser() -> argparse.ArgumentParser:
    """Create argument parser with common arguments for all experiments."""

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
        "--realtime-priority",
        "--rt",
        action="store_true",
        help="set the robot control thread priority to realtime",
    )
    return parser


def create_agent(
    env: subtask_env.SubTaskEnvironment, args: argparse.Namespace
) -> screwmpc.ScrewMPCAgent:
    """Creates a screwmpc.ScrewMPCAgent from a moma subtask environment and arguments."""
    return screwmpc.ScrewMPCAgent(
        env.action_spec(),
        args.goal_tolerance,
        args.sclerp,
        args.manipulability,
        args.output,
    )
