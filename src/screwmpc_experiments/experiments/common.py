from __future__ import annotations

import argparse
import logging
import pathlib

import numpy as np
import panda_py
from dm_control import composer
from dm_env import specs
from dm_robotics.agentflow.preprocessors import observation_transforms, rewards
from dm_robotics.moma import subtask_env
from dm_robotics.moma.sensors import site_sensor
from dm_robotics.panda import arm_constants, environment, run_loop
from dm_robotics.panda import parameters as params
from panda_py import constants

from . import screwmpc


def create_environment(
    xml_path: pathlib.Path, args: argparse.Namespace
) -> tuple[environment.PandaEnvironment, params.RobotParams, screwmpc.ScrewMPCAgent]:
    """Creates the basic environment for the experiments."""
    agent = create_agent(args)

    if not args.move_to_start and args.robot_ip is not None:
        q = panda_py.Panda(args.robot_ip).q
    else:
        q = constants.JOINT_POSITION_START
        # q = np.array([-0.13083284218478622, -0.6435437375202514, -1.2014952440764255, -1.3871403382451892, 0.2967545260853238, 1.124549853383633, 2.329884358906789])
    robot_params = params.RobotParams(
        robot_ip=args.robot_ip,
        joint_positions=q,
        actuation=arm_constants.Actuation.JOINT_VELOCITY,
        enforce_realtime=args.realtime_priority,
    )
    goal = screwmpc.Goal(color=(0, 1, 0, 0.3))
    arena = composer.Arena(xml_path=xml_path)
    arena.attach(goal)
    intermediate = []
    blend = np.linspace(0, 1, 10)

    if args.visualize_sclerp:
        for i in range(10):
            intermediate.append(screwmpc.Goal(color=(1 - blend[i], blend[i], 0, 0.3)))
            arena.attach(intermediate[-1])

    panda_env = environment.PandaEnvironment(
        robot_params, arena, control_timestep=0.02, physics_timestep=0.001
    )

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
    panda_env.add_extra_effectors([screwmpc.SceneEffector(goal, intermediate)])
    panda_env.add_timestep_preprocessors(
        [
            # observation_transforms.AddObservation(
            #     "u_state", agent.get_u_state_observation
            # ),
            # observation_transforms.AddObservation(
            #     "mpc_state", agent.get_mpc_state_observation
            # ),
            observation_transforms.AddObservation(
                "plucker",
                agent.get_plucker_observation,
                specs.Array((8,), dtype=np.float64),
            ),
            observation_transforms.AddObservation(
                "plucker_des",
                agent.get_plucker_desired_observation,
                specs.Array((8,), dtype=np.float64),
            ),
            observation_transforms.AddObservation(
                "manipulability",
                screwmpc.manipulability,
                specs.Array((1,), dtype=np.float32),
            ),
            rewards.ComputeReward(screwmpc.goal_reward, output_spec_shape=(3,)),
            observation_transforms.RetainObservations(
                [
                    "time",
                    "manipulability",
                    "panda_joint_pos",
                    "panda_joint_vel",
                    "panda_tcp_pos",
                    "panda_tcp_quat",
                    "panda_force",
                    "panda_torque",
                    "plucker",
                    "plucker_des",
                ]
            ),
        ]
    )
    return panda_env, robot_params, agent


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
    parser.add_argument(
        "--grasp-time",
        type=float,
        help="Time to wait for a grasp to complete",
        default=2.0,
    )
    parser.add_argument(
        "--move-to-start",
        action="store_true",
        help="move the robot into the start position in the beginning",
    )
    parser.add_argument(
        "--visualize-sclerp",
        action="store_true",
        help="visualize sclerp poses",
    )
    return parser


def create_agent(
    args: argparse.Namespace,
) -> screwmpc.ScrewMPCAgent:
    """Creates a screwmpc.ScrewMPCAgent from a moma subtask environment and arguments."""
    return screwmpc.ScrewMPCAgent(
        specs.BoundedArray((86,), np.float64, -np.inf, np.inf),
        args.goal_tolerance,
        args.sclerp,
        args.manipulability,
        args.output,
        grasp_time=args.grasp_time,
    )


def run(env: subtask_env.SubTaskEnvironment, agent: screwmpc.ScrewMPCAgent) -> None:
    """Run an experiment without the GUI."""
    try:
        run_loop.run(env, agent, [], max_steps=100000, real_time=True)
    except KeyboardInterrupt:
        logging.info("Exiting")
