"""
Launcher for a dm_robotics_panda experiment that
generates motion between random poses.
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
from dm_env import specs
from dm_robotics.agentflow.preprocessors import observation_transforms, rewards
from dm_robotics.moma.models.arenas import empty
from dm_robotics.moma.sensors import site_sensor
from dm_robotics.panda import arm_constants, environment, run_loop, utils
from dm_robotics.panda import parameters as params

from ..experiments import screwmpc

logging.basicConfig(level=logging.INFO, force=True)


def _float_array(arg: str) -> list[float]:
    return [float(x) for x in arg.split()]


def main() -> None:
    """
    Main function of this experiment.
    Run from the terminal by executing `screwmpc-random`.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot-ip", default=None, type=str, help="robot IP or hostname for HIL"
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="deactivate visualization"
    )
    parser.add_argument(
        "--seed", type=int, help="set the random seed (default: 1)", default=1
    )
    parser.add_argument(
        "--goal-tolerance",
        type=float,
        help="norm between two dual quaternion poses to consider a goal reached (default: 0.05)",
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
        "--position-delta",
        type=_float_array,
        default=[0.1, 0.3, 0.3],
        help="position delta for random poses (default: 0.1 0.3 0.3)",
    )
    parser.add_argument(
        "--rotation-delta",
        type=_float_array,
        default=[45, 45, 45],
        help="rotation delta for random poses (default: 45 45 45)",
    )
    args = parser.parse_args()

    robot_params = params.RobotParams(
        robot_ip=args.robot_ip, actuation=arm_constants.Actuation.JOINT_VELOCITY
    )

    goal = screwmpc.Goal()
    arena = empty.Arena()
    arena.attach(goal)

    panda_env = environment.PandaEnvironment(robot_params, arena, control_timestep=0.02)

    # Add extra sensors for flange and goal reference sites
    # to make them observable to the agent and preprocessors.
    flange_sensor = site_sensor.SiteSensor(
        panda_env.robots["panda"].arm.mjcf_model.find("site", "real_aligned_tcp"),
        "flange",
    )
    goal_sensor = site_sensor.SiteSensor(
        goal.mjcf_model.find("site", "wrist_site"), "goal"
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
        ]
    )

    with panda_env.build_task_environment() as env:
        rng = np.random.RandomState(seed=args.seed)  # pylint: disable=no-member

        # Generate 10 random poses within these bounds
        # given as x, y, z (linear) and X, Y, Z (angular, euler angles)
        dp = args.position_delta
        dr = np.deg2rad(args.rotation_delta)
        min_pose_bounds = np.array(
            [
                0.3 - dp[0],
                0 - dp[1],
                0.5 - dp[2],
                np.pi - dr[0],
                0 - dr[1],
                0.25 * np.pi - dr[2],
            ]
        )
        max_pose_bounds = np.array(
            [
                0.3 + dp[0],
                0 + dp[1],
                0.5 + dp[2],
                np.pi + dr[0],
                0 + dr[1],
                0.25 * np.pi + dr[2],
            ]
        )
        poses = screwmpc.generate_random_poses(
            10, min_pose_bounds, max_pose_bounds, rng
        )

        agent = screwmpc.ScrewMPCAgent(
            env.action_spec(), args.goal_tolerance, args.sclerp, args.manipulability
        )
        for p in poses:
            agent.add_waypoint(p)

        # Run the environment and agent either in headless mode or inside the GUI.
        if not args.no_gui:
            app = utils.ApplicationWithPlot()
            app.launch(env, policy=agent.step)
        else:
            run_loop.run(env, agent, [], max_steps=1000, real_time=True)
