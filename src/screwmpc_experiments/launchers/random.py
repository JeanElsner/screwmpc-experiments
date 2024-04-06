"""
Launcher for a dm_robotics_panda experiment that
generates motion between random poses.
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np

from ..experiments import common, screwmpc

logging.basicConfig(level=logging.INFO, force=True)


def _float_array(arg: str) -> list[float]:
    return [float(x) for x in arg.split()]


def main() -> None:
    """
    Main function of this experiment.
    Run from the terminal by executing `screwmpc-random`.
    """
    parser = common.create_argparser()
    parser.add_argument(
        "--seed", type=int, help="set the random seed (default: 1)", default=1
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
    parser.add_argument(
        "-n", type=int, help="number of random poses (default: 10)", default=10
    )
    xml_path = pathlib.Path(__file__).parent / ".." / "assets" / "random.xml"
    args = parser.parse_args()
    panda_env, robot_params, agent = common.create_environment(xml_path, args)

    with panda_env.build_task_environment() as env:
        rng = np.random.RandomState(seed=args.seed)  # pylint: disable=no-member

        # Generate 10 random poses within these bounds centered around the starting pose
        # given as x, y, z (linear) and X, Y, Z (angular, euler angles)
        dp = args.position_delta
        dr = np.deg2rad(args.rotation_delta)
        min_pose_bounds = np.array(
            [
                0.307 - dp[0],
                0 - dp[1],
                0.487 + 0.1034 - dp[2],
                np.pi - dr[0],
                0 - dr[1],
                0.25 * np.pi - dr[2],
            ]
        )
        max_pose_bounds = np.array(
            [
                0.307 + dp[0],
                0 + dp[1],
                0.487 + 0.1034 + dp[2],
                np.pi + dr[0],
                0 + dr[1],
                0.25 * np.pi + dr[2],
            ]
        )
        poses = screwmpc.generate_random_poses(
            args.n, robot_params.joint_positions, min_pose_bounds, max_pose_bounds, rng
        )
        agent.add_waypoints(poses)

        # Run the environment and agent either in headless mode or inside the GUI.
        if not args.no_gui:
            app = screwmpc.ScrewMPCApp(title="ScrewMPC Random Experiment")
            app.launch(env, policy=agent.step)
        else:
            common.run(env, agent)
