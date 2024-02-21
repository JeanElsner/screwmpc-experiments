from __future__ import annotations

import numpy as np

from screwmpc_experiments.experiments import screwmpc

dp = [0.2, 0.2, 0.2]
dr = np.deg2rad([45, 45, 45])
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
rng = np.random.RandomState(0)


def test_se3_pose_transform():
    poses = screwmpc.generate_random_poses(100, min_pose_bounds, max_pose_bounds, rng)
    for pose in poses:
        se3 = screwmpc.pose_to_se3(pose)
        pose_2 = screwmpc.se3_to_pose(se3)
        se3_2 = screwmpc.pose_to_se3(pose_2)

        assert np.allclose(se3, se3_2)
        assert np.allclose(pose[0], pose_2[0])
        assert np.allclose(pose[1], pose_2[1])
