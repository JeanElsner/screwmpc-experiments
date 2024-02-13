from __future__ import annotations

import numpy as np
from dqrobotics import robots
from screwmpcpy import pandamg, screwmpc


class ScrewMPCAgent:
    """
    Basic dm-robotics agent that initiates a screwmpc motion generator.
    """

    def __init__(self) -> None:
        Np = 50  # prediction horizon, can be tuned;
        Nc = 10  # control horizon, can be tuned
        R = 10e-3  # weight matirix
        Q = 10e9  # weight matrix

        ub_jerk = np.array([8500.0, 8500.0, 8500.0, 4500.0, 4500.0, 4500.0])
        lb_jerk = -ub_jerk.copy()

        ub_acc = np.array([17.0, 17.0, 17.0, 9.0, 9.0, 9.0])
        lb_acc = -ub_acc.copy()

        ub_v = np.array([2.5, 2.5, 2.5, 3.0, 3.0, 3.0])
        lb_v = -ub_v.copy()

        jerk_bound = screwmpc.BOUND(lb_jerk, ub_jerk)
        acc_bound = screwmpc.BOUND(lb_acc, ub_acc)
        vel_bound = screwmpc.BOUND(lb_v, ub_v)

        mg = pandamg.PandaScrewMotionGenerator(
            Np, Nc, Q, R, vel_bound, acc_bound, jerk_bound
        )
        kin = robots.FrankaEmikaPandaRobot.kinematics()

        del mg, kin
