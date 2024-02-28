from __future__ import annotations

import argparse
import logging
from xmlrpc import client

import numpy as np
import spatialmath

logging.basicConfig(level=logging.INFO)

# this is the transform between flange and TCP of the Panda
T_F_EE: spatialmath.SE3 = spatialmath.SE3(0, 0, 0.1034) * spatialmath.SE3.Rz(
    -45, unit="deg"
)


def homogeneous_to_waypoint(
    T: np.ndarray, grasp: float
) -> tuple[list[float], list[float], float]:
    """Computes waypoint arguments given pose as homogeneous transform.

    You may use this function to apply any necessary transforms."""
    # TODO: the frame seems to be neither flange nor TCP, please investigate
    se3 = spatialmath.SE3(T, check=False)
    # we apply a 45 degree rotation around the *local* z-axis
    se3 *= spatialmath.SE3.Rz(45, unit="deg")  # this fixes the orientation
    return (se3.t.tolist(), spatialmath.UnitQuaternion(se3).vec.tolist(), grasp)


def main() -> None:
    """Entry point, triggers a grasp in the simulation."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hostname",
        type=str,
        help="Hostname of the computer running the simulation",
        default="localhost",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Target an experiment without running GUI.",
    )
    args = parser.parse_args()

    with client.ServerProxy(f"http://{args.hostname}:9000/") as proxy:
        # get last observation
        logging.info("Received observation %s", proxy.get_observation())

    # pose is direct output of the task planner
    se3 = spatialmath.SE3(
        np.array(
            [
                [
                    4.340418381597311859e-01,
                    -9.008927143266958204e-01,
                    0.000000000000000000e00,
                    2.054405438859776001e-01,
                ],
                [
                    9.008927143266959314e-01,
                    4.340418381597315189e-01,
                    -0.000000000000000000e00,
                    -3.533969723184959832e-01,
                ],
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    1.000000000000000000e00,
                    2.626889896036288530e-01,
                ],
                [
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    0.000000000000000000e00,
                    1.000000000000000000e00,
                ],
            ]
        )
    )
    pos, quat = (se3.t, spatialmath.UnitQuaternion(se3).vec)
    pose = (pos.tolist(), quat.tolist())
    # size is computed based on the bounding box vertices
    size = [0.1100, 0.0350, 0.0850]

    # height of the table computed as pos[2]-size[2]/2
    # see src/screwmpc_experiments/assets/pivoting.xml for reference

    # Make sure arguments to the server proxy are lists,
    # numpy arrays are not marshable.
    if not args.no_gui:
        with client.ServerProxy(f"http://{args.hostname}:9001/") as proxy:
            proxy.reload_box(pose, size)  # reload bounding box in sim

    # these waypoints were generated by the task planner
    approach = np.array(
        [
            [
                0.000000000000000000e00,
                9.008927143266958204e-01,
                4.340418381597311859e-01,
                1.032081088989951073e-01,
            ],
            [
                0.000000000000000000e00,
                -4.340418381597315189e-01,
                9.008927143266959314e-01,
                -5.601555803357990415e-01,
            ],
            [
                1.000000000000000000e00,
                0.000000000000000000e00,
                0.000000000000000000e00,
                3.943490077179162778e-01,
            ],
            [
                0.000000000000000000e00,
                0.000000000000000000e00,
                0.000000000000000000e00,
                1.000000000000000000e00,
            ],
        ]
    )

    pre_grasp = np.array(
        [
            [
                0.000000000000000000e00,
                9.008927143266958204e-01,
                4.340418381597311859e-01,
                1.032081088989951073e-01,
            ],
            [
                0.000000000000000000e00,
                -4.340418381597315189e-01,
                9.008927143266959314e-01,
                -5.601555803357990415e-01,
            ],
            [
                1.000000000000000000e00,
                0.000000000000000000e00,
                0.000000000000000000e00,
                2.943490077179162778e-01,
            ],
            [
                0.000000000000000000e00,
                0.000000000000000000e00,
                0.000000000000000000e00,
                1.000000000000000000e00,
            ],
        ]
    )

    grasp = np.array(
        [
            [
                0.000000000000000000e00,
                9.008927143266958204e-01,
                4.340418381597311859e-01,
                1.162727682276030183e-01,
            ],
            [
                0.000000000000000000e00,
                -4.340418381597315189e-01,
                9.008927143266959314e-01,
                -5.330387096345654552e-01,
            ],
            [
                1.000000000000000000e00,
                0.000000000000000000e00,
                0.000000000000000000e00,
                2.943490077179162778e-01,
            ],
            [
                0.000000000000000000e00,
                0.000000000000000000e00,
                0.000000000000000000e00,
                1.000000000000000000e00,
            ],
        ]
    )

    pivot = np.array(
        [
            [
                4.340418381597311859e-01,
                9.008927143266958204e-01,
                4.818832423755594462e-17,
                3.019041961584617573e-01,
            ],
            [
                9.008927143266959314e-01,
                -4.340418381597315189e-01,
                1.000191834162551453e-16,
                -1.477440530594816004e-01,
            ],
            [
                1.110223024625156540e-16,
                0.000000000000000000e00,
                -1.000000000000000000e00,
                4.920299628947414905e-01,
            ],
            [
                0.000000000000000000e00,
                0.000000000000000000e00,
                0.000000000000000000e00,
                1.000000000000000000e00,
            ],
        ]
    )

    waypoints = []
    waypoints.append(homogeneous_to_waypoint(approach, 1))
    waypoints.append(homogeneous_to_waypoint(pre_grasp, 1))
    waypoints.append(homogeneous_to_waypoint(grasp, 1))
    waypoints.append(homogeneous_to_waypoint(grasp, 0))  # same pose, grasp only
    waypoints.append(homogeneous_to_waypoint(pivot, 0))
    waypoints.append(homogeneous_to_waypoint(pivot, 1))  # same pose, grasp only

    with client.ServerProxy(f"http://{args.hostname}:9000/") as proxy:
        # check ik of subsampled trajectory
        if proxy.check_ik(waypoints):
            proxy.add_waypoints(waypoints)  # add waypoints to sim
        else:
            logging.warning("IK check failed.")
