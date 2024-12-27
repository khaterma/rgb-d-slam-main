"""
This simple script plots the ground truth, unoptimized and optimized pose graphs.
"""

import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


def get_poses(g2o_file):
    file = open(g2o_file)
    A = file.readlines()
    file.close()
    X = []
    Y = []
    THETA = []
    for line in A:
        if "VERTEX_SE2" in line:
            (ver, pose_id, x, y, theta) = line.split(' ')
            X.append(float(x))
            Y.append(float(y))
            THETA.append(float(theta.rstrip('\n')))

    return X, Y, THETA


def get_ground_truth(path: Path):
    data = np.loadtxt(str(path), delimiter=';', skiprows=1)
    X, Y = data.transpose((1, 0))
    return X, Y


def plot_g2o_graphs(path_to_csvs):
    # extract the x and y coordinates from the ground truth csv
    X_gt, Y_gt = get_ground_truth(path_to_csvs / 'poses_gt.csv')
    # extract the unoptimized pose graph poses of the camera odometry
    X, Y, _ = get_poses(path_to_csvs / 'pose_graph.g2o')
    # extract the optimized 2d coordinates of the graph poses
    X_opt, Y_opt, THETA_opt = get_poses(path_to_csvs / 'optimized_pose_graph.g2o')
    # Plot them respectively
    plt.figure()
    plt.axis('equal')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.grid()
    plt.plot(X, Y, '--', color='orange', label="Unoptimized_Graph")
    plt.plot(X_gt, Y_gt, '--',  color='yellowgreen', label='Ground_Truth')
    plt.plot(X_opt, Y_opt, color='#1f77b4', linewidth=2.0, label="Optimized_Graph")
    plt.legend()
    plt.show()
