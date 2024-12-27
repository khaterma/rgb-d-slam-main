"""
This script is responsible for converting the .g2o format file to csv file to be further used for point cloud creation.
"""

import numpy as np
from tf2_py import BufferCore
from rospy import Time
from pathlib import Path
import csv
import os
from tf.transformations import quaternion_from_euler, quaternion_matrix
from src.g2o.csv_to_g2o_dataset import create_buffer_of_poses


def extract_pose_from_g2o(g2o_file):
    A = g2o_file.readlines()
    g2o_file.close()
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


def get_poses(X, Y, THETA):
    all_poses = []
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        z = 0.0

        array = quaternion_from_euler(0.0, 0.0, THETA[i], axes='sxyz')
        quat_x, quat_y, quat_z, quat_w = array

        all_poses.append([i, x, y, z, quat_x, quat_y, quat_z, quat_w])

    return all_poses


def get_transform_between_poses(buffer: BufferCore,  num_samples: int, all_poses:np.array ) -> np.array:
    odom_transform = []
    for i in range(0, num_samples-1):
        j = i + 1
        t = buffer.lookup_transform_core(str(i), str(j), Time(0))
        x = t.transform.translation.x
        y = t.transform.translation.y
        z = t.transform.translation.z
        quat_x = t.transform.rotation.x
        quat_y = t.transform.rotation.y
        quat_z = t.transform.rotation.z
        quat_w = t.transform.rotation.w
        pose_id, xp, yp, zp, quat_xp, quat_yp, quat_zp, quat_wp = all_poses[i]
        odom_transform.append([i, j, x, y, z, quat_x, quat_y, quat_z, quat_w , xp, yp , zp, quat_xp, quat_yp,quat_zp, quat_wp])

    return odom_transform


def store_transforms_to_csv(pose_transforms, path_to_csv: Path) -> None:
    f_transforms = open(path_to_csv, 'w')
    writer = csv.writer(f_transforms, delimiter= ';')
    header = ['pose_id1', 'pose_id2', 'x', 'y', 'z', 'quat_x', 'quat_y', 'quat_z', 'quat_w' , 'pose_x', 'pose_y' ,
              'pose_z', 'pose_rotx', 'pose_roty', 'pose_rotz', 'pose_rotw']
    writer.writerow(header)
    for i in range(0, len(pose_transforms)):
        poseid1, poseid2, del_x, del_y, del_theta, quat_x, quat_y, quat_z, quat_w , xp , yp , zp, quat_xp, quat_yp, quat_zp, quat_wp = pose_transforms[i]
        writer.writerow([ poseid1, poseid2, del_x, del_y, del_theta, quat_x, quat_y, quat_z, quat_w ,xp, yp , zp, quat_xp, quat_yp,quat_zp, quat_wp])
    f_transforms.close()
    f_transforms.close()


def store_opt_pose_tfs_to_csv(path_to_optimized_g2o: Path):
    # Define the path to store the csv file
    path_to_csv = str(os.path.dirname(path_to_optimized_g2o) + '/optimized_transforms.csv')
    opt_g2o_file = open(path_to_optimized_g2o, 'r')
    # extract 2D coordinates of the pose from the g2o file
    x_opt, y_opt, theta_opt = extract_pose_from_g2o(opt_g2o_file)
    # convert the 2d coordinates to 3d for all the poses
    all_poses = get_poses(x_opt, y_opt, theta_opt)
    # create buffer of all the poses
    buffer = create_buffer_of_poses(all_poses)
    # get the transforms betwwen the poses consecutive poses
    transforms = get_transform_between_poses(buffer, len(all_poses), all_poses)
    # Store the transforms into csv file
    store_transforms_to_csv(transforms, Path(path_to_csv))





