"""
This script involves the conversion of odometry and loop closure data in csv files to .g2o dataset file.
This is done since the g2o framework accepts the data in .g2o file format.
"""

import numpy as np
import os
from tf2_py import BufferCore
from rospy import Duration, Time
from pathlib import Path
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion


def create_buffer_of_poses(all_poses: np.array)->BufferCore:
    # buffer where we will store all the sampled poses
    buffer_samples = BufferCore(Duration(10))
    pose = TransformStamped()
    pose.header.frame_id = "world"
    for pose_id in range(len(all_poses)):
        pose_id, x, y, z, quat_x, quat_y, quat_z, quat_w = all_poses[pose_id]

        pose.child_frame_id = str(int(pose_id))
        pose.transform.translation.x = x
        pose.transform.translation.y = y
        pose.transform.translation.z = z
        pose.transform.rotation.x = quat_x
        pose.transform.rotation.y = quat_y
        pose.transform.rotation.z = quat_z
        pose.transform.rotation.w = quat_w
        # set the poses as static transforms
        buffer_samples.set_transform_static(pose, "default_authority")
    return buffer_samples


def get_transform_between_poses(buffer: BufferCore,  num_samples: int) -> np.array:
    # array where all the pose transform is stored
    odom_transform = []
    for i in range(1, num_samples):
        j = i + 1
        # get transform between ith and (i+1)th pose
        t = buffer.lookup_transform_core(str(i), str(j), Time(0))
        x = t.transform.translation.x
        y = t.transform.translation.y
        z = t.transform.translation.z
        quat_x = t.transform.rotation.x
        quat_y = t.transform.rotation.y
        quat_z = t.transform.rotation.z
        quat_w = t.transform.rotation.w
        odom_transform.append([i, j, x, y, z, quat_x, quat_y, quat_z, quat_w])
    return odom_transform 


def write_vertices(poses: np.array, g2o_file):
    for i in range(len(poses)):
        pose_id, x, y, z, quat_x, quat_y, quat_z, quat_w = poses[i]
        roll, pitch, yaw = euler_from_quaternion([quat_x, quat_y, quat_z, quat_w], axes='sxyz') 
        theta = yaw
        pose_id = int(pose_id)
        line = "VERTEX_SE2 " + str(pose_id) + " " + str(x) + " " + str(y) + " " + str(theta)
        g2o_file.write(line)
        g2o_file.write("\n")


def write_odom_edges(transforms: np.array, g2o_file):
    # information matrix for odometry data i.e x, y and theta is choosen randomly
    info_matrix = np.array([1.0, 1.0, 5.0]) * 15
    # Info matrix structure : inverse(cov(x,x)), inverse(cov(x,y)), inverse(cov(x,theta))
    # inverse(cov(y,y)), inverse(cov(y,theta)), inverse(cov(theta,theta))
    info_mat = f"{info_matrix[0]} 0.0 0.0 {info_matrix[1]} 0.0 {info_matrix[2]}"
    for i in range(len(transforms)):
        poseid1, poseid2, del_x, del_y, del_theta, quat_x, quat_y, quat_z, quat_w = transforms[i]
        roll, pitch, yaw = euler_from_quaternion([quat_x, quat_y, quat_z, quat_w], axes='sxyz')
        del_theta = yaw
        line = "EDGE_SE2 " + str(poseid1) + " " + str(poseid2) + " " + str(del_x) + " " + str(del_y) + " " \
               + str(del_theta) + " " + info_mat
        g2o_file.write(line)
        g2o_file.write("\n")


def write_loop_closure_edges(transforms: np.array, g2o_file, info_mats:np.array = None):
    for i in range(0, len(transforms)):
        poseid1, poseid2, del_x, del_y, del_theta, quat_x, quat_y, quat_z, quat_w = transforms[i]
        roll, pitch, yaw = euler_from_quaternion([quat_x, quat_y, quat_z, quat_w], axes='sxyz')
        del_theta = yaw
        # if Information matrix for loop closure edges is extracted from the ICP else use default randomly chosen
        if info_mats is not None:
            info_mat = np.delete(info_mats[i], (0), axis=0)
            line = "EDGE_SE2 " + str(int(poseid1)) + " " + str(int(poseid2)) + " " + str(del_x) + " " + str(del_y) + " " \
                  + str(del_theta) + " " + str(info_mat[0]) + " " + str(info_mat[1]) + " " + str(info_mat[2]) + " " + \
                   str(info_mat[4]) + " " + str(info_mat[5]) + " " + str(info_mat[8])
        else:
            diag = np.array([1.0, 1.0, 1.0])
            info_mat = f"{diag[0]} 0.0 0.0 {diag[1]} 0.0 {diag[2]}"
            line = "EDGE_SE2 " + str(int(poseid1)) + " " + str(int(poseid2)) + " " + str(del_x) + " " + str(del_y) + " " \
                   + str(del_theta) + " " + info_mat
        g2o_file.write(line)
        g2o_file.write("\n")


def gen_g2o_dataset(path_to_csvs: Path) -> Path:
    # Define path to the odometry poses csv
    path_to_pose_csv = str(path_to_csvs / 'poses.csv')
    #  Define path to the loop closures csv
    path_to_loop_closure_csv = str(path_to_csvs / 'loop_closure.csv')
    # extract all the poses from csv and store as an array
    all_poses = np.loadtxt(path_to_pose_csv, skiprows=1, delimiter=';')
    # extract all the loop closures from csv and store as an array
    loop_closures = np.loadtxt(path_to_loop_closure_csv, skiprows=1, delimiter=';')
    # Define path for information matrix from the icp for loop closures
    path_to_info_mat_csv = str(path_to_csvs / 'info_mat.csv')
    if os.path.exists(path_to_info_mat_csv) is True:
        info_mat = np.loadtxt(path_to_info_mat_csv, delimiter=';')
    else:
        info_mat = None
    # Define path to store the gstore the pose graph g2o file
    path_to_gen_g2o = str(path_to_csvs / 'pose_graph.g2o')
    g2o_file = open(path_to_gen_g2o, 'w')
    # Create buffer of poses
    pose_buffer = create_buffer_of_poses(all_poses)
    # get the transforms between two consecutive poses
    odom_transforms = get_transform_between_poses(pose_buffer, len(all_poses))
    # Add all poses as vertex to the g2o file
    write_vertices(all_poses, g2o_file)
    # Add all odometry edges to the g2o file
    write_odom_edges(odom_transforms, g2o_file)
    # Add all loop closure edges based on camera measurements
    write_loop_closure_edges(loop_closures, g2o_file, info_mat)
    # Fix the first pose/vertex of the graph
    g2o_file.write("FIX 1")
    g2o_file.write("\n")

    return Path(path_to_gen_g2o)


