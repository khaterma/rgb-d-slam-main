"""
In this file an implementation to fuse the point clouds based on the poses either from odometry alone or after graph
optimization is given.
"""

# cv2 throws error when imported after open3d thus we always import it before open3d even if not using it.
import cv2
import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from icp.ICP import create_ptcloud
from tqdm import tqdm
from tf.transformations import quaternion_matrix
from utils.data_directory_from_json import get_data_directory_path
from icp.image_loader import load_rgb_depth


def fuse_point_clouds(start_idx: int = None, end_idx: int = None, voxel_size: float = 0.01, min_x: float = 0,
                      max_x: float = 6, min_y: float = -np.inf, max_y: float = np.inf, min_z: float = 0,
                      max_z: float = np.inf, show_before_optim: bool = False, show_img_start_end_pose: bool = False):
    """
    Fuses all point clouds from start_idx to end_idx pose. The boundaries for each point clouds contributions can be
    specified.

    :param start_idx: First pose to include in point cloud fusing.
    :param end_idx: Last pose to include in point cloud fusing
    :param voxel_size: For downsampling, higher values translate to more downsampling. If fusing more pointclouds should
    be chosen higher.
    :param min_x: lower x limit of single point cloud contribution in ZED left camera frame (depth)
    :param max_x: upper x limit of single point cloud contribution in ZED left camera frame (depth)
    :param min_y: lower y limits of single point cloud contribution in ZED left camera frame (width)
    :param max_y: upper y limits of single point cloud contribution in ZED left camera frame (width)
    :param min_z: lower z limits of single point cloud contribution in ZED left camera frame (height)
    :param max_z: upper z limits of single point cloud contribution in ZED left camera frame (height)
    :param show_before_optim: if True align point clouds based on odometry poses instead of optimized graph
    :param show_img_start_end_pose: if True displays RGB image from first and last pose before showing fused point cloud
    :return: None
    """
    if show_before_optim:
        path_to_pose_transform_csv = str(get_data_directory_path()) + '/poses.csv'
        all_poses_transforms = np.loadtxt(path_to_pose_transform_csv, skiprows=1, delimiter=';')[:, 1:]
    else:
        path_to_pose_transform_csv = str(get_data_directory_path()) + '/optimized_transforms.csv'
        all_poses_transforms = np.loadtxt(path_to_pose_transform_csv, skiprows=1, delimiter=';')[:, 9:]

    len_dataset = len(all_poses_transforms)   # no. of poses

    if start_idx is None:
        start_idx = 1
    elif start_idx > len_dataset:
        print(f"start_idx is greater than the number of poses. E.g. there exists no such pose. There are {len_dataset}"
              f"poses.")
        return

    if end_idx is None:
        end_idx = len_dataset
    elif end_idx > len_dataset:
        print(f"end_idx is greater than the number of poses. E.g. there exists no such pose. There are {len_dataset}"
              f"poses. Limiting the poses included to up to pose {len_dataset}.")
        end_idx = len_dataset

    # Build the combined point cloud by transforming the point clouds into the world frame and fusing them.
    pcd_combined = o3d.geometry.PointCloud()
    for i in tqdm(range(start_idx, end_idx + 1)):
        pose_matrix = get_transform_matrix(all_poses_transforms[i-1])
        pcd = create_ptcloud(i, min_x, max_x, min_y, max_y, min_z, max_z)
        pcd.transform(pose_matrix)
        pcd_combined += pcd     # fuse pcd operator

    # show img at initial pose before displaying point cloud
    if show_img_start_end_pose:
        rgb, _ = load_rgb_depth(start_idx, True)
        plt.imshow(rgb)
        plt.show()
        rgb, _ = load_rgb_depth(end_idx, True)
        plt.imshow(rgb)
        plt.show()
    # downsample the fused point cloud and display it w it a coordinate frame.
    pcd_combined_down = pcd_combined.voxel_down_sample(voxel_size=voxel_size)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([pcd_combined_down, mesh_frame])


def get_transform_matrix(poses_transform: np.array):
    """
    Helper function that computes transformation matrix from world frame to pose as specified in poses_transform.
    :param poses_transform: array of len 7: [x, y, z, quaternion_x, quaternion_y, quaternion_z, quaternion_w]
    """
    r_matrix = quaternion_matrix(poses_transform[3:])
    pose_matrix = r_matrix
    pose_matrix[0, 3] = poses_transform[0]
    pose_matrix[1, 3] = poses_transform[1]
    pose_matrix[2, 3] = poses_transform[2]

    return pose_matrix
