"""
This file implements the feature extraction required for RANSAC as well as global registration using RANSAC.
This implementation including all parameter settings is taken over from the open3d tutorial:
http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html
"""
import open3d as o3d


def downsample_point_cloud(pcd, voxel_size):
    """
    Downsamples the pointcloud and computes point feature histograms for this downsampled pointcloud.
    Point Feature Histograms characterize points by the curvature properties of their neighborhood.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)

    # compute the normal estimate of point based on neighborhooed size radius_normal
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # compute features per point based on neighborhood of size radius_feature
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    """
    Find a transform from source to target point cloud using RANSAC and using the point feature histograms provided.
    """
    distance_threshold = voxel_size * 1.5
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result


def initial_transform_from_ransac(source, target):
    """
    Takes the provided point clouds, downsamples them and finds a approximate transformation from source to target based
    on geometric features using RANSAC (global registration).
    """
    # downsample point clouds and extract features.
    source_down, source_fpfh = downsample_point_cloud(source, 0.05)
    target_down, target_fpfh = downsample_point_cloud(target, 0.05)

    # run RANSAC on these and extract the transformation form the result.
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, 0.05)

    return result_ransac.transformation
