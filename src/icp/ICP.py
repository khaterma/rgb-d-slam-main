"""
In this file we implement point cloud creation from RGBD images and ICP on point clouds.
We based our implementations on the tutorials from open3d, so there might still be some lines / variable names identical
to these.
Function draw_registration_result was taken over completely and we added a coordinate frame.
http://www.open3d.org/docs/latest/tutorial/Basic/rgbd_image.html
http://www.open3d.org/docs/latest/tutorial/Basic/icp_registration.html
"""


# cv2 throws error when imported after open3d thus we always import it before open3d even if not using it.
import cv2
import open3d as o3d
import numpy as np
from icp.image_loader import load_rgb_depth
from icp.ransac import initial_transform_from_ransac
from utils.freicar_representation import FreiCAR2
import copy


def create_ptcloud(i, min_x: float = 0, max_x: float = 50, min_y: float = -np.inf, max_y: float = np.inf,
                   min_z: float = -np.inf, max_z: float = np.inf, visualize: bool = False) -> o3d.geometry.PointCloud:
    """
    Creates point cloud at pose i of the ZED camera. The point cloud will be in the left camera frame of the ZED camera
    (not in the optical frame). The point cloud is cropped according the chosen boundaries (in m). Boundaries are also
    according to ZED camera frame, e.g. x: depth, y: width, z: height.
    """
    rgb_img, depth_img = load_rgb_depth(i, True, True)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, depth_scale=1,
                                                                    depth_trunc=max_x, convert_rgb_to_intensity=False)

    # set intrinsic parameters of the ZED camera
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, o3d.camera.PinholeCameraIntrinsic(1920, 1080, 1058.4301, 1058.2400, 948.3100, 534.7870))

    # transform point cloud from the optical frame to the left camera frame (which is our reference also for odometry)
    quaternion = FreiCAR2.get_quaternion_zed_left_to_zed_left_optical()
    quaternion = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w])
    r_matrix = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
    t_matrix = np.zeros((4, 4))
    t_matrix[:3, :3] = r_matrix
    t_matrix[3, 3] = 1
    pcd.transform(t_matrix)

    # crop the point cloud according to specified boundary values
    box = o3d.geometry.AxisAlignedBoundingBox(np.array([min_x, min_y, min_z]), np.array([np.inf, max_y, max_z]))
    pcd = pcd.crop(box)
    if visualize:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, mesh_frame])

    return pcd


def icp_registration(i, j, min_x: float = 0, max_x: float = 50, min_y: float = -np.inf, max_y: float = np.inf,
                     min_z: float = 0, max_z: float = 3, visualize: bool = False):
    """
    Computes transformations between the point clouds at poses i and j using ICP. An initial transform is obtained using
    RANSAC. Point clouds will be cropped according to chosen boundaries (in m) before alignment. Boundaries are
    according to ZED camera frame, e.g. x: depth, y: width, z: height.
    If visualize is selected alignment of point clouds will be displayed before and after RANSAC+ICP.
    """
    source = create_ptcloud(i, min_x, max_x, min_y, max_y, min_z, max_z, False)
    target = create_ptcloud(j, min_x, max_x, min_y, max_y, min_z, max_z, False)

    # obtain initial transform from RANSAC
    trans_init = initial_transform_from_ransac(source, target)

    # Perform ICP registration using point-to-plane, for this we need to estimate the surface normals
    source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=100))
    target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=100))
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    result = o3d.pipelines.registration.registration_icp(
        source, target, 0.05, trans_init, estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))

    tf = result.transformation
    err = result.inlier_rmse
    information_matrix = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source, target, 0.01, tf)

    if visualize:
        draw_registration_result(source, target, tf, f"Point clouds at {i} and {j} aligned")
        draw_registration_result(source, target, np.eye(4), f"Point clouds for {i} {j} before alignment")

    return tf, err, information_matrix


def draw_registration_result(source, target, transformation, name="TBD"):
    """
    Displays the given point clouds, source transformed accordint to given transformation.
    Source is painted in yellow and target in blue.
    """
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([source_temp, target_temp, mesh_frame], window_name=name)
