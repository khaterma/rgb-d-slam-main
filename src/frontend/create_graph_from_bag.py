"""
In this file we implement the extraction of all relevant data from the rosbag. This is the sampled poses based on our
odometry estimates and RGBD scans (RGB image + depth image) of the ZED camera at these poses. The poses will be stored
in poses.csv, the images in respective folders in a folder created in src/data for the selected bag in config.json.
The rough approach can be characterized as follows:
1. check current pose against previous pose, if threshold for translation or rotation is exceeded...
2. ...start looking for new pair of RGB and depth image.
3. After RGB and depth image where found pick the next pose as camera pose from which these images where taken.

REMARKS:
1. Since we will register the point clouds in the left camera frame for scan matching we are also storing the poses of
the left camera frame in the .csv for later graph optimization. However we are evaluating the thresholds and storing the
ground truth in the ZED center frame. As a refactoring of the code it could make sense to register the point clouds in
the ZED camera center frame to reduce the number of frames that need to be handled.
2. Also currently only the translation threshold criterion is actually being used.
3. Currently storing the ground truth poses is handled by the plotting manager, because this feature was only
implemented late and it was the fastest way to do this.
In general there is certainly room for improvement to make the code less complex, break it better up into components.
"""


from cv_bridge import CvBridge
from rosbag import Bag
import numpy as np
from tf2_py import BufferCore
from rospy import Duration, Time
import os
from pathlib import Path
import csv
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image
from utils.plot_poses_mgr import PlotPosesMgr
import cv2
import tqdm
from utils.freicar_representation import FreiCAR2
from utils.data_directory_from_json import get_data_directory_path, get_bag_name, get_thresholds


def create_graph_from_bag(plotting=False):
    """
    Function that extracts poses of the ZED left camera frame based on thresholds for robot translation and rotation as
    specified in config.json and the related RGBD data at these poses.
    Optionally the estimated odometry basd trajectory and the ground truth trajectory of the VIVE tracker can be plotted
    """
    # define limit after which a new pose and scan sample should be taken.
    lim_transl, lim_rot = get_thresholds()

    path_to_data = get_data_directory_path()

    # If the data directory does not exist at all.
    if not os.path.exists(path_to_data.parent):
        os.mkdir(path_to_data.parent)

    # if the data directory has not yet been created, only then we store the extracted data from the bag
    # E.g. we assume the data to be extracted in one sweep.
    if not os.path.exists(path_to_data):
        os.mkdir(path_to_data)
        os.mkdir(path_to_data / 'rgb')
        os.mkdir(path_to_data / 'depth')
        store_data = True

    # we might only want to plot the trajectory and the sampled poses
    elif plotting:
        store_data = False
    else:
        print("Nothing to do. The data is there and you don't want to see the plot. So lets just relax.")
        return

    path_to_bag = Path(__file__).parent.parent / 'rosbags' / get_bag_name()
    bag = Bag(path_to_bag, 'r')

    # buffer where we will store all the sampled poses (transform world -> ZED left camera frame)
    buffer_samples = BufferCore(Duration(10))

    # with the plot_mgr we collect all the data we want to plot (e.g. trajectory based on odometry and based on ground
    # truth). Because the plot_mgr tracks the ground truth poses from VIVE anyways we also use it to store the ground
    # truth poses in a csv after data extraction
    plot_mgr = PlotPosesMgr()

    # Setup a buffer with a transform between the world frame and the odom frame which is reference for the T265 poses.
    # Also set transforms between the ZED camera center and T265 pose (used for the graph) and the ZED camera center
    # and VIVE (used as ground truth reference).
    buffer = BufferCore(Duration(bag.get_end_time() + 1))
    buffer.set_transform_static(FreiCAR2.get_transform_world_odom_frame(), "default_authority")
    buffer.set_transform_static(FreiCAR2.get_reference_vive_zed_center(FreiCAR2.ZED_CENTER_GT), "default_authority")
    buffer.set_transform_static(FreiCAR2.t_t265_zed_center, "default_authority")

    # first read out all static transforms from the bag to use them right from the beginning.
    for topic, msg, rec_time in bag.read_messages(['/tf_static']):
        for t in msg.transforms:
            buffer.set_transform_static(t, "default_authority")

    # because we decide on sampling new poses based on distance to previous pose we need to add a frame that represents
    # the last sampled frame to the buffer. Also we only start reading the bag messages after the time the first trans-
    # form for T265 was published.
    start_time = add_last_sampled_pose_frame_to_buffer(bag, buffer)   # start only after the first pose was published

    # since pose ('/tf') messages are published with higher frequencies than ZED scans we first sample a scan and
    # afterwards pick the next pose as camera pose for the scan. For this we use the looking_for_pose and the
    # sample_scans flag. We also keep track of the numbers of sampled poses / scans to enumerate them.
    looking_for_pose = False
    num_scans = 1

    # store the first scan and set the and set looking_for_pose=True to pick the next transform as related transform.
    rgb, depth = sample_pair_of_scans_with_pose(bag, start_time)
    last_scan_stamp = rgb.header.stamp
    if store_data:
        store_image(path_to_data / 'rgb' / (str(num_scans) + '_rgb.png'), rgb)
        store_image(path_to_data / 'depth' / (str(num_scans) + '_depth.tif'), depth)
    looking_for_pose = True

    print('########## PROCESSING ROBOT TRAJECTORY ##########')
    with tqdm.tqdm(total = bag.get_message_count(['/tf'])) as pbar:
        for topic, msg, rec_time in bag.read_messages(['/tf'], start_time + Duration(0, 1)):
            for t in msg.transforms:

                # add all transforms to the buffer
                buffer.set_transform(t, 'default_authority')

                # if it is an update of the T265 pose check if we moved far enough to sample a new scan or pick the pose
                # if we just stored new scans from the ZED cam.
                if t.header.frame_id == FreiCAR2.T265_ODOM_FRAME and t.child_frame_id == FreiCAR2.T265_POSE_FRAME:

                    # use helper fct to check if we moved far enough such that we should select a new scan from ZED cam
                    sample_scan = process_odom_update(t, buffer, lim_transl, lim_rot, plot_mgr, looking_for_pose)

                    if sample_scan:
                        # use helper function to get next RGB and depth image pair given the current point in time
                        rgb, depth = sample_pair_of_scans_with_pose(bag, rec_time)

                        # store time when the scan of the ZED cam was taken because we will use it to pick the first
                        # pose update of the T265 cam after the scan as related pose.
                        last_scan_stamp = rgb.header.stamp
                        if store_data:
                            store_image(path_to_data / 'rgb' / (str(num_scans) + '_rgb.png'), rgb)
                            store_image(path_to_data / 'depth' / (str(num_scans) + '_depth.tif'), depth)

                        # set the flag to start looking for a pose
                        looking_for_pose = True

                    # if we are looking for a pose and the transform was published after the last chosen scan we store
                    # the current transform between the world frame and ZED left camera frame
                    elif looking_for_pose and t.header.stamp >= last_scan_stamp:
                        pose = buffer.lookup_transform_core(FreiCAR2.WORLD, FreiCAR2.ZED_LEFT, t.header.stamp)

                        # enumerate the pose and add them to the buffer used to store all the sampled poses
                        pose.child_frame_id = str(num_scans)
                        buffer_samples.set_transform_static(pose, 'default_authority')
                        pose = buffer.lookup_transform_core(FreiCAR2.WORLD, FreiCAR2.ZED_CENTER, t.header.stamp)

                        # update the last pose where a scan was taken.
                        pose.child_frame_id = FreiCAR2.ZED_CENTER_LAST
                        buffer.set_transform_static(pose, 'default_authority')
                        if plotting:
                            plot_mgr.add_sampled_pose(pose)
                        looking_for_pose = False
                        num_scans += 1
                elif t.header.frame_id == FreiCAR2.WORLD and t.child_frame_id == FreiCAR2.VIVE_GT:
                    plot_mgr.add_pose(t, is_gt=True)
                else:
                    continue
            pbar.update(1)

    if store_data:
        store_poses_from_buffer(buffer_samples, path_to_data / 'poses.csv', num_scans - 1)
        plot_mgr.store_gt(path_to_data / 'poses_gt.csv')

    if plotting:
        plot_mgr.create_plot()


def sample_pair_of_scans_with_pose(bag: Bag, start_time: Time):
    """
    Returns next pair of scans consisting of a rgb and depth image that was published after start time.
    """
    found_rgb = False
    found_depth = False
    for topic, msg, rec_time in bag.read_messages(['/freicar_2/zed/left/image_rect_color',
                                                   '/freicar_2/zed/depth/depth_registered'], start_time=start_time):
        # TODO: Avoid infinite alternation between searching for rgb and depth scans in case the timestamps are
        # misaligned for some reason
        # ZED assigns same time stamp to rgb and depth image that were taken together. Thus we will check that the time
        # stamps correspond to each other before returning the pair of scans.
        if topic == '/freicar_2/zed/left/image_rect_color':
            candidate_rgb = msg
            stamp_rgb = msg.header.stamp
            found_rgb = True
            if found_depth and stamp_rgb > stamp_depth:
                found_depth = False
        elif topic == '/freicar_2/zed/depth/depth_registered':
            candidate_depth = msg
            stamp_depth = msg.header.stamp
            found_depth = True
            if found_rgb and stamp_depth > stamp_rgb:
                found_rgb = False

        if found_rgb and found_depth:
            break
    return candidate_rgb, candidate_depth


def process_odom_update(t: TransformStamped, buffer: BufferCore, lim_transl, lim_rot, plot_mgr: PlotPosesMgr = None,
                        looking_for_pose = True) -> bool:
    """
    Helper function that stores current pose of ZED camera for plotting and checks if we have moved far enough that we
    need to sample a new pose and RGB and depth image.
    """
    new_pose = buffer.lookup_transform_core(FreiCAR2.WORLD, FreiCAR2.ZED_CENTER, t.header.stamp)
    if plot_mgr is not None:
        plot_mgr.add_pose(new_pose)
    # check if we have moved far enough to get a new sample.
    if looking_for_pose:
        return False
    transf_to_last = buffer.lookup_transform_core(FreiCAR2.ZED_CENTER_LAST, FreiCAR2.ZED_CENTER, t.header.stamp)
    transl_x = transf_to_last.transform.translation.x
    transl_y = transf_to_last.transform.translation.y
    return np.sqrt(transl_x ** 2 + transl_y ** 2) > lim_transl


def store_poses_from_buffer(buffer: BufferCore, path_to_csv: Path, num_samples: int) -> None:
    """
    Helper function that stores all sampled poses of the ZED left camera frame to a csv.
    """
    f = open(path_to_csv, 'w')

    writer = csv.writer(f, delimiter= ';')

    header = ['pose_id', 'x', 'y', 'z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    writer.writerow(header)
    for i in range(1, num_samples + 1):
        t = buffer.lookup_transform_core(FreiCAR2.WORLD, str(i), Time(0))
        x = t.transform.translation.x
        y = t.transform.translation.y
        z = t.transform.translation.z
        quat_x = t.transform.rotation.x
        quat_y = t.transform.rotation.y
        quat_z = t.transform.rotation.z
        quat_w = t.transform.rotation.w

        writer.writerow([i, x, y, z, quat_x, quat_y, quat_z, quat_w])

    f.close()


def store_image(path_to_store: Path, msg: Image):
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    cv2.imwrite(str(path_to_store), img)


def add_last_sampled_pose_frame_to_buffer(bag: Bag, buffer: BufferCore):
    """
    Adds a frame for the last sampled pose of the ZED center to the buffer. To set the transform the first published
    transform of the T265 (which is parent of the ZED camera) will be used.
    """
    for topic, message, rec_time in bag.read_messages(['/tf']):
        for t in message.transforms:
            buffer.set_transform(t, 'default_authority')
            if t.header.frame_id == FreiCAR2.T265_ODOM_FRAME and t.child_frame_id == FreiCAR2.T265_POSE_FRAME:
                init_pose = buffer.lookup_transform_core(FreiCAR2.WORLD, FreiCAR2.ZED_CENTER, t.header.stamp)
                init_pose.child_frame_id = FreiCAR2.ZED_CENTER_LAST
                buffer.set_transform_static(init_pose, 'default_authority')
                return rec_time
