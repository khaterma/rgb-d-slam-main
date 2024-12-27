"""
The class FreiCAR2 in this file is used to access the frame names and all relevant transforms throughout the pipeline.
With more time one could certainly make this more consistent and better understandable.
"""

from geometry_msgs.msg import TransformStamped
from tf.transformations import quaternion_from_euler
from math import pi
from utils.data_directory_from_json import get_bag_name
import rospy
from rosbag import Bag
from tf2_py import BufferCore
from pathlib import Path


class FreiCAR2:
    """
    Class that holds relevant information about relative positions of components on FreiCAR and their frames.
    Namely VIVE tracker is given in world coordinates, whereas T265 has its own reference frame that needs to be
    transformed into the world frame.
    """

    # all frames used for ground truth and debugging
    # VIVE is also used to get initial position of robot
    VIVE_GT = "freicar_2"
    T265_POSE_GT = "freicar_2/t265_ref"
    ZED_CENTER_GT = "freicar_2/zed_ref"

    # frames that will be actually used for mapping
    WORLD = "world"
    T265_ODOM_FRAME = "freicar_2/t265_odom_frame"
    T265_POSE_FRAME = "freicar_2/t265_pose_frame"
    ZED_CENTER_LAST = "zed_last_pose"
    ZED_CENTER = "base_link"
    ZED_LEFT = "freicar_2/zed_left_camera_frame"
    ZED_LEFT_OPT = 'freicar_2/zed_left_camera_optical_frame'

    # define transform between VIVE tracker and T265
    t_vive_t265 = TransformStamped()
    t_vive_t265.header.stamp = rospy.Time(0)
    t_vive_t265.header.frame_id = VIVE_GT
    t_vive_t265.child_frame_id = T265_POSE_FRAME
    t_vive_t265.transform.translation.x = -51.3 * 1e-2
    t_vive_t265.transform.translation.y = 0
    t_vive_t265.transform.translation.z = -11.3 * 1e-2
    quat = quaternion_from_euler(0, 0.75 * pi, pi, axes="szyx")
    t_vive_t265.transform.rotation.x = quat[0]
    t_vive_t265.transform.rotation.y = quat[1]
    t_vive_t265.transform.rotation.z = quat[2]
    t_vive_t265.transform.rotation.w = quat[3]

    # define transform between VIVE tracker and ZED camera center
    t_vive_zed_center = TransformStamped()
    t_vive_zed_center.header.stamp = rospy.Time(0)
    t_vive_zed_center.header.frame_id = VIVE_GT
    t_vive_zed_center.child_frame_id = ZED_CENTER
    t_vive_zed_center.transform.translation.x = 3.3e-2
    t_vive_zed_center.transform.translation.y = 0
    t_vive_zed_center.transform.translation.z = -8e-2
    t_vive_zed_center.transform.rotation.x = 0
    t_vive_zed_center.transform.rotation.y = 0
    t_vive_zed_center.transform.rotation.z = 0
    t_vive_zed_center.transform.rotation.w = 1

    # look up the transform between the T265 and ZED center to be able to use it independently from the VIVE tracker
    _temp = BufferCore(rospy.Duration(2000000000))
    _temp.set_transform_static(t_vive_t265, "default_authority")
    _temp.set_transform_static(t_vive_zed_center, "default_authority")
    t_t265_zed_center = _temp.lookup_transform_core(T265_POSE_FRAME, ZED_CENTER, rospy.Time(1))

    @staticmethod
    def get_reference_vive_t265(reference_frame_name: str) -> TransformStamped:
        """
        Returns a transform to a frame that can be used as reference for plotting
        """
        t = FreiCAR2.t_vive_t265
        t.child_frame_id = reference_frame_name
        return t

    @staticmethod
    def get_reference_vive_zed_center(reference_frame_name: str) -> TransformStamped:
        """
        Returns a transform to a frame that can be used as reference for plotting
        """
        t = FreiCAR2.t_vive_zed_center
        t.child_frame_id = reference_frame_name
        return t

    @staticmethod
    def get_transform_world_odom_frame() -> TransformStamped:
        """
        Computes relative transform between world frame and odom frame of t265, such that t265 moves correctly in world
        frame according to its relative position to VIVE  tracker at beginning of recording.
        """

        # GOAL: build tf tree: world -> VIVE -> T265_pose -> T265_odom to look up world -> T265_odom
        # 1. look up transforms world -> VIVE and T265_pose -> T265_odom
        buffer_init = FreiCAR2.get_vive_and_t265_in_frames()
        t_world_vive = buffer_init.lookup_transform_core(FreiCAR2.WORLD, FreiCAR2.VIVE_GT, rospy.Time(1))
        t_pose_odom = buffer_init.lookup_transform_core(FreiCAR2.T265_POSE_FRAME, FreiCAR2.T265_ODOM_FRAME, rospy.Time(1))

        # 2. build the tree and look up the transform
        buffer_tree = BufferCore(rospy.Duration(10))
        buffer_tree.set_transform_static(t_world_vive, "default_authority")
        buffer_tree.set_transform_static(FreiCAR2.t_vive_t265, "default_authority")
        buffer_tree.set_transform_static(t_pose_odom, "default_authority")

        return buffer_tree.lookup_transform_core(FreiCAR2.WORLD, FreiCAR2.T265_ODOM_FRAME, rospy.Time(1))

    @staticmethod
    def get_vive_and_t265_in_frames() -> BufferCore:
        """
        returns BufferCore with transforms of VIVE tracker and T265 in their reference frames. Usually reference frame
        T265 needs to be assigned a transform into world frame for initial positioning and comparison of odometry and
        ground truth data. In short returns buffer with transforms:
        WORLD -> VIVE
        T265_ODOM_FRAME -> T265_POSE_FRAME
        Transforms are static and timed at rospy.Time(0).
        """
        bag = Bag(Path(__file__).parent.parent / 'rosbags' / get_bag_name(), 'r')
        added_world_vive = False
        added_odom_pose = False
        buffer = BufferCore(rospy.Duration(10))

        # iterate over messages in Bag until the required transforms have been found.
        for topic, msg, t in bag.read_messages(['/tf']):
            for transform in msg.transforms:
                if transform.header.frame_id == FreiCAR2.WORLD and transform.child_frame_id == FreiCAR2.VIVE_GT:
                    transform.header.stamp = rospy.Time(0)
                    buffer.set_transform_static(transform, "default_authority")
                    added_world_vive = True
                elif (transform.header.frame_id == FreiCAR2.T265_ODOM_FRAME and
                      transform.child_frame_id == FreiCAR2.T265_POSE_FRAME):
                    transform.header.stamp = rospy.Time(0)
                    buffer.set_transform_static(transform, "default_authority")
                    added_odom_pose = True
                else:
                    continue

            if added_world_vive and added_odom_pose:
                break

        return buffer

    @staticmethod
    def get_quaternion_zed_left_to_zed_left_optical():
        t = TransformStamped()

        t.header.frame_id = FreiCAR2.ZED_LEFT
        t.header.stamp = rospy.Time(0)
        t.child_frame_id = FreiCAR2.ZED_LEFT_OPT

        t.transform.translation.x = 0
        t.transform.translation.y = 0
        t.transform.translation.z = 0

        t.transform.rotation.x = 0.5
        t.transform.rotation.y = -0.5
        t.transform.rotation.z = 0.5
        t.transform.rotation.w = -0.5

        return t.transform.rotation

    @staticmethod
    def get_zed_camera_intrinsic_param():
        """
        Returns the ZED left camera intrinsic parameters
        """
        fx = 527.55
        fy = 527.305
        cx = 631.195
        cy = 355.3865
        k1 = -0.0401327
        k2 = 0.00847428

        return fx, fy, cx, cy, k1, k2
