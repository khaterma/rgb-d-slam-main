#!/usr/bin/env python

"""
Simple node that publishes the transform between the world frame and the t265_odom_frame for visualization of the T265
camera pose in RVIZ.
Additionally publishes transforms to visualize ZED camera frames relative to VIVE (as GT) and to T265 (as estimated pose
based on odometry).
"""

import rospy
import tf2_ros
from utils.freicar_representation import FreiCAR2

rospy.init_node('my_static_tf2_broadcaster')
broadcaster = tf2_ros.StaticTransformBroadcaster()

# get transform between world and odometry frame to be able to visualize T265 pose in world frame.
broadcaster.sendTransform(FreiCAR2.get_transform_world_odom_frame())

# transform between VIVE and reference frame for T265 as GT reference, dito for ZED camera
broadcaster.sendTransform(FreiCAR2.get_reference_vive_t265(FreiCAR2.T265_POSE_GT))
broadcaster.sendTransform(FreiCAR2.get_reference_vive_zed_center(FreiCAR2.ZED_CENTER_GT))

# transform to visualize ZED center pose estimate relative to T265
broadcaster.sendTransform(FreiCAR2.t_t265_zed_center)
rospy.spin()
