"""
Holds names of all relevant frames used throughout the code. As well as the path to the rosbag to use. A yaml would be
more appropriate for this use case and might still be implemented if time
"""

from pathlib import Path

PATH_BAG = Path(__file__).parent.parent.__str__() + '/rosbags/' + 'dataset_final.bag'

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
