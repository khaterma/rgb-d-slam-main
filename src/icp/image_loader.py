"""
This file implements the loading of RGB and depth images from the data that was extracted from the rosbag currently
being processed as specified in the config.json.
"""

import cv2
import open3d as o3d
from utils.data_directory_from_json import get_data_directory_path
from matplotlib import pyplot as plt


def load_rgb_depth(index: int, load_rgb: bool = False, load_depth: bool = False, crop_top: int = 0,
                   greyscale: bool = False, visualize=False):
    """
    Loads rgb and depth image at pose index if selected for loading. If crop_top given it will remove the upper part of
    the image (in pixels). If greyscale will return greyscale image instead of rgb.
    If visualize will show depth image and rgb (greyscale) image before returning.
    """
    path_to_bag_data = get_data_directory_path()
    path_to_rgb = path_to_bag_data / 'rgb' / (str(index) + '_rgb.png')
    path_to_depth = path_to_bag_data / 'depth' / (str(index) + '_depth.tif')

    rgb_img = None
    depth_img = None
    depth_raw = None

    if load_rgb or visualize:
        # ZED cam images come with an alpha channel (transparency) which we don't use and thus remove here.
        color_raw = cv2.imread(str(path_to_rgb), cv2.COLOR_BGRA2BGR)[crop_top:]
        if greyscale:
            color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2GRAY)
        else:
            color_raw = cv2.cvtColor(color_raw, cv2.COLOR_BGR2RGB)
        rgb_img = o3d.geometry.Image(color_raw)

    if load_depth or visualize:
        if depth_raw is None:
            depth_raw = cv2.imread(str(path_to_depth), cv2.IMREAD_UNCHANGED)[crop_top:]
        depth_img = o3d.geometry.Image(depth_raw)

    if visualize:
        plt.subplot(1, 2, 1)
        plt.title(f"RGB image at pose {index}")
        plt.imshow(rgb_img)
        plt.subplot(1, 2, 2)
        plt.title(f"Depth image at pose {index}")
        plt.imshow(depth_img)
        plt.show()

    rgb_img = rgb_img if load_rgb else None
    depth_img = depth_img if load_depth else None

    return rgb_img, depth_img
