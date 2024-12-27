"""
Running the 4 functions as given in this script starting with a rosbag will result in an optimized graph and a point
cloud based on the optimized poses can be created.
Prerequisite is, that all the transforms from VIVE tracker, T265 and ZED camera are in the rosbag and that the names of
the frames match with the ones given in utils.FreiCAR2_representation.
"""

from icp.matching import loop_closure_detection_and_icp
from src.g2o.run_optimizer import run_optimization
from icp.aligned_pcd import fuse_point_clouds
from frontend.create_graph_from_bag import create_graph_from_bag


# sample poses  that will be used for graph optimization and extract and store the related RGB and depth images
create_graph_from_bag(plotting=True)

# run ORB to detect loop closures in the greyscale images and RANSAC+ICP to find transforms at loop closures
# the function arguments allow to configure parameters of the algorithms and visualize intermediate results
# will store the transforms in the data folder
loop_closure_detection_and_icp(plot_error_dist_mat=True, plot_selected_loop_closures=False, plot_matched_images=False,
                               skip_images=0)

# run g2o optimizer, might need to adapt the information matrices in g2o_pose_transforms_to_csv.py
run_optimization(plot_pose_graphs=True)

# show fused point cloud based on optimized poses, optionally limit point cloud along subtrajectory.
fuse_point_clouds(voxel_size=0.1, show_img_start_end_pose=True)
