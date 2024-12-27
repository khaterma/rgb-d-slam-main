"""
In this file we do loop-closure detection using ORB feature matching after converting the RGB images to greyscale.
After detecting loop-closures we call the ICP implementation as given in icp/ICP.py to find relative transforms between
the point clouds and store these in loop_closure.csv in the data folder for the related bag.
The implementation of ORB is, apart from the comments, identical to: https://github.com/cohnt/ICP-SLAM-with-Loop-Closure

REMARK:
An improvement to this approach would be to store results from loop closure detection (ORB) first and run the ICP in a
separate function.
"""

import csv
import numpy as np
import cv2
from tqdm import tqdm
import os
from joblib import Parallel, delayed
from pathlib import Path
import matplotlib.pyplot as plt
from tf.transformations import euler_from_quaternion, quaternion_from_matrix
from src.icp.image_loader import load_rgb_depth
from src.icp.ICP import icp_registration
from src.utils.data_directory_from_json import get_data_directory_path


# For each keypoint obtain the stored information related to the point
def serialize_keypoints(kp, des):
    return [(point.pt, point.size, point.angle, point.response, point.octave, point.class_id, desc) for point, desc in
            zip(kp, des)]


# Combine all the keypoint information in a single array
def deserialize_keypoints(serialized_keypoints):
    kp = [cv2.KeyPoint(x=point[0][0], y=point[0][1], size=point[1], angle=point[2],
                      response=point[3], octave=point[4], class_id=point[5]) for point in serialized_keypoints]
    des = np.array([point[6] for point in serialized_keypoints])
    return kp, des


# Matches descriptors of two specified images and gets distance error, and matched keypoints
def match_image_descriptors(des1, des2, i, j, n_matches=10):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    # sort matches according to distance from the descriptors Lower is better
    matches = sorted(matches, key=lambda x: x.distance)
    # get the lowest 10 distance and sum them up lower is better.
    metric = np.sum([match.distance for match in matches[:n_matches]])
    return metric, serialize_matches(matches[:n_matches]), (i, j)


# Extract matches information for each match
def serialize_matches(matches):
    return [(match.queryIdx, match.trainIdx, match.distance) for match in matches]


# Combine all the matches in a single array
def deserialize_matches(serialized_matches):
    return [cv2.DMatch(serialized_match[0], serialized_match[1], serialized_match[2]) for serialized_match in
            serialized_matches]


# Apply ORB Algorithm to detect features in the grey scale images and store the returned keypoints and descriptors
def apply_orb_on_images(grey):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(grey, None)
    return serialize_keypoints(kp, des)


# This funcion is called to execute the feature detection on the rgb and depth image dataset using the ORB Algorithm,
# matching is applied to the images based on the image thresholds to get good matches
# then the ICP algorithm is applied to get the transformation good matches
def loop_closure_detection_and_icp(image_err_thresh: int = 110, skip_images: int = 30,
                                   orb_crop_top: float = 200, icp_min_x: float = 0, icp_max_x: float = 50,
                                   icp_min_y: float = -np.inf, icp_max_y: float = np.inf, icp_min_z: float = 0,
                                   icp_max_z: float = 3, plot_error_dist_mat: bool = None,
                                   plot_selected_loop_closures: bool = None, show_info_mat_csv: bool = False,
                                   plot_matched_images: bool = None, show_icp_alignments: bool = False):
    """
    This function finds loop closures based on ORB and finds transforms between loop closure poses using scan matching
    based on ICP. Parameters for ORB (upper part of images to match can be cropped) and ICP (only part of point cloud,
    in ZED left camera frame: x -> depth, y -> width, z -> height) can be adapted. Results from intermediate steps can
    be visualized.
    """


    # Load the images and convert them into grey scale images
    path_to_data = get_data_directory_path()
    lst = os.listdir(path_to_data / 'rgb')
    num_imgs = len(lst)
    print(num_imgs)
    greys = [np.asarray(load_rgb_depth(index=i, load_rgb=True, crop_top=orb_crop_top, greyscale=True)[0])
             for i in range(1, num_imgs + 1)]

    # Apply the ORB algorithm and get descriptors and keypoints of all the images
    # in the dataset at the specified image rate(this specifies rate at which images are considered)
    parallel = Parallel(n_jobs=-1, verbose=0, backend="loky")
    serialized_keypoints_list = parallel(
        delayed(apply_orb_on_images)(greys[i]) for i in tqdm(range(0, len(greys))))
    keypoints, descriptors = zip(
        *[deserialize_keypoints(serialized_keypoints) for serialized_keypoints in serialized_keypoints_list])

    # Match descriptors of two images and get the
    # dist_mat : stores the error between the keypoints of the images and
    # matched_keypoints : stores the list of keypoints that are matched
    # for matching first n images are skipped to avoid loop closure detection of nearby images
    print("\nMatching Descriptors of two images\n")
    dist_mat_s, matched_keypoints_s, idx_s = zip(*parallel(
        delayed(match_image_descriptors)(descriptors[i], descriptors[j], i, j, n_matches=10)
        for i in tqdm(range(0, len(descriptors)))
        for j in range(i + skip_images, len(descriptors))))

    # store the matching keypoints and the distance errors in a single array
    matched_keypoints = [[None for _ in range(len(greys))] for _ in range(len(greys))]
    dist_mat = np.full((len(descriptors), len(descriptors)), np.inf)
    for idx in range(len(dist_mat_s)):
        i, j = idx_s[idx]
        dist_mat[i, j] = dist_mat_s[idx]
        dist_mat[j, i] = dist_mat_s[idx]
        matched_keypoints[i][j] = deserialize_matches(matched_keypoints_s[idx])

    # Plot the distance error matrix
    if plot_error_dist_mat is True:
        plt.title("Distance matrix from ORB feature matching")
        plt.xlabel("pose index")
        plt.ylabel("pose index")
        plt.imshow(dist_mat)
        plt.colorbar()
        plt.show()

    # get the good matches according to the image threshold
    # i.e. if the dist error is less than the image threshold then the 2 images are stored as good matches
    good_matches = []
    good_matches_keypoints = []
    for j in range(dist_mat.shape[1]):
        i = np.argmin(dist_mat[:, j])
        if dist_mat[i, j] < image_err_thresh:
            good_matches.append([i, j])
            good_matches_keypoints.append(matched_keypoints[i][j])

    # Plot the matching keypoints of two images of good matches
    if plot_matched_images is True:
        for idx in range(len(good_matches)):
            i, j = good_matches[idx]
            img3 = cv2.drawMatches(greys[i], keypoints[i], greys[j], keypoints[j], matched_keypoints[i][j], None,
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            image_name = "Matched image numbers " + str(good_matches[idx])
            plt.title(image_name)
            plt.imshow(img3)
            plt.show()

    if plot_selected_loop_closures is True:
        plot_loop_closures(path_to_data, good_matches)

    # Apply ICP algorithm to the good matches and get the transformations and information matrix
    # between the two images
    print("ICP STARTED ON THE GOOD MATCHES")
    tfs, _, information_matrix = zip(*parallel(delayed(icp_registration)(i + 1, j + 1, icp_min_x, icp_max_x, icp_min_y,
                                                                         icp_max_y, icp_min_z, icp_max_z,
                                                                         show_icp_alignments)
                                               for i, j in tqdm(good_matches)))
    print("ICP COMPLETED")
    # Get the pose id's from the image numbers of the good matches for storing into loop closure csv
    pose_id1 = []
    pose_id2 = []
    for idx in range(len(good_matches)):
        i, j = good_matches[idx]
        pose_id1.append(i)
        pose_id2.append(j)
    # Store the transforms and information matrix in the csv file
    print("STORING THE TRANSFORMS FROM ICP AS LOOP CLOSURES DETECTION DATA")
    store_icp_transforms_csv(pose_id1, pose_id2, tfs, information_matrix, path_to_data, len(good_matches),
                             show_info_mat_csv)
    print("CSV FILE CREATED CHECK THE DATA FOLDER FOR THE RESULT")


def plot_loop_closures(path_to_data: Path, loop_closures: np.array):
    # Plot the odometry poses from the camera
    poses = np.loadtxt(str(path_to_data / 'poses.csv'), skiprows=1, delimiter=';')
    odometry = plt.plot(poses[:, 1], poses[:, 2])
    odom = odometry[-1]
    # Extract the yaw angle from the quaternion
    orientation = []
    for i in range(0, len(poses), 10):
        _, _, yaw = euler_from_quaternion(poses[i, 4:])
        dx = np.cos(yaw)
        dy = np.sin(yaw)
        orientation = plt.quiver(poses[i, 1], poses[i, 2], dx, dy)

    # Plot the loop closures based on the good matches
    loops = []
    for i, j in loop_closures:
        loops = plt.plot([poses[i, 1], poses[j, 1]], [poses[i, 2], poses[j, 2]], color='orange')
    loop = loops[-1]
    plt.legend([orientation, odom, loop], ["Car Orientation", "Odometry based trajectory", "loop closures"])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.axis('equal')
    plt.grid()
    plt.show()


def store_icp_transforms_csv(poseid1, poseid2, transforms, inform_matrix, path_to_csv: Path, num_samples: int,
                             is_info_mat_stored: bool = None) -> None:
    # Open the loop closure csv file in write mode
    f_loop = open(str(path_to_csv / 'loop_closure.csv'), 'w')
    writer_loop = csv.writer(f_loop, delimiter=';')
    # Header for the loop closure csv file
    header = ['pose_id1', 'pose_id2', 'x', 'y', 'z', 'quat_x', 'quat_y', 'quat_z', 'quat_w']
    writer_loop.writerow(header)
    for i in range(0, num_samples):
        # Extract the translations from the last column of the transformation matrix
        x = transforms[i][0, 3]
        y = transforms[i][1, 3]
        z = transforms[i][2, 3]
        # Extract the quaternion from the rotation matrix
        quat_x, quat_y, quat_z, quat_w = quaternion_from_matrix(transforms[i][0:4, 0:4])
        # Write the loop closure data in the csv file
        writer_loop.writerow([poseid1[i] + 1, poseid2[i] + 1, x, y, z, quat_x, quat_y, quat_z, quat_w])
    f_loop.close()

    # Stores the information matrix csv only if asked for
    if is_info_mat_stored is True:
        f_info_mat = open(str(path_to_csv / 'hide_info_mat.csv'), 'w')
        writer_info_mat = csv.writer(f_info_mat, delimiter=';')
        for i in range(0, num_samples):
            # Only the information matrix for x, y translation and yaw angle is extracted
            info_mat = np.delete(inform_matrix[i], slice(2, 5), axis=0)
            info_mat = np.delete(info_mat, slice(2, 5), axis=1)
            info_mat = info_mat.flatten()
            writer_info_mat.writerow([i, str(info_mat[0]), str(info_mat[1]), str(info_mat[2]),
                                      str(info_mat[3]), str(info_mat[4]), str(info_mat[5]),
                                      str(info_mat[6]), str(info_mat[7]), str(info_mat[8])])
        f_info_mat.close()
