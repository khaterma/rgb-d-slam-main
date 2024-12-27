import numpy as np
from geometry_msgs.msg import TransformStamped
from matplotlib import pyplot as plt
import csv
from pathlib import Path
from tf.transformations import euler_from_quaternion
from utils.data_directory_from_json import get_thresholds


class PlotPosesMgr():
    def __init__(self):
        self.poses_zed_odom = np.ndarray((2, 0))
        self.poses_zed_odom_samples = np.ndarray((3, 0))
        self.poses_zed_gt = np.ndarray((2, 0))

    def add_pose(self, t: TransformStamped, is_gt=False):
        new_x = t.transform.translation.x
        new_y = t.transform.translation.y
        if not is_gt:
            self.poses_zed_odom = np.append(self.poses_zed_odom, [[new_x], [new_y]], axis=1)
        else:
            self.poses_zed_gt = np.append(self.poses_zed_gt, [[new_x], [new_y]], axis=1)

    def add_sampled_pose(self, t: TransformStamped):
        new_x = t.transform.translation.x
        new_y = t.transform.translation.y
        q = t.transform.rotation
        new_z = euler_from_quaternion([q.x, q.y, q.z, q.w])[-1]
        self.poses_zed_odom_samples = np.append(self.poses_zed_odom_samples, [[new_x], [new_y], [new_z]], axis=1)

    def create_plot(self):
        plt.figure()
        plt.title(f"Pose sampling (translation threshold {get_thresholds()[0]}) in world frame")
        plt.plot(self.poses_zed_odom[0, :], self.poses_zed_odom[1, :], label="ZED center poses (odometry)")
        plt.plot(self.poses_zed_gt[0, :], self.poses_zed_gt[1, :], label="ZED center poses (ground truth)")
        # show sampled poses as arrows
        x = self.poses_zed_odom_samples[0, :]
        y = self.poses_zed_odom_samples[1, :]
        u = np.cos(self.poses_zed_odom_samples[2, :])
        v = np.sin(self.poses_zed_odom_samples[2, :])
        plt.quiver(x, y, u, v, label="Sampled poses w. orientation.", color="olive")

        plt.plot(self.poses_zed_odom[0, 0], self.poses_zed_odom[1, 0], 'go', label="start pose")
        plt.plot(self.poses_zed_odom[0, -1], self.poses_zed_odom[1, -1], 'ro', label="end pose (odometry)")
        plt.plot(self.poses_zed_gt[0, -1], self.poses_zed_gt[1, -1], 'rx', label="end pose (ground truth)")

        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

        plt.legend()
        plt.grid()
        plt.axis('scaled')
        plt.show()

    def store_gt(self, path_to_csv: Path):
        f = open(path_to_csv, 'w')

        writer = csv.writer(f, delimiter=';')

        header = ['x', 'y']
        writer.writerow(header)
        for pose in self.poses_zed_gt.transpose((1, 0)):
            x = pose[0]
            y = pose[1]

            writer.writerow([x, y])

        f.close()
