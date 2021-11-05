#!/usr/bin/env python

import traceback
import numpy as np
import rospy
from sensor_msgs.msg import Image as Img
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from rospkg import RosPack
import matplotlib.pyplot as plt
import cv2
import cv_bridge


class Pointcloud2Image:
    def __init__(self):
        self.robot_ns = rospy.get_param('~robot_ns')
        self.sensor_frame = rospy.get_param('~sensor_frame')
        self.path = RosPack().get_path('pc2_to_image')
        rospy.Subscriber('/' + self.robot_ns + '/camera/depth_registered/points', PointCloud2, self.cb_point_cloud,
                         queue_size=1, buff_size=52428800)
        self.pc2_received = False
        self.image_publisher = rospy.Publisher('/' + self.robot_ns + '/camera/rgb/image_raw', Img, queue_size=1)

    def cb_point_cloud(self, ros_point_cloud):
        xyz = np.array([[0, 0, 0]])
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        for data in int_data:
            xyz = np.append(xyz, [[data[0], data[1], data[2]]], axis=0)

        xyz = np.delete(xyz, 0, axis=0)
        xyz = xyz[xyz[:, 0] < 1.5]
        xyz = xyz[xyz[:, 2] > -0.1]
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.set(xlim=(-3.45, 3.45), ylim=(-1.9, 2.3))
        fig.add_axes(ax)
        plt.scatter(y, z, c=x, s=0.5, cmap='jet', linewidths=5)
        plt.savefig('{}/snapshot.png'.format(self.path))
        plt.close(fig)
        # plt.xlabel('y', fontsize=16)
        # plt.ylabel('z', fontsize=16)
        # plt.grid(b=True, which='both', axis='both')
        # plt.show()
        # img = Image.open(self.path + '/snapshot.png')
        # img = img.resize((640, 480))
        img = cv2.imread(self.path + '/snapshot.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)
        cvbridge = cv_bridge.CvBridge()
        pointcloud_image = cvbridge.cv2_to_imgmsg(np.asarray(img))
        pointcloud_image.header.frame_id = self.sensor_frame
        pointcloud_image.encoding = 'bgr8'
        pointcloud_image.is_bigendian = 0
        self.image_publisher.publish(pointcloud_image)
        rospy.sleep(0.5)

    def check_ros(self):
        img = cv2.imread(self.path + '/images/image1.png')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA)

        cvbridge = cv_bridge.CvBridge()
        pointcloud_image = cvbridge.cv2_to_imgmsg(np.asarray(img))
        pointcloud_image.header.frame_id = self.sensor_frame
        pointcloud_image.encoding = 'bgr8'
        pointcloud_image.is_bigendian = 0
        # while self.image_publisher.get_num_connections() < 1:
        #     rospy.sleep(0.5)
        self.image_publisher.publish(pointcloud_image)


if __name__ == "__main__":

    try:
        rospy.init_node('Pointcloud2Image', log_level=rospy.INFO)
        rospy.loginfo('[Pointcloud2Image]: Node started')
        rd = Pointcloud2Image()
        # while 1:
        #     rd.check_ros()
        rospy.spin()
    except Exception as e:
        rospy.logfatal('[Pointcloud2Image]: Exception %s', str(e.message) + str(e.args))
        e = traceback.format_exc()
        rospy.logfatal(e)
