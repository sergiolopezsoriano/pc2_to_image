#!/usr/bin/env python

import traceback
import numpy as np
import rospy
import os
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import tensorflow as tf
from tensorflow.keras import models
from PIL import Image
from rospkg import RosPack
import matplotlib.pyplot as plt
import ctypes
import struct
import pcl
import ros_numpy
from scipy.interpolate import griddata


class DatasetGenerator:
    def __init__(self):
        # ROS Params
        # self.namespace = rospy.get_namespace().strip('/')
        self.path = RosPack().get_path('pc2_to_image')
        # self.dataset_folder = rospy.get_param('~dataset_folder')
        # self.model_path = rospy.get_param('~model_path')
        rospy.Subscriber('scan_3D', PointCloud2, self.callback, queue_size=1, buff_size=52428800)
        # self.pc2 = PointCloud2()
        self.pc2_received = False

    def callback(self, ros_point_cloud):
        if input('press space'):
            pass
        else:
            pass
        xyz = np.array([[0, 0, 0]])
        rgb = np.array([[0, 0, 0]])
        # self.lock.acquire()
        gen = pc2.read_points(ros_point_cloud, skip_nans=True)
        int_data = list(gen)

        for data in int_data:
            test = data[3]
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f', test)
            i = struct.unpack('>l', s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000) >> 16
            g = (pack & 0x0000FF00) >> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
            # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz, [[data[0], data[1], data[2]]], axis=0)
            rgb = np.append(rgb, [[r, g, b]], axis=0)

        xyz = np.delete(xyz, 0, axis=0)
        rgb = np.delete(rgb, 0, axis=0)
        xyz = xyz[xyz[:, 0] < 1.5]
        xyz = xyz[xyz[:, 2] > -0.1]
        # print(xyz[(xyz[:, 1] > -0.1) * (xyz[:, 1] < 0.1)])
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]

        # target grid to interpolate to
        # height = 60
        # width = 160
        # yi = np.arange(-3.5, 3.5, 7. / width)
        # zi = np.arange(-2, 2.5, 4.5 / height)
        # yi, zi = np.meshgrid(yi, zi)
        # # interpolate
        # xi = griddata((y, z), x, (yi, zi), method='linear')
        # plt.contourf(yi, zi, xi, np.arange(0, 4.5, 0.02))

        # plot
        fig = plt.figure()
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        ax.set(xlim=(-3.45, 3.45), ylim=(-1.9, 2.3))
        fig.add_axes(ax)
        plt.scatter(y, z, c=x, s=0.5, cmap='jet', linewidths=5)
        num_img = len([name for name in os.listdir(self.path + '/images')])
        plt.savefig('{}/images/image{}.png'.format(self.path, num_img+1), dpi=100)
        # plt.xlabel('y', fontsize=16)
        # plt.ylabel('z', fontsize=16)
        # plt.grid(b=True, which='both', axis='both')
        plt.show()

        # test_image = plt.imread('{}/images/image{}.png'.format(self.path, num_img+1))
        # plt.imshow(test_image)
        # plt.show()
        # np.save('/home/sergi/xyz', xyz)
        # np.save('/home/sergi/rgb', rgb)
        plt.close()
        # rospy.signal_shutdown('s')


if __name__ == "__main__":

    try:
        rospy.init_node('DatasetGenerator', log_level=rospy.INFO)
        rospy.loginfo('[DatasetGenerator]: Node started')
        rd = DatasetGenerator()
        rospy.spin()
    except Exception as e:
        rospy.logfatal('[DatasetGenerator]: Exception %s', str(e.message) + str(e.args))
        e = traceback.format_exc()
        rospy.logfatal(e)
