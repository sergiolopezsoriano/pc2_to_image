#!/usr/bin/env python

import traceback
import rospy
from sensor_msgs.msg import Image as Img
import cv2
import cv_bridge


class PlotDetectionImage:
    def __init__(self):
        # self.robot_ns = rospy.get_param('~robot_ns')
        self.robot_ns = 'robot4'
        rospy.Subscriber('/' + self.robot_ns + '/darknet_ros/detection_image', Img, self.cb_detection_image, queue_size=1)

    def cb_detection_image(self, imgmsg):
        cvbridge = cv_bridge.CvBridge()
        array = cvbridge.imgmsg_to_cv2(imgmsg)
        cv2.imshow('detection', array)


if __name__ == "__main__":

    try:
        rospy.init_node('PlotDetectionImage', log_level=rospy.INFO)
        rospy.loginfo('[PlotDetectionImage]: Node started')
        PlotDetectionImage()
        rospy.spin()
    except Exception as e:
        rospy.logfatal('[PlotDetectionImage]: Exception %s', str(e.message) + str(e.args))
        e = traceback.format_exc()
        rospy.logfatal(e)
