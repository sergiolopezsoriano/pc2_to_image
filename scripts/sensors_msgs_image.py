#!/usr/bin/env python

import rospy
from rosbag import Bag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


msg_list = list()
rospy.init_node('ImageMessage', log_level=rospy.INFO)
bag = Bag('/home/sergi/catkin_ws/src/pc2_to_image/bags/image_raw.bag')
msg = Image()
for topic, msg, t2 in bag.read_messages():
    msg_list.append(msg)
bridge = CvBridge()
cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
bag.close()


# Get the image with the bounding boxes
# rospy.Subscriber('/darknet_ros/detection_image', Img, self.cb_detection_image, queue_size=1)
#
# def cb_detection_image(self, imgmsg):
#     cvbridge = cv_bridge.CvBridge()
#     array = cvbridge.imgmsg_to_cv2(imgmsg)
#     cv2.imshow('detection', array)
