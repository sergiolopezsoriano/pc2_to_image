#!/usr/bin/env python
import rospy
import tf


def broadcast_frame():
    br = tf.TransformBroadcaster()
    br.sendTransform((0, 0, 0),
                     tf.transformations.quaternion_from_euler(0, 0, 0),
                     rospy.Time.now(),
                     "robot4_tf/cyglidar_link",
                     "laser_link")


if __name__ == '__main__':
    rospy.init_node('frame_broadcaster')
    while not rospy.is_shutdown():
        broadcast_frame()
