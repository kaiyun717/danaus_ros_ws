#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import PoseStamped

def publish_pose():
    rospy.init_node('zero_pose_publisher', anonymous=True)
    rate = rospy.Rate(100)  # 30 Hz

    pub = rospy.Publisher('/mavros/vision_pose/pose', PoseStamped, queue_size=10)

    while not rospy.is_shutdown():
        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.pose.position.x = 0.0
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 0.0

        pub.publish(pose_msg)
        rospy.loginfo("Published zero pose")
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_pose()
    except rospy.ROSInterruptException:
        pass