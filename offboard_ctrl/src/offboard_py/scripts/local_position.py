"""
 * File: local_position.py
 Check if local position is broadcasted well from Vicon.
"""

#! /usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

current_state = State()
local_pose = PoseStamped()

def state_cb(msg):
    global current_state
    current_state = msg

def local_pose_cb(msg):
    global local_pose
    local_pose = msg


if __name__ == "__main__":
    rospy.init_node("takeoff_and_land_node")

    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)

    curr_pose_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, callback=local_pose_cb)

    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)


    # Setpoint publishing MUST be faster than 2Hz
    sleep_rate = rospy.Rate(20)  # 2Hz
    ctrl_rate = rospy.Rate(200)  # 20Hz


    # Wait for Flight Controller connection
    while(not rospy.is_shutdown() and not current_state.connected):
        sleep_rate.sleep()

    print("Flight controller connected!")

    curr_pose = PoseStamped()

    last_req = rospy.Time.now()

    while(not rospy.is_shutdown()):
        # if (rospy.Time.now() - last_req) > rospy.Duration(5.0):
        #     if(set_mode_client.call(offb_set_mode).mode_sent == True):
        #         rospy.loginfo("OFFBOARD enabled")

        #     last_req = rospy.Time.now()
        # else:
        #     if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
        #         if(arming_client.call(arm_cmd).success == True):
        #             rospy.loginfo("Vehicle armed")

        #         last_req = rospy.Time.now()

        print(local_pose)

        sleep_rate.sleep()
