"""
 * File: takeoff_and_land.py
 * Stack and tested in Gazebo Classic 9 SITL
"""

#! /usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped, Point
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

current_state = State()
current_pose = PoseStamped()

def state_cb(msg):
    global current_state
    current_state = msg

def pose_cb(msg):
    global current_pose
    current_pose = msg

def distance(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5

def setpoint_reached(setpoint, current_pose, tolerance=0.1):
    return distance(setpoint.pose.position, current_pose.pose.position) < tolerance

if __name__ == "__main__":
    rospy.init_node("takeoff_and_land_node")

    state_sub = rospy.Subscriber("mavros/state", State, callback=state_cb)
    pose_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, callback=pose_cb)

    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)

    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while not rospy.is_shutdown() and not current_state.connected:
        rate.sleep()

    takeoff_pose = PoseStamped()
    takeoff_pose.pose.position.x = 0
    takeoff_pose.pose.position.y = 0
    takeoff_pose.pose.position.z = 1

    # Send a few setpoints before starting
    for i in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(takeoff_pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    square_points = [
        # Point(0, 0, 1),
        # Point(1, 0, 1),
        # Point(1, 1, 1),
        # Point(0, 1, 1)
        Point(0, 0, 1),
        Point(0, 0, 2),
        Point(0, 1, 2),
        Point(0, 1, 1)
    ]
    current_setpoint_index = 0
    tolerance = 0.1

    while not rospy.is_shutdown():
        if current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
            if set_mode_client.call(offb_set_mode).mode_sent == True:
                rospy.loginfo("OFFBOARD enabled")
            last_req = rospy.Time.now()
        else:
            if not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                arming_resp = arming_client.call(arm_cmd)
                if arming_resp.success == True:
                    rospy.loginfo("Vehicle armed")
                else:
                    rospy.loginfo("Vehicle arming failed")
                    rospy.loginfo(arming_resp)
                last_req = rospy.Time.now()

        if setpoint_reached(takeoff_pose, current_pose, tolerance):
            current_setpoint_index = (current_setpoint_index + 1) % len(square_points)
            next_setpoint = square_points[current_setpoint_index]
            takeoff_pose.pose.position.x = next_setpoint.x
            takeoff_pose.pose.position.y = next_setpoint.y
            takeoff_pose.pose.position.z = next_setpoint.z

        local_pos_pub.publish(takeoff_pose)

        rate.sleep()
