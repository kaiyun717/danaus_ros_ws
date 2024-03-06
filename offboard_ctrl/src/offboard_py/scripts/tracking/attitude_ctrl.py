"""
 * File: takeoff_and_land.py
 * Stack and tested in Gazebo Classic 9 SITL
"""

#! /usr/bin/env python

import rospy

from geometry_msgs.msg import PoseStamped, Quaternion, Vector3, TwistStamped
from std_msgs.msg import Header
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
import math
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from sensor_msgs.msg import Imu


current_state = State()
linear_z_acceleration = 0.0
linear_z_vel = 0.0 


def state_cb(msg):
    global current_state
    current_state = msg

def imu_cb(msg):
    global linear_z_acceleration
    linear_z_acceleration = msg.linear_acceleration.z

def vel_body_cb(msg):
    global linear_z_vel
    linear_z_vel = msg.twist.linear.z

def send_attitude_setpoint(pitch_degrees, duration):
    # attitude_pub = rospy.Publisher("mavros/setpoint_attitude/attitude", PoseStamped, queue_size=10)
    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

    rate = rospy.Rate(20)
    start_time = rospy.Time.now()
    while (rospy.Time.now() - start_time) < rospy.Duration(duration):
        attitude_setpoint = PoseStamped()
        attitude_setpoint.header.stamp = rospy.Time.now()
        # attitude_setpoint.pose.orientation.x = math.sin(pitch_degrees/2.0*math.pi/180.0)
        # attitude_setpoint.pose.orientation.y = 0
        # attitude_setpoint.pose.orientation.z = 0
        # attitude_setpoint.pose.orientation.w = math.cos(pitch_degrees/2.0*math.pi/180.0)
        attitude_quaternion = quaternion_from_euler(0, pitch_degrees*math.pi/180.0, 0)
        attitude_setpoint.pose.position.x = 0
        attitude_setpoint.pose.position.y = 0
        attitude_setpoint.pose.position.z = 1
        attitude_setpoint.pose.orientation.x = attitude_quaternion[0]
        attitude_setpoint.pose.orientation.y = attitude_quaternion[1]
        attitude_setpoint.pose.orientation.z = attitude_quaternion[2]
        attitude_setpoint.pose.orientation.w = attitude_quaternion[3]
        rospy.loginfo_once("Attitude setpoint: %s", attitude_quaternion)
        local_pos_pub.publish(attitude_setpoint)
        rate.sleep()

def send_att(pitch_degrees, duration):
        att = AttitudeTarget()
        att_setpoint_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
        
        att.body_rate = Vector3()
        att.header = Header()
        att.header.frame_id = "base_footprint"
        att.orientation = Quaternion(*quaternion_from_euler(0, pitch_degrees*math.pi/180.0, 0))
        att.type_mask = 7  # ignore body rate
        # att.type_mask = 71  # ignore body rate + thrust

        rate = rospy.Rate(20)
        start_time = rospy.Time.now()
        while (rospy.Time.now() - start_time) < rospy.Duration(duration):
            att.header.stamp = rospy.Time.now()
            # att.thrust = 0.415 + 0.05 * (9.807 - linear_z_acceleration)
            att.thrust = 0.415 + 0.2 * (- linear_z_vel)
            rospy.loginfo("Error: %s, Thrust: %s", - linear_z_vel, att.thrust)
            att_setpoint_pub.publish(att)
            try:  # prevent garbage in console output when thread is killed
                rate.sleep()
            except rospy.ROSInterruptException:
                pass
        
if __name__ == "__main__":
    print("Helllloooo Soham & Kai")

    rospy.init_node("takeoff_and_land_node")

    state_sub = rospy.Subscriber("mavros/state", State, callback = state_cb)
    imu_sub = rospy.Subscriber("mavros/imu/data", Imu, callback=imu_cb)
    vel_body_sub = rospy.Subscriber("mavros/local_position/velocity_body", TwistStamped, callback=vel_body_cb)
    

    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)


    # Setpoint publishing MUST be faster than 2Hz
    rate = rospy.Rate(20)

    # Wait for Flight Controller connection
    while(not rospy.is_shutdown() and not current_state.connected):
        rate.sleep()

    takeoff_pose = PoseStamped()

    takeoff_pose.pose.position.x = 0
    takeoff_pose.pose.position.y = 0
    takeoff_pose.pose.position.z = 1

    # Send a few setpoints before starting
    for i in range(100):
        if(rospy.is_shutdown()):
            break

        local_pos_pub.publish(takeoff_pose)
        rate.sleep()

    offb_set_mode = SetModeRequest()
    offb_set_mode.custom_mode = 'OFFBOARD'

    arm_cmd = CommandBoolRequest()
    arm_cmd.value = True

    last_req = rospy.Time.now()

    rate = rospy.Rate(20)
    start_time = rospy.Time.now()
    while (rospy.Time.now() - start_time) < rospy.Duration(20):
        if(current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
            if(set_mode_client.call(offb_set_mode).mode_sent == True):
                rospy.loginfo("OFFBOARD enabled")

            last_req = rospy.Time.now()
        else:
            if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
                arming_resp = arming_client.call(arm_cmd)
                if(arming_resp.success == True):
                    rospy.loginfo("Vehicle armed")
                else:
                    rospy.loginfo("Vehicle arming failed")
                    rospy.loginfo(arming_resp)
                last_req = rospy.Time.now()

        local_pos_pub.publish(takeoff_pose)
        rate.sleep()

    rospy.loginfo("Attitude Control Start")
    # send_attitude_setpoint(5, 5)
    send_att(5, 2)
    rospy.loginfo("Attitude Control Finished")

    # while (rospy.Time.now() - start_time) < rospy.Duration(20):
    #     # if(current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
    #     #     if(set_mode_client.call(offb_set_mode).mode_sent == True):
    #     #         rospy.loginfo("OFFBOARD enabled")

    #     #     last_req = rospy.Time.now()
    #     # else:
    #     #     if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
    #     #         arming_resp = arming_client.call(arm_cmd)
    #     #         if(arming_resp.success == True):
    #     #             rospy.loginfo("Vehicle armed")
    #     #         else:
    #     #             rospy.loginfo("Vehicle arming failed")
    #     #             rospy.loginfo(arming_resp)
    #     #         last_req = rospy.Time.now()

    #     rospy.loginfo("Coming back to takeoff pose.")
    #     local_pos_pub.publish(takeoff_pose)
    #     rate.sleep()

        # to disarm: arm_cmd.value = False and then push it to arming_client
