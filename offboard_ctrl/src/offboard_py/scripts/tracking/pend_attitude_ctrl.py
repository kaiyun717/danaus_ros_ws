"""
Attitude control script for controlling the pendulum using the attitude setpoints.
This employs a simple PD controller that tries to minimize the roll and pitch of the
pendulum from the vertical line. The thrust is controlled by the velocity of the 
pendulum in the z direction.
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
local_pos = PoseStamped()
roll_rad = 0
pitch_rad = 0
abort = False

def state_cb(msg):
    global current_state
    current_state = msg

def imu_cb(msg):
    global linear_z_acceleration
    linear_z_acceleration = msg.linear_acceleration.z

def vel_body_cb(msg):
    global linear_z_vel
    linear_z_vel = msg.twist.linear.z

def pend_pos_cb(msg):
    dx = msg.pose.position.x - local_pos.pose.position.x
    dy = msg.pose.position.y - local_pos.pose.position.y
    dz = msg.pose.position.z - local_pos.pose.position.z
    
    # dx = msg.pose.position.x - -3.4203612964389123
    # dy = msg.pose.position.y - -0.4856407524683649
    # dz = msg.pose.position.z - 0
    
    global roll_rad, pitch_rad
    pitch_rad = math.atan2(dx, dz)
    roll_rad = math.atan2(dy, dz)
    pitch_deg = pitch_rad * 180.0 / math.pi
    roll_deg = roll_rad * 180.0 / math.pi

    if abs(roll_rad) > 20 / 180 * math.pi:
        roll_rad = 20 / 180 * math.pi * roll_rad / abs(roll_rad)

    if abs(pitch_rad) > 20 / 180 * math.pi:
        pitch_rad = 20 / 180 * math.pi * pitch_rad / abs(pitch_rad)
        
    # rospy.loginfo("Pendulum angles: %s, %s", pitch_deg, roll_deg)

def local_pos_cb(msg):
    global local_pos
    local_pos = msg

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
        attitude_setpoint.pose.position.z = 0.5
        attitude_setpoint.pose.orientation.x = attitude_quaternion[0]
        attitude_setpoint.pose.orientation.y = attitude_quaternion[1]
        attitude_setpoint.pose.orientation.z = attitude_quaternion[2]
        attitude_setpoint.pose.orientation.w = attitude_quaternion[3]
        rospy.loginfo_once("Attitude setpoint: %s", attitude_quaternion)
        local_pos_pub.publish(attitude_setpoint)
        rate.sleep()

def send_att(duration):
    global abort
    att = AttitudeTarget()
    att_setpoint_pub = rospy.Publisher('mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=1)
    
    att.body_rate = Vector3()
    att.header = Header()
    att.header.frame_id = "base_footprint"
    # att.orientation = Quaternion(*quaternion_from_euler(0, pitch_degrees*math.pi/180.0, 0))
    # att.type_mask = 7  # ignore body rate
    # att.type_mask = 71  # ignore body rate + thrust

    rate = rospy.Rate(20)
    start_time = rospy.Time.now()
    while (rospy.Time.now() - start_time) < rospy.Duration(duration):
        bound_xy = 1.8
        bound_z = 3
        if -bound_xy < local_pos.pose.position.x < bound_xy and \
           -bound_xy < local_pos.pose.position.y < bound_xy and \
            local_pos.pose.position.z < bound_z and not abort:
            att.header.stamp = rospy.Time.now()
            # att.thrust = 0.415 + 0.05 * (9.807 - linear_z_acceleration)
            att.orientation = Quaternion(*quaternion_from_euler(-1 * roll_rad, 1 * pitch_rad, 0))
            att.thrust = 0.415 + 0.25 * (- linear_z_vel)
            # att.thrust = 0.415 + 0.5 * (1 - local_pos.pose.position.z)
            rospy.loginfo("Error: %s, Thrust: %s", - linear_z_vel, att.thrust)
            att_setpoint_pub.publish(att)
        else:
            rospy.loginfo_once("Aborting.")
            abort = True
            takeoff_pose = PoseStamped()
            takeoff_pose.pose.position.x = 0
            takeoff_pose.pose.position.y = 0
            takeoff_pose.pose.position.z = 0.5
            local_pos_pub.publish(takeoff_pose)
            # if abs(takeoff_pose.pose.position.x - local_pos.pose.position.x) < 0.3 and \
            #    abs(takeoff_pose.pose.position.y - local_pos.pose.position.y) < 0.3 and \
            #    abs(takeoff_pose.pose.position.z - local_pos.pose.position.z) < 0.3:
            #     abort = False
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
    pend_pos_sub = rospy.Subscriber("vicon/pose/pendulum/pendulum", PoseStamped, callback=pend_pos_cb)
    local_pos_sub = rospy.Subscriber("mavros/local_position/pose", PoseStamped, callback=local_pos_cb)    

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

    # while(not rospy.is_shutdown()):
        # rate.sleep()

    takeoff_pose = PoseStamped()

    takeoff_pose.pose.position.x = 0
    takeoff_pose.pose.position.y = 0
    takeoff_pose.pose.position.z = 0.5

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
    while (rospy.Time.now() - start_time) < rospy.Duration(25):
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
    send_att(20)
    rospy.loginfo("Attitude Control Finished")

    while (rospy.Time.now() - start_time) < rospy.Duration(20):
        # if(current_state.mode != "OFFBOARD" and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
        #     if(set_mode_client.call(offb_set_mode).mode_sent == True):
        #         rospy.loginfo("OFFBOARD enabled")

        #     last_req = rospy.Time.now()
        # else:
        #     if(not current_state.armed and (rospy.Time.now() - last_req) > rospy.Duration(5.0)):
        #         arming_resp = arming_client.call(arm_cmd)
        #         if(arming_resp.success == True):
        #             rospy.loginfo("Vehicle armed")
        #         else:
        #             rospy.loginfo("Vehicle arming failed")
        #             rospy.loginfo(arming_resp)
        #         last_req = rospy.Time.now()

        rospy.loginfo("Coming back to takeoff pose.")
        local_pos_pub.publish(takeoff_pose)
        rate.sleep()

        # to disarm: arm_cmd.value = False and then push it to arming_client
