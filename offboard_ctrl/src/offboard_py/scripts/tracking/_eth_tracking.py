"""
This script is used to track the position of the drone using the ETH controller, 
based on the paper "A Flying Inverted Pendulum" by Raffaello D'Andrea and Markus Hehn.
"""
import rospy
import math
import numpy as np
import scipy

import argparse

from geometry_msgs.msg import PoseStamped, Quaternion, Vector3, TwistStamped
from std_msgs.msg import Header
from sensor_msgs.msg import Imu
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from offboard_ctrl.src.offboard_py.src.controllers.eth_constant_controller import ConstantPositionTracker
from src.callbacks.fcu_state_callbacks import VehicleStateCB
from src.callbacks.fcu_modes import FcuModes

current_state = State()
local_pos = PoseStamped()

vel_linear_z_acc = 0.0
vel_linear_z_vel = 0.0
pend_roll_rad = 0
pend_pitch_rad = 0

def state_cb(msg):
    global current_state
    current_state = msg

def imu_cb(msg):
    global linear_z_acceleration
    linear_z_acceleration = msg.linear_acceleration.z

def vel_body_cb(msg):
    global linear_z_vel
    linear_z_vel = msg.twist.linear.z

def local_pos_cb(msg):
    global local_pos
    local_pos = msg

def pend_pos_cb(msg):
    r = msg.pose.position.x - local_pos.pose.position.x
    s = msg.pose.position.y - local_pos.pose.position.y
    dz = msg.pose.position.z - local_pos.pose.position.z

    global pend_roll_rad, pend_pitch_rad
    pend_pitch_rad = math.atan2(r, dz)
    pend_roll_rad = math.atan2(s, dz)

    if abs(roll_rad) > 20 / 180 * math.pi:
        roll_rad = 20 / 180 * math.pi * roll_rad / abs(roll_rad)

    if abs(pitch_rad) > 20 / 180 * math.pi:
        pitch_rad = 20 / 180 * math.pi * pitch_rad / abs(pitch_rad)

def send_command(thrust, wx, wy, wz, duration):
    pass


if __name__ == "__main__":
    
    # Argparse
    parser = argparse.ArgumentParser(description="ETH Tracking Node for Constant Position")
    parser.add_argument("--hz", type=int, default=50, help="Frequency of the control loop")

    args = parser.parse_args()
    hz = args.hz
    
    print("#####################################################")
    print("## ETH Tracking Node for Constant Position Started ##")
    print("#####################################################")
    
    rospy.init_node('eth_tracking_node', anonymous=True)
    rate = rospy.Rate(hz)

    ##### Subscribers #####
    # Vehicle
    vel_state_sub = rospy.Subscriber('mavros/state', State, callback=state_cb)
    vel_imu_sub = rospy.Subscriber('mavros/imu/data', Imu, callback=imu_cb)
    vel_pos_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, callback=local_pos_cb)
    vel_body_sub = rospy.Subscriber('mavros/local_position/velocity_body', TwistStamped, callback=vel_body_cb)
    # Pendulum
    pend_pos_sub = rospy.Subscriber('pend_pos', PoseStamped, callback=pend_pos_cb)
    
    ##### Publishers #####
    # Vehicle
    vel_set_pos_pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)

    ##### Arming and Mode #####
    rospy.wait_for_service("/mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)

    rospy.wait_for_service("/mavros/set_mode")
    set_mode_client = rospy.ServiceProxy("mavros/set_mode", SetMode)
    
    