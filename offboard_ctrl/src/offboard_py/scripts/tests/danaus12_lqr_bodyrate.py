"""
This script is used to test whether the most-updated Danaus12 quadrotor
works well in Gazebo simulation using LQR for bodyrates.
"""
#!/usr/bin/env python

import os
import math
import datetime
import IPython

import rospy
import numpy as np
import scipy

import argparse

from tf.transformations import euler_from_quaternion, quaternion_from_euler

from gazebo_msgs.srv import GetLinkProperties, SetLinkProperties, SetLinkState, SetLinkPropertiesRequest
from gazebo_msgs.msg import LinkState

from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from geometry_msgs.msg import PoseStamped, Quaternion, Vector3, TwistStamped
from std_msgs.msg import Header

from src.controllers.eth_constant_controller import ConstantPositionTracker
from src.callbacks.fcu_state_callbacks import VehicleStateCB
from src.callbacks.pend_state_callbacks import PendulumCB
from src.callbacks.fcu_modes import FcuModes

class Danaus12TestNode:
    def __init__(self, 
                 mode,
                 hz, 
                 track_type,
                 mass,
                 takeoff_height=0.5, 
                 lqr_itr=1000) -> None:
        
        self.mode = mode                        # "sim" or "real"
        self.hz = hz                            # Control Loop Frequency
        self.track_type = track_type            # "constant" or "circular"
        self.takeoff_height = takeoff_height    # Takeoff Height
        
        self.mass = mass                        # Mass of the quadrotor + pendulum
        self.dt = 1/self.hz                     # Time step
        self.lqr_itr = lqr_itr                  # Number of iterations for Infinite-Horizon LQR
        self.cont_duration = cont_duration      # Duration for which the controller should run (in seconds)

        self.nx = 13
        self.nu = 4

        ### Subscribers ###
        self.quad_cb = VehicleStateCB(mode=self.mode)
        ### Services ###
        self.quad_modes = FcuModes()
        ### Publishers ###
        self.quad_pose_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.quad_att_setpoint_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)

        self.hover_thrust = None

        ### Goal Position ###
        init_goal_pose = np.array([0, 0, self.takeoff_height])
        self.goal_poses = np.array([[1, 1, 0.5],[2, 3, 1.5],[-1, 1, 1.0],[-3, -4, 1.5],[1, -1, 0.5],[0, 0, self.takeoff_height]])

        ### LQR Controller ###
        Q_takeoff = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        R_takeoff = 1.0 * np.diag([10, 10, 10, 1])
        self.lqr_cont = ConstantPositionTracker(1.0, Q_takeoff, R_takeoff, init_goal_pose, self.dt)
        self.lqr_K_inf = self.lqr_cont.infinite_horizon_LQR(self.lqr_itr)
        self.lqr_goal = self.lqr_cont.xgoal
        self.lqr_input = self.lqr_cont.ugoal

        ### Attitude Setpoint ###
        self.att_setpoint = AttitudeTarget()
        self.att_setpoint.header = Header()
        self.att_setpoint.header.frame_id = "base_footprint"
        self.att_setpoint.type_mask = 128  # Ignore attitude/orientation

        ### Initialize the node ###
        self._init_node()
        self.rate = rospy.Rate(self.hz)

    def _init_node(self):
        rospy.init_node('danaus12_test_node', anonymous=True)

    def _quad_lqr_controller(self):
        quad_xyz = self.quad_cb.get_xyz_pose()
        quad_xyz_vel = self.quad_cb.get_xyz_velocity()
        quad_zyx_ang = self.quad_cb.get_zyx_angles()
        pend_rs = np.zeros((2))
        pend_rs_vel = np.zeros((2))
        
        # IPython.embed()
        x = np.concatenate((quad_xyz.T, quad_xyz_vel.T, quad_zyx_ang.T, pend_rs.T, pend_rs_vel.T))  
        x = x.reshape((self.nx, 1))
        u = self.lqr_input - self.lqr_K_inf @ (x - self.lqr_goal)  # 0 - K * dx = +ve

        self.att_setpoint.header.stamp = rospy.Time.now()
        self.att_setpoint.body_rate.x = u[0]
        self.att_setpoint.body_rate.y = u[1]
        self.att_setpoint.body_rate.z = u[2]
        self.att_setpoint.thrust = (u[3]/(9.81)) * self.hover_thrust
        self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)

        try:
            self.quad_att_setpoint_pub.publish(self.att_setpoint) 
        except:
            IPython.embed()

    def _takeoff_sequence(self):
        
        target_pose = PoseStamped()
        target_pose.pose.position.x = 0
        target_pose.pose.position.y = 0
        target_pose.pose.position.z = self.takeoff_height

        # Send initial commands
        for _ in range(100):
            if rospy.is_shutdown():
                break
            
            self.quad_pose_pub.publish(target_pose)
            # self._quad_lqr_controller()
            self.rate.sleep()
        
        # Arm and Offboard Mode
        last_req = rospy.Time.now()
        start_time = rospy.Time.now()

        if self.mode == "sim":
            _t = 20
        else:
            _t = 20

        while (not rospy.is_shutdown()) and (rospy.Time.now() - start_time) < rospy.Duration(_t):
            if (self.quad_cb.get_state().mode != "OFFBOARD") and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                offb_mode_srv_msg = self.quad_modes.set_offboard_mode()
                if (offb_mode_srv_msg.mode_sent == True):
                    rospy.loginfo("OFFBOARD enabled")
                last_req = rospy.Time.now()
            else:
                if (not self.quad_cb.get_state().armed) and (rospy.Time.now() - last_req) > rospy.Duration(5.0):
                    arming_resp = self.quad_modes.set_arm()
                    if (arming_resp.success == True):
                        rospy.loginfo("Vehicle armed")
                    else:
                        rospy.loginfo("Vehicle arming failed")
                        rospy.loginfo(arming_resp)
                    last_req = rospy.Time.now()

            self.quad_pose_pub.publish(target_pose)
            # self._quad_lqr_controller()     # This is better than takeoff_pose

            self.rate.sleep()
        
        self.hover_thrust = self.quad_cb.thrust

        if self.hover_thrust < 0.3:
            rospy.loginfo("Hover thrust too low. Exiting the node.")
            self.hover_thrust = 0.45

        rospy.loginfo("Recorded hover thrust: {}".format(self.hover_thrust))
        rospy.loginfo("Takeoff achieved!")

    def run(self, duration):
        # Log Array
        num_itr = int(duration * self.hz)
        state_log = np.zeros((self.nx, num_itr))
        input_log = np.zeros((self.nu, num_itr))
        error_log = np.zeros((self.nx, num_itr))

        ##### Takeoff Sequence #####
        self._takeoff_sequence()
        
        ##### Tracking for Danaus12 #####
        rospy.loginfo("Tracking Started!")
        start_time = rospy.Time.now()

        goal_idx = 0

        # for itr in range(int(1e10)):  # If you want to test with console.
        for itr in range(num_itr):
            if rospy.is_shutdown():
                rospy.loginfo_throttle(3, "Node shutdown detected. Exiting the control loop.")
                break
            
            if (itr%250 == 0):
                goal_idx = (goal_idx+1)%len(self.goal_poses)
                xyz_goal = self.goal_poses[goal_idx]
                self.lqr_goal = np.hstack((xyz_goal,np.zeros((10)))).reshape(self.nx,1)
            
            self._quad_lqr_controller()
            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log, input_log, error_log


if __name__ == "__main__":
    
    ##### Argparse #####
    parser = argparse.ArgumentParser(description="ETH Tracking Node for Constant Position")
    parser.add_argument("--mode", type=str, default="sim", help="Mode of operation (sim or real)")
    parser.add_argument("--hz", type=int, default=90, help="Frequency of the control loop")
    parser.add_argument("--track_type", type=str, default="constant", help="Type of tracking to be used")
    parser.add_argument("--mass", type=float, default=0.70, help="Mass of the quadrotor + pendulum (in kg)")
    # parser.add_argument("--mass", type=float, default=0.740, help="Mass of the quadrotor + pendulum (in kg)")
    parser.add_argument("--takeoff_height", type=float, default=1.5, help="Height to takeoff to (in meters)")
    parser.add_argument("--lqr_itr", type=int, default=100000, help="Number of iterations for Infinite-Horizon LQR")
    parser.add_argument("--cont_duration", type=int, default=50, help="Duration for which the controller should run (in seconds)")

    args = parser.parse_args()
    mode = args.mode
    hz = args.hz
    track_type = args.track_type
    mass = args.mass
    takeoff_height = args.takeoff_height
    lqr_itr = args.lqr_itr
    cont_duration = args.cont_duration

    print("#####################################################")
    print("## ETH Tracking Node for Constant Position Started ##")
    print("#####################################################")
    print("")
    
    danaus12_test_node = Danaus12TestNode(mode, hz, track_type, mass, 
                                          takeoff_height, lqr_itr)
    
    state_log, input_log, error_log = danaus12_test_node.run(duration=cont_duration)

    print("#####################################################")
    print("### ETH Tracking Node for Constant Position Over  ###")
    print("#####################################################")
    print("")

    #################################
    ######### Save the logs #########
    #################################

    # curr_dir = os.getcwd()
    save_dir = "/home/oem/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs"

    # Get current date and time
    current_time = datetime.datetime.now()
    # Format the date and time into the desired filename format
    formatted_time = current_time.strftime("%m%d_%H%M%S-Sim")
    directory_path = os.path.join(save_dir, formatted_time)
    os.makedirs(directory_path, exist_ok=True)

    np.save(os.path.join(directory_path, "state.npy"), state_log)
    np.save(os.path.join(directory_path, "input.npy"), input_log)
    np.save(os.path.join(directory_path, "error.npy"), error_log)
    np.save(os.path.join(directory_path, "gains.npy"), {"Q": Q, "R": R})

    print("#####################################################")
    print(f"########### LOG DATA SAVED IN {formatted_time} ###########")
    print("#####################################################")

