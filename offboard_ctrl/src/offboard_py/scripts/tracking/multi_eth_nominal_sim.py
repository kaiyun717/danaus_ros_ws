"""
This script is used to track the position of the drone using the ETH controller, 
based on the paper "A Flying Inverted Pendulum" by Raffaello D'Andrea and Markus Hehn.

This is for testing the nominal controller using theta and phi, the pitch and roll of pendulum,
instead of using the r and s positions of the pendulum.
"""
#!/usr/bin/env python3

import os
import math
import datetime
import time

import rospy
import numpy as np
import scipy

import argparse

from tf.transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import String

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

class ETHTrackingNode:
    def __init__(self, 
                 vehicle,
                 cont_type,
                 mode,
                 hz, 
                 track_type,
                 Q, 
                 R, 
                 takeoff_height=0.5, 
                 lqr_itr=1000,
                 pend_upright_time=0.5,
                 pend_upright_tol=0.05) -> None:
        
        self.vehicle = vehicle
        if self.vehicle == "danaus12_old":
            self.pend_z_pose = 0.531659
            self.mass = 0.67634104 + 0.03133884
            self.L = 0.5
        elif self.vehicle == "danaus12_newold":
            self.pend_z_pose = 0.721659
            # self.pend_z_pose = 0.69
            # self.L = 0.69
            self.L = 0.79
            self.mass = 0.70034104 + 0.046 #0.03133884
        
        self.cont_type = cont_type              # "rs" or "tp"

        self.mode = mode                        # "sim" or "real"
        self.hz = hz                            # Control Loop Frequency
        self.track_type = track_type            # "constant" or "circular"
        self.takeoff_height = takeoff_height    # Takeoff Height
        
        self.Q = Q                              # State cost matrix
        self.R = R                              # Input cost matrix
        self.dt = 1/self.hz                     # Time step
        self.lqr_itr = lqr_itr                  # Number of iterations for Infinite-Horizon LQR

        self.nx = 13
        self.nu = 4

        self.pend_upright_time = pend_upright_time  # Time to keep the pendulum upright
        self.pend_upright_tol = pend_upright_tol    # Tolerance for pendulum relative position [r,z] (norm in meters)

        ### Subscribers ###
        self.quad_cb = VehicleStateCB(mode=self.mode)
        self.pend_cb = PendulumCB(mode=self.mode, vehicle=vehicle)
        ### Services ###
        self.quad_modes = FcuModes()
        ### Publishers ###
        self.quad_pose_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=20)
        self.quad_att_setpoint_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=20)

        # Time-checking
        self.time_pub = rospy.Publisher("time_check", String, queue_size=20)

        self.hover_thrust = None

        ### Goal Position ###
        takeoff_pose = np.array([3, 3, self.takeoff_height])

        ### Takeoff Controller ###
                                # γ, β, α, x, y, z, x_dot, y_dot, z_dot, pendulum (4)
        Q_takeoff = 1.0 * np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])      # Without pendulum
        R_takeoff = 1.0 * np.diag([1, 10, 10, 10])
        self.takeoff_cont = ConstantPositionTracker(cont_type, self.L, Q_takeoff, R_takeoff, takeoff_pose, self.dt)
        self.takeoff_K_inf = self.takeoff_cont.infinite_horizon_LQR(self.lqr_itr)
        self.takeoff_goal = self.takeoff_cont.xgoal
        self.takeoff_input = self.takeoff_cont.ugoal

        ### Tracking Controller ###
        if self.track_type == "constant":
            self.cont = ConstantPositionTracker(cont_type, self.L, self.Q, self.R, np.array([0, 0, self.takeoff_height]), self.dt)
            self.cont_K_inf = self.cont.infinite_horizon_LQR(self.lqr_itr)
            self.xgoal = self.cont.xgoal
            self.ugoal = self.cont.ugoal
        else:
            raise ValueError("Invalid tracking type. Circular not implemented.")
        
        ### Takeoff Pose ###
        self.takeoff_pose = PoseStamped()
        self.takeoff_pose.pose.position.x = 0
        self.takeoff_pose.pose.position.y = 0
        self.takeoff_pose.pose.position.z = self.takeoff_height
        
        ### Attitude Setpoint ###
        self.att_setpoint = AttitudeTarget()
        self.att_setpoint.header = Header()
        self.att_setpoint.header.frame_id = "base_footprint"
        self.att_setpoint.type_mask = 128  # Ignore attitude/orientation

        ### Initialize the node ###
        self._init_node()
        self.rate = rospy.Rate(self.hz)

    def _init_node(self):
        rospy.init_node('eth_tracking_node', anonymous=True)

    def _quad_lqr_controller(self):
        quad_xyz = self.quad_cb.get_xyz_pose()
        quad_xyz_vel = self.quad_cb.get_xyz_velocity()
        quad_xyz_ang = self.quad_cb.get_xyz_angles()
        if self.cont_type == "rs":
            # pend_pos = self.pend_cb.get_rs_pose(vehicle_pose=quad_xyz)
            # pend_vel = self.pend_cb.get_rs_vel(vehicle_vel=quad_xyz_vel)
            pend_pos = np.zeros((2,))
            pend_vel = np.zeros((2,))
        elif self.cont_type == "tp":
            # pend_pos = self.pend_cb.get_rs_ang(vehicle_pose=quad_zyx_ang)
            # pend_vel = self.pend_cb.get_rs_ang_vel(vehicle_vel=quad_xyz_vel)
            pend_pos = np.zeros((2,))
            pend_vel = np.zeros((2,))
            
        x = np.concatenate((quad_xyz_ang.T, quad_xyz.T, quad_xyz_vel.T, pend_pos.T, pend_vel.T))
        x = x.reshape((self.nx, 1))
        u = self.takeoff_input - self.takeoff_K_inf @ (x - self.takeoff_goal)  # 0 - K * dx = +ve

        self.att_setpoint.header.stamp = rospy.Time.now()
        self.att_setpoint.body_rate.x = u[1]
        self.att_setpoint.body_rate.y = u[2]
        self.att_setpoint.body_rate.z = u[3]
        self.att_setpoint.thrust = (u[0]/(9.81)) * self.hover_thrust
        self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)
        self.quad_att_setpoint_pub.publish(self.att_setpoint) 

    def _takeoff_sequence(self):
        
        target_pose = np.array([self.takeoff_pose.pose.position.x, self.takeoff_pose.pose.position.y, self.takeoff_pose.pose.position.z])

        # Send initial commands
        for _ in range(1000):
            if rospy.is_shutdown():
                break
            
            self.quad_pose_pub.publish(self.takeoff_pose)
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

            self.quad_pose_pub.publish(self.takeoff_pose)
            # self._quad_lqr_controller(thrust_ratio=0.45)     # This is better than takeoff_pose

            self.rate.sleep()
        
        self.hover_thrust = self.quad_cb.thrust

        if self.hover_thrust < 0.3:
            rospy.loginfo("Hover thrust too low. Exiting the node.")
            self.hover_thrust = 0.45

        rospy.loginfo("Recorded hover thrust: {}".format(self.hover_thrust))
        rospy.loginfo("Takeoff pose achieved!")

    def _pend_upright_sim(self, req_time=0.5, tol=0.05):
        get_link_properties_service = rospy.ServiceProxy('/gazebo/get_link_properties', GetLinkProperties)
        self.set_link_state_service = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        set_link_properties_service = rospy.ServiceProxy('/gazebo/set_link_properties', SetLinkProperties)

        consecutive_time = rospy.Duration(0.0)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            # self._time_calls()

            # # Keep the quadrotor at this pose!
            # self.quad_pose_pub.publish(self.takeoff_pose)  
            
            # self._quad_lqr_controller()     # This is better than takeoff_pose
            
            link_state = LinkState()
            # link_state.pose.position.x = -0.001995
            # link_state.pose.position.y = 0.000135
            # link_state.pose.position.x = 0.001995
            # link_state.pose.position.y = -0.000135
            link_state.pose.position.x = -0.001995
            link_state.pose.position.y = 0.000135
            link_state.pose.position.z = self.pend_z_pose
            link_state.link_name = self.vehicle+'::pendulum'
            link_state.reference_frame = 'base_link'
            _ = self.set_link_state_service(link_state)

            # Get the position of the pendulum
            quad_xyz = self.quad_cb.get_xyz_pose()
            pendulum_position = self.pend_cb.get_rs_pose(vehicle_pose=quad_xyz)

            # Calculate the norm of the position
            position_norm = np.linalg.norm(pendulum_position)
            print(f"{position_norm=}")

            # Check if the norm is less than 0.05m
            if position_norm < tol:
                consecutive_time += rospy.Time.now() - start_time
                if consecutive_time >= rospy.Duration(req_time):
                    rospy.loginfo("Pendulum position has been less than 0.05m for 0.5 seconds straight.")
                    
                    # Turn gravity on!
                    curr_pend_properties = get_link_properties_service(link_name=self.vehicle+'::pendulum')
                    new_pend_properties = SetLinkPropertiesRequest(
                                            link_name=self.vehicle+'::pendulum',
                                            gravity_mode=True,
                                            com=curr_pend_properties.com,
                                            mass=curr_pend_properties.mass,
                                            ixx=curr_pend_properties.ixx,
                                            ixy=curr_pend_properties.ixy,
                                            ixz=curr_pend_properties.ixz,
                                            iyy=curr_pend_properties.iyy,
                                            iyz=curr_pend_properties.iyz,
                                            izz=curr_pend_properties.izz
                                        )
                    set_link_properties_service(new_pend_properties)
                    self._quad_lqr_controller() 
                    return True
                self.rate.sleep()
            else:
                consecutive_time = rospy.Duration(0.0)
                start_time = rospy.Time.now()
                self.rate.sleep()

            self.rate.sleep()
        
    def _time_calls(self):
        current_sim_time = rospy.get_time()
        current_real_time = time.time()
        time_msg = String()
        time_msg.data = f"Sim Time: {current_sim_time}, Real Time: {current_real_time}"
        self.time_pub.publish(time_msg)
        rospy.loginfo(time_msg.data)

    def run(self, duration):
        # Log Array
        num_itr = int(duration * self.hz)
        state_log = np.zeros((self.nx, num_itr))
        input_log = np.zeros((self.nu, num_itr))
        error_log = np.zeros((self.nx, num_itr))

        ##### Takeoff Sequence #####
        self._takeoff_sequence()
        
        # ##### Pendulum Mounting #####
        if self.mode == "real":
            # Sleep for 5 seconds so that pendulum can be mounted
            rospy.loginfo("Sleeping for 5 seconds to mount the pendulum.")
            rospy.sleep(5)

            # Keep the pendulum upright
            rospy.loginfo("Keeping the pendulum upright.")
            self._pend_upright_real(req_time=self.pend_upright_time, tol=self.pend_upright_tol)
        elif self.mode == "sim":
            rospy.loginfo("Swing the pendulum upright.")
            self._pend_upright_sim(req_time=self.pend_upright_time, tol=self.pend_upright_tol)
        
        ##### Setting near origin & upright #####
        if self.mode == "sim":
            rospy.loginfo("Setting near origin & upright")
            for _ in range(1):
                # self._time_calls()

                link_state = LinkState()
                #link_state.pose.position.x = -0.001995
                #link_state.pose.position.y = 0.000135
                link_state.pose.position.x = 0.001995
                link_state.pose.position.y = -0.000135
                #link_state.pose.position.x = 0 ## TODO THESE NEED TO CHANGE! Don't publish a global position, use the drone's position plus the relative distance in x,y,z
                #link_state.pose.position.y = 0
                link_state.pose.position.z = self.pend_z_pose
                link_state.link_name = self.vehicle+'::pendulum'
                link_state.reference_frame = 'base_link'
                _ = self.set_link_state_service(link_state)
                
                # self.quad_pose_pub.publish(self.takeoff_pose)
                self._quad_lqr_controller()     # This is better than takeoff_pose
                self.rate.sleep()
        else:
            pass
        
        ##### Pendulum Position Control #####
        rospy.loginfo("Starting the constant position control!")
        start_time = rospy.Time.now()

        # for itr in range(int(1e10)):  # If you want to test with console.
        for itr in range(num_itr):
            # self._time_calls()

            if rospy.is_shutdown():
                rospy.loginfo_throttle(3, "Node shutdown detected. Exiting the control loop.")
                break
            
            # Quad XYZ
            quad_xyz = self.quad_cb.get_xyz_pose()
            # Quad XYZ Velocity
            quad_xyz_vel = self.quad_cb.get_xyz_velocity()
            # Quad ZYX Angles
            quad_xyz_ang = self.quad_cb.get_xyz_angles()
            # Pendulum RS
            if self.cont_type == "rs":
                pend_pos = self.pend_cb.get_rs_pose(vehicle_pose=quad_xyz)
                pend_vel = self.pend_cb.get_rs_vel(vehicle_vel=quad_xyz_vel)
                if np.linalg.norm(pend_pos) > 0.3:
                    self._quad_lqr_controller()
                    rospy.loginfo_throttle(3,"Pendulum too far away. Exiting the control loop.")
                    continue
            elif self.cont_type == "tp":
                pend_pos = self.pend_cb.get_rs_ang(vehicle_pose=quad_xyz)
                pend_vel = self.pend_cb.get_rs_ang_vel(vehicle_vel=quad_xyz_vel)
                if np.linalg.norm(pend_pos) > np.pi/3:
                    self._quad_lqr_controller()
                    rospy.loginfo_throttle(3,"Pendulum too far away. Exiting the control loop.")
                    continue

            # State Vector
            x = np.concatenate((quad_xyz_ang.T, quad_xyz.T, quad_xyz_vel.T, pend_pos.T, pend_vel.T))
            x = x.reshape((self.nx, 1))

            # Control Input
            u = self.ugoal - self.cont_K_inf @ (x - self.xgoal)  # 0 - K * dx = +ve

            # Publish the attitude setpoint
            self.att_setpoint.header.stamp = rospy.Time.now()
            self.att_setpoint.body_rate.x = u[1]    # np.clip(u[0], -20, 20)
            self.att_setpoint.body_rate.y = u[2]    # np.clip(u[1], -20, 20)
            self.att_setpoint.body_rate.z = u[3]    # np.clip(u[2], -20, 20)
            self.att_setpoint.thrust = (u[0]/(9.81)) * self.hover_thrust
            self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)
            self.quad_att_setpoint_pub.publish(self.att_setpoint) # Uncomment this line to publish the attitude setpoint
            
            # Log the state and input
            state_log[:, itr] = x.flatten()
            input_log[:, itr] = u.flatten()
            error_log[:, itr] = (x - self.xgoal).flatten()

            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log, input_log, error_log


if __name__ == "__main__":
    
    ##### Argparse #####
    parser = argparse.ArgumentParser(description="ETH Tracking Node for Constant Position")
    parser.add_argument("--mode", type=str, default="sim", help="Mode of operation (sim or real)")
    parser.add_argument("--hz", type=int, default=90, help="Frequency of the control loop")
    parser.add_argument("--track_type", type=str, default="constant", help="Type of tracking to be used")
    parser.add_argument("--takeoff_height", type=float, default=2.0, help="Height to takeoff to (in meters)")
    parser.add_argument("--pend_upright_time", type=float, default=0.5, help="Time to keep the pendulum upright")
    parser.add_argument("--pend_upright_tol", type=float, default=0.05, help="Tolerance for pendulum relative position [r,z] (norm in meters)")
    parser.add_argument("--lqr_itr", type=int, default=100000, help="Number of iterations for Infinite-Horizon LQR")
    parser.add_argument("--cont_duration", type=int, default=20, help="Duration for which the controller should run (in seconds)")
    parser.add_argument("--vehicle", type=str)
    parser.add_argument("--cont_type", type=str, help="Either rs or tp")    # rs: r and s, tp: theta and phi

    args = parser.parse_args()
    mode = args.mode
    hz = args.hz
    track_type = args.track_type
    takeoff_height = args.takeoff_height
    pend_upright_time = args.pend_upright_time
    pend_upright_tol = args.pend_upright_tol
    lqr_itr = args.lqr_itr
    cont_duration = args.cont_duration
    vehicle = args.vehicle
    cont_type = args.cont_type

    print("#####################################################")
    print("## ETH Tracking Node for Constant Position Started ##")
    print("#####################################################")
    print("")
    
                        # x  y  z  x_dot y_dot z_dot yaw pitch roll r s r_dot s_dot
    # Q = 1.0 * np.diag([2, 2, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.4, 0.4])      # With pendulum
    
    if cont_type == ".":
                           # γ, β, α, x, y, z, x_dot, y_dot, z_dot, r, s, r_dot, s_dot
        Q = 1.0 * np.diag([0.0, 0.0, 0.0, 2, 2, 2, 0.0, 0.0, 0.0, 2.0, 2.0, 0.4, 0.4])
        R = 1.0 * np.diag([1, 7, 7, 7])
    elif cont_type == "tp":
                           # γ, β, α, x, y, z, x_dot, y_dot, z_dot, θ, ϕ, θ_dot, ϕ_dot
        Q = 1.0 * np.diag([0.0, 0.0, 0.0, 2, 2, 3, 10.0, 10.0, 1.0, 2, 2, 0.4, 0.4])    # Stable controller
        # Q = 1.0 * np.diag([0.0, 0.0, 0.0, 2, 2, 3, 1.0, 1.0, 1.0, 2.0, 2.0, 0.4, 0.4])    # Stable controller
        # Q = 1.0 * np.diag([0.0, 0.0, 0.0, 2, 2, 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])      # Unstable as hell
        # Q = 1.0 * np.diag([0.0, 0.0, 0.0, 8, 8, 8, 0.0, 0.0, 0.0, 1.0, 1.0, 0.4, 0.4])      
        R = 1.0 * np.diag([1, 7, 7, 7])
        # R = 1.0 * np.diag([1, 1, 1, 1])

    eth_node = ETHTrackingNode(vehicle, cont_type, mode, hz, track_type, Q, R, 
                               takeoff_height=takeoff_height, 
                               lqr_itr=lqr_itr, 
                               pend_upright_time=pend_upright_time, 
                               pend_upright_tol=pend_upright_tol)
    
    state_log, input_log, error_log = eth_node.run(duration=cont_duration)

    print("#####################################################")
    print("### ETH Tracking Node for Constant Position Over  ###")
    print("#####################################################")
    print("")

    # #################################
    # ######### Save the logs #########
    # #################################

    # # curr_dir = os.getcwd()
    # save_dir = "/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs"

    # # Get current date and time
    # current_time = datetime.datetime.now()
    # # Format the date and time into the desired filename format
    # formatted_time = current_time.strftime("%m%d_%H%M%S-Sim")
    # directory_path = os.path.join(save_dir, formatted_time)
    # os.makedirs(directory_path, exist_ok=True)

    # np.save(os.path.join(directory_path, "state.npy"), state_log)
    # np.save(os.path.join(directory_path, "input.npy"), input_log)
    # np.save(os.path.join(directory_path, "error.npy"), error_log)
    # np.save(os.path.join(directory_path, "gains.npy"), {"Q": Q, "R": R})

    # print("#############################################0, 0, 0, 0, 0, 3, 0, 0, 0########")
    # print(f"########### LOG DATA SAVED IN {formatted_time} ###########")
    # print("#####################################################")

