"""
This script is used to track the position of the drone using the ETH controller, 
based on the paper "A Flying Inverted Pendulum" by Raffaello D'Andrea and Markus Hehn.

This is for testing the nominal controller using theta and phi, the pitch and roll of pendulum,
instead of using the r and s positions of the pendulum.
"""
#!/usr/bin/env python

import os
import math
import datetime
import time
import IPython

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
from src.controllers.eth_constant_torque_controller import ConstantPositionTorqueTracker
from src.callbacks.fcu_state_callbacks import VehicleStateCB
from src.callbacks.pend_state_callbacks import PendulumCB
from src.callbacks.fcu_modes import FcuModes

class ETHTrackingTorqueNode:
    def __init__(self, 
                 vehicle,
                 cont_type,
                 ang_vel_type,
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
            self.J = np.array([	# NOTE: danaus12_old
                [0.00320868, 0.00011707,  0.00004899],
                [0.00011707, 0.00288707,  0.00006456],
                [0.00004899, 0.00006456,  0.00495141]])
            self.J_inv = np.linalg.inv(self.J)
        elif self.vehicle == "danaus12_newold":
            self.pend_z_pose = 0.721659
            # self.pend_z_pose = 0.69
            # self.L = 0.69
            self.L = 0.79
            self.mass = 0.70034104 + 0.046 #0.03133884
        
        self.cont_type = cont_type              # "rs" or "tp"
        self.ang_vel_type = ang_vel_type        # "euler" or "body"

        self.mode = mode                        # "sim" or "real"
        self.hz = hz                            # Control Loop Frequency
        self.track_type = track_type            # "constant" or "circular"
        self.takeoff_height = takeoff_height    # Takeoff Height
        
        self.Q = Q                              # State cost matrix
        self.R = R                              # Input cost matrix
        self.dt = 1/self.hz                     # Time step
        self.lqr_itr = lqr_itr                  # Number of iterations for Infinite-Horizon LQR
        self.prev_bodyrate = np.zeros((3,1))
        self.accum_bodyrate = np.zeros((3,1))

        self.nx = 16
        self.nu = 4

        self.pend_upright_time = pend_upright_time  # Time to keep the pendulum upright
        self.pend_upright_tol = pend_upright_tol    # Tolerance for pendulum relative position [r,s] (norm in meters)

        ### Subscribers ###
        self.quad_cb = VehicleStateCB(mode=self.mode)
        self.pend_cb = PendulumCB(mode=self.mode, vehicle=vehicle)
        ### Services ###
        self.quad_modes = FcuModes()
        ### Publishers ###
        self.quad_pose_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.quad_att_setpoint_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)

        # Time-checking
        self.time_pub = rospy.Publisher("time_check", String, queue_size=10)

        self.hover_thrust = None

        ### Goal Position ###
        takeoff_pose = np.array([0, 0, self.takeoff_height])

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
            ### Actual controller ###
            self.cont = ConstantPositionTorqueTracker(cont_type, ang_vel_type, self.J, self.L, self.Q, self.R, np.array([0, 0, self.takeoff_height]), self.dt)
            self.cont_K_inf = self.cont.infinite_horizon_LQR(self.lqr_itr)
            self.xgoal = self.cont.xgoal
            self.ugoal = self.cont.ugoal
            ### ETH controller for comparison ###
                                # γ, β, α, wx, wy, wz, x_dot, y_dot, z_dot, θ, ϕ, θ_dot, ϕ_dot
            Q = 1.0 * np.diag([0.0, 0.0, 0.0, 2, 2, 3, 1.0, 1.0, 1.0, 2.0, 2.0, 0.4, 0.4])
            R = 1.0 * np.diag([1, 7, 7, 7])
            self.eth_cont = ConstantPositionTracker(cont_type, self.L, Q, R, np.array([0, 0, self.takeoff_height]), self.dt)
            self.eth_K_inf = self.eth_cont.infinite_horizon_LQR(self.lqr_itr)
            self.eth_xgoal = self.eth_cont.xgoal
            self.eth_ugoal = self.eth_cont.ugoal
        else:
            raise ValueError("Invalid tracking type. Circualr not implemented.")
        
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

    def _quad_takeoff_controller(self):
        quad_xyz = self.quad_cb.get_xyz_pose()
        quad_xyz_vel = self.quad_cb.get_xyz_velocity()
        quad_xyz_ang = self.quad_cb.get_xyz_angles()
        if self.cont_type == "rs":
            pend_pos = np.zeros((2,))
            pend_vel = np.zeros((2,))
        elif self.cont_type == "tp":
            pend_pos = np.zeros((2,))
            pend_vel = np.zeros((2,))
            
        x = np.concatenate((quad_xyz_ang.T, quad_xyz.T, quad_xyz_vel.T, pend_pos.T, pend_vel.T))
        x = x.reshape((13, 1))
        u_takeoff = self.takeoff_input - self.takeoff_K_inf @ (x - self.takeoff_goal)
        return u_takeoff

    def _send_attitude_setpoint(self, u):
        self.prev_bodyrate = u[1:]

        self.att_setpoint.header.stamp = rospy.Time.now()
        self.att_setpoint.body_rate.x = u[1]
        self.att_setpoint.body_rate.y = u[2]
        self.att_setpoint.body_rate.z = u[3]
        self.att_setpoint.thrust = (u[0]/(9.81)) * self.hover_thrust
        self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)
        self.quad_att_setpoint_pub.publish(self.att_setpoint) 

    def _quad_lqr_torque_input(self, x):
        u_torque = self.ugoal - self.cont_K_inf @ (x - self.xgoal)
        return u_torque
    
    def _eth_lqr_bodyrate_input(self, x):
        x_eth = np.concatenate((x[0:3], x[6:]))
        u_eth = self.eth_ugoal - self.eth_K_inf @ (x_eth - self.eth_xgoal)
        return u_eth
    
    def _bodyrate_dynamics(self, u_torque, omega):
        ### Convert torque to bodyrate ###
        # IPython.embed()
        omega_hat = np.array([[0, -omega[2], omega[1]],
                              [omega[2], 0, -omega[0]],
                              [-omega[1], omega[0], 0]])
        omega = omega.reshape((3,1))
        # IPython.embed()
        omega_dot = self.J_inv @ (u_torque[1:] - omega_hat @ (self.J @ omega))
        return omega_dot.squeeze()  # convert to (3,1) shape
 
    def _rk4(self, u_torque, omega):
        k1 = self.dt * self._bodyrate_dynamics(u_torque, omega)
        k2 = self.dt * self._bodyrate_dynamics(u_torque, omega + 0.5*k1)
        k3 = self.dt * self._bodyrate_dynamics(u_torque, omega + 0.5*k2)
        k4 = self.dt * self._bodyrate_dynamics(u_torque, omega + k3)
        omega_next = omega + (k1 + 2*k2 + 2*k3 + k4) / 6
        return omega_next

    def _torque_to_bodyrate(self, u_torque, omega):
        bodyrate = self._rk4(u_torque, omega)
        # bodyrate = self._rk4(u_torque, self.prev_bodyrate.squeeze())
        u_bodyrate = np.array([u_torque[0].item(), bodyrate[0], bodyrate[1], 0])
        return u_bodyrate
        
        # omega_dot = self._bodyrate_dynamics(u_torque, omega)
        ### Option 1 ###
        # u_bodyrate[1:] = self.prev_bodyrate + self.dt * omega_dot
        ### Option 2 ###
        # u_bodyrate[1:] = self.dt * omega_dot
        ### Option 3 ###
        # u_bodyrate = self.prev_bodyrate + self.dt * omega_dot

        # self.prev_bodyrate = u_bodyrate[1:]
        # self.accum_bodyrate = self.accum_bodyrate * 0.9 + u_bodyrate[1:] * 0.1
        # u_bodyrate = np.array([u_bodyrate[0], self.accum_bodyrate[0], self.accum_bodyrate[1], self.accum_bodyrate[2]])

        # return u_bodyrate
    
    def _get_states(self):
        quad_xyz = self.quad_cb.get_xyz_pose()
        quad_xyz_vel = self.quad_cb.get_xyz_velocity()
        quad_xyz_ang = self.quad_cb.get_xyz_angles()
        quad_xyz_ang_vel = self.quad_cb.get_xyz_angular_velocity()   # TODO: Need to verify
        omega = self.quad_cb.get_xyz_angular_velocity_body()

        if self.cont_type == "tp":
            pend_pos = self.pend_cb.get_rs_ang(vehicle_pose=quad_xyz)
            pend_vel = self.pend_cb.get_rs_ang_vel(vehicle_vel=quad_xyz_vel)
        else:
            NotImplementedError(f"{self.cont_type} not implemented in NCBFTrackingNode")

        # γ, β, α, γ_dot, β_dot, α_dot, x, y, z, x_dot, y_dot, z_dot, θ, ϕ, θ_dot, ϕ_dot
        x_16 = np.concatenate((quad_xyz_ang.T, omega.T, quad_xyz.T, quad_xyz_vel.T, pend_pos.T, pend_vel.T)) # Torque LQR expects (theta, phi).
        x_16 = x_16.reshape((self.nx, 1))

        return x_16, omega

    def _takeoff_sequence(self):
        target_pose = np.array([self.takeoff_pose.pose.position.x, self.takeoff_pose.pose.position.y, self.takeoff_pose.pose.position.z])

        # Send initial commands
        for _ in range(100):
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
            ### Keep the quadrotor at this pose! ###
            u_takeoff = self._quad_takeoff_controller()     # This is better than takeoff_pose
            self._send_attitude_setpoint(u_takeoff)
            
            link_state = LinkState()
            # link_state.pose.position.x = -0.001995
            # link_state.pose.position.y = 0.000135
            link_state.pose.position.x = 0.001995
            link_state.pose.position.y = -0.000135
            link_state.pose.position.z = self.pend_z_pose
            link_state.link_name = self.vehicle+'::pendulum'
            link_state.reference_frame = 'base_link'
            _ = self.set_link_state_service(link_state)

            # Get the position of the pendulumu_bodyrate
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
                    
                    ### Turn gravity on! ###
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
                    u_takeoff = self._quad_takeoff_controller()
                    self._send_attitude_setpoint(u_takeoff) 
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
        eth_input_log = np.zeros((self.nu, num_itr))
        bodyrate_input_log = np.zeros((self.nu, num_itr))
        torque_input_log = np.zeros((self.nu, num_itr))
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
            for _ in range(500):
                # self._time_calls()

                link_state = LinkState()
                # link_state.pose.position.x = -0.001995
                # link_state.pose.position.y = 0.000135
                link_state.pose.position.x = 0.001995
                link_state.pose.position.y = -0.000135
                link_state.pose.position.z = self.pend_z_pose
                link_state.link_name = self.vehicle+'::pendulum'
                link_state.reference_frame = 'base_link'
                _ = self.set_link_state_service(link_state)
                
                # self.quad_pose_pub.publish(self.takeoff_pose)
                u_takeoff = self._quad_takeoff_controller()     # This is better than takeoff_pose
                self._send_attitude_setpoint(u_takeoff)
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
            
            # if (itr % int(0.3*self.hz) == 0):
            #     self.accum_bodyrate = np.zeros((3,1))

            # Quad XYZ
            quad_xyz = self.quad_cb.get_xyz_pose()
            # Quad XYZ Velocity
            quad_xyz_vel = self.quad_cb.get_xyz_velocity()
            # Pendulum RS
            if self.cont_type == "rs":
                pend_pos = self.pend_cb.get_rs_pose(vehicle_pose=quad_xyz)
                pend_vel = self.pend_cb.get_rs_vel(vehicle_vel=quad_xyz_vel)
                if np.linalg.norm(pend_pos) > 0.3:
                    u_takeoff = self._quad_takeoff_controller()
                    self._send_attitude_setpoint(u_takeoff)
                    rospy.loginfo_throttle(3,"Pendulum too far away. Exiting the control loop.")
                    continue
            elif self.cont_type == "tp":
                pend_pos = self.pend_cb.get_rs_ang(vehicle_pose=quad_xyz)
                pend_vel = self.pend_cb.get_rs_ang_vel(vehicle_vel=quad_xyz_vel)
                if np.linalg.norm(pend_pos) > np.pi/3:
                    u_takeoff = self._quad_takeoff_controller()
                    self._send_attitude_setpoint(u_takeoff)
                    rospy.loginfo_throttle(3,"Pendulum too far away. Exiting the control loop.")
                    continue

            # State Vector
            x, omega = self._get_states()

            # Control Input
            u_eth = self._eth_lqr_bodyrate_input(x)
            u_torque = self._quad_lqr_torque_input(x)
            u_bodyrate = self._torque_to_bodyrate(u_torque, omega)
            self._send_attitude_setpoint(u_bodyrate)
            # self._send_attitude_setpoint(u_eth)

            # Log the state and input
            state_log[:, itr] = x.flatten()
            eth_input_log[:, itr] = u_eth.flatten()
            bodyrate_input_log[:, itr] = u_bodyrate.flatten()
            torque_input_log[:, itr] = u_torque.flatten()
            error_log[:, itr] = (x - self.xgoal).flatten()

            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log, eth_input_log, bodyrate_input_log, torque_input_log, error_log


if __name__ == "__main__":
    
    ##### Argparse #####
    parser = argparse.ArgumentParser(description="ETH Tracking Node for Constant Position")
    parser.add_argument("--mode", type=str, default="sim", help="Mode of operation (sim or real)")
    parser.add_argument("--hz", type=int, default=90, help="Frequency of the control loop")
    parser.add_argument("--track_type", type=str, default="constant", help="Type of tracking to be used")
    parser.add_argument("--takeoff_height", type=float, default=1.5, help="Height to takeoff to (in meters)")
    parser.add_argument("--pend_upright_time", type=float, default=0.5, help="Time to keep the pendulum upright")
    parser.add_argument("--pend_upright_tol", type=float, default=0.05, help="Tolerance for pendulum relative position [r,z] (norm in meters)")
    parser.add_argument("--lqr_itr", type=int, default=100000, help="Number of iterations for Infinite-Horizon LQR")
    parser.add_argument("--cont_duration", type=int, default=20, help="Duration for which the controller should run (in seconds)")
    parser.add_argument("--vehicle", type=str)
    parser.add_argument("--cont_type", type=str, help="Either rs or tp")    # rs: r and s, tp: theta and phi
    parser.add_argument("--ang_vel_type", type=str, default="body", help="Type of angular velocity to be used (euler or body)")
    parser.add_argument("--save", type=bool, default=True, help="Save the logs")

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
    ang_vel_type = args.ang_vel_type
    save = args.save
    

    print("#####################################################")
    print("## ETH Tracking Node for Constant Position Started ##")
    print("#####################################################")
    print("")
    
                        # x  y  z  x_dot y_dot z_dot yaw pitch roll r s r_dot s_dot
    # Q = 1.0 * np.diag([2, 2, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.4, 0.4])      # With pendulum
    
    if cont_type == "rs":
        raise NotImplementedError("rs not implemented in ETHTrackingTorqueNode")
                           # γ, β, α, x, y, z, x_dot, y_dot, z_dot, r, s, r_dot, s_dot
        Q = 1.0 * np.diag([0.0, 0.0, 0.0, 2, 2, 2, 0.0, 0.0, 0.0, 2.0, 2.0, 0.4, 0.4])
        R = 1.0 * np.diag([1, 7, 7, 7])
    elif cont_type == "tp":
                        # γ, β, α, γ_dot (wx), β_dot (wy), α_dot (wz), x, y, z, x_dot, y_dot, z_dot, θ, ϕ, θ_dot, ϕ_dot
        Q = 1.0 * np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, 2, 3, 1.0, 1.0, 1.0, 2.0, 2.0, 0.4, 0.4])    # Stable controller
        # Q = 1.0 * np.diag([0.0, 0.0, 0.0, 2, 2, 3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])      # Unstable as hell
        # Q = 1.0 * np.diag([0.0, 0.0, 0.0, 8, 8, 8, 0.0, 0.0, 0.0, 1.0, 1.0, 0.4, 0.4])      
        R = 1.0 * np.diag([1, 7, 7, 7])     # Stable controller
        # R = 1.0 * np.diag([1, 1, 1, 1])

    eth_node = ETHTrackingTorqueNode(vehicle, cont_type, ang_vel_type, mode, hz, track_type, Q, R, 
                               takeoff_height=takeoff_height, 
                               lqr_itr=lqr_itr, 
                               pend_upright_time=pend_upright_time, 
                               pend_upright_tol=pend_upright_tol)
    
    state_log, eth_input_log, bodyrate_input_log, torque_input_log, error_log = eth_node.run(duration=cont_duration)

    print("#####################################################")
    print("### ETH Tracking Node for Constant Position Over  ###")
    print("#####################################################")
    print("")

    #################################
    ######### Save the logs #########
    #################################

    if save:
        save_dir = "/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/torque"

        # Get current date and time
        current_time = datetime.datetime.now()
        # Format the date and time into the desired filename format
        formatted_time = current_time.strftime("%m%d_%H%M%S-Torque_Body-Sim")
        directory_path = os.path.join(save_dir, formatted_time)
        os.makedirs(directory_path, exist_ok=True)

        np.save(os.path.join(directory_path, "log.npy"),
            {
                "state_log": state_log,
                "eth_input_log": eth_input_log,
                "bodyrate_input_log": bodyrate_input_log,
                "torque_input_log": torque_input_log,
                "error_log": error_log
            })
        np.save(os.path.join(directory_path, "params.npy"), 
            {
                "mode": mode,
                "hz": hz,
                "track_type": track_type,
                "takeoff_height": takeoff_height,
                "pend_upright_time": pend_upright_time,
                "pend_upright_tol": pend_upright_tol,
                "lqr_itr": lqr_itr,
                "cont_duration": cont_duration,
                "vehicle": vehicle,
                "cont_type": cont_type,
                "ang_vel_type": ang_vel_type,
                "Q": Q,
                "R": R
            })

        print("#####################################################")
        print(f"########### LOG DATA SAVED IN {formatted_time} ###########")
        print("#####################################################")

