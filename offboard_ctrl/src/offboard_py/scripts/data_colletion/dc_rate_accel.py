"""
This script is used to gather data for body rate to angular acceleration mapping.

States: [gamma, beta, alpha, x, y, z, xdot, ydot, zdot]
Inputs: [a, wx, wy, wz]
"""
#!/usr/bin/env python

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

from src.controllers.quad_lqr_controller import QuadLQRController
from src.callbacks.fcu_state_callbacks import VehicleStateCB
from src.callbacks.fcu_modes import FcuModes

class DataCollectionRateAccel:
    def __init__(self, 
                 vehicle,
                 cont_type,
                 mode,
                 hz, 
                 track_type,
                 Q, 
                 R, 
                 takeoff_height=0.5) -> None:
        """
        States: [gamma, beta, alpha, x, y, z, xdot, ydot, zdot]
        Inputs: [a, wx, wy, wz]

        vehicle: Name of the vehicle (danaus12)
        cont_type: Type of controller (lqr or random)
        mode: Mode of operation (sim or real)
        hz: Frequency of the control loop
        track_type: Type of tracking to be used such as xyz goal, orientation goal, etc.
        Q: State cost matrix
        R: Input cost matrix
        takeoff_height: Height to takeoff to (in meters)
        """
        self.vehicle = vehicle
        self.cont_type = cont_type              # "lqr" or "random"

        self.mode = mode                        # "sim" or "real"
        self.hz = hz                            # Control Loop Frequency
        self.track_type = track_type            # Goal of controller: "xyz", "orient"
        self.takeoff_height = takeoff_height    # Takeoff Height
        
        self.Q = Q                              # State cost matrix
        self.R = R                              # Input cost matrix
        self.dt = 1/self.hz                     # Time step

        self.nx = 9
        self.nu = 4

        self.min_height = 2.0

        ### Subscribers ###
        self.quad_cb = VehicleStateCB(mode=self.mode)
        ### Services ###
        self.quad_modes = FcuModes()
        ### Publishers ###
        self.quad_pose_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.quad_att_setpoint_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)

        ### Time-checking
        self.time_pub = rospy.Publisher("time_check", String, queue_size=10)

        ### Hover thrust for the quadrotor
        self.hover_thrust = None

        ### Tracking Controller ###
        if self.cont_type == "lqr":
            self.cont = QuadLQRController(self.Q, self.R, np.array([0, 0, self.takeoff_height]), self.dt)
            self.cont_K_inf = self.cont.infinite_horizon_LQR()
            self.xgoal = self.cont.xgoal
            self.ugoal = self.cont.ugoal
            self.controller = self._quad_lqr_control

        elif self.cont_type == "random":
            self.controller = self._quad_rand_control
            raise NotImplementedError("Random controller not implemented yet.")
        
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
        rospy.init_node('dc_tracking_node', anonymous=True)

    def _time_calls(self):
        current_sim_time = rospy.get_time()
        current_real_time = time.time()
        time_msg = String()
        time_msg.data = f"Sim Time: {current_sim_time}, Real Time: {current_real_time}"
        self.time_pub.publish(time_msg)
        rospy.loginfo(time_msg.data)

    def _send_attitude_setpoint(self, u):
        self.att_setpoint.header.stamp = rospy.Time.now()
        self.att_setpoint.body_rate.x = u[1]
        self.att_setpoint.body_rate.y = u[2]
        self.att_setpoint.body_rate.z = u[3]
        self.att_setpoint.thrust = (u[0]/(9.81)) * self.hover_thrust
        self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)
        self.quad_att_setpoint_pub.publish(self.att_setpoint) 

    def _quad_lqr_control(self, x_quad):
        u = self.ugoal - self.cont_K_inf @ (x_quad - self.xgoal)
        
        # u[1] += np.random.uniform(-1, 1)
        # u[2] += np.random.uniform(-1, 1)
        # u[3] += np.random.uniform(-1, 1)
        
        # u[1] += np.random.normal(loc=0, scale=0.1, size=1).item()
        # u[2] += np.random.normal(loc=0, scale=0.1, size=1).item()
        # u[3] += np.random.normal(loc=0, scale=0.1, size=1).item()
        
        return u

    def _quad_rand_control(self, x_quad):
        pass

    def _get_states(self):
        quad_xyz = self.quad_cb.get_xyz_pose()
        quad_xyz_vel = self.quad_cb.get_xyz_velocity()
        quad_xyz_ang = self.quad_cb.get_xyz_angles()
        omega, omega_timestamp = self.quad_cb.get_xyz_angular_velocity_body_timestamped()
        accel = np.array([0, 0, 0])

        # γ, β, α, x, y, z, x_dot, y_dot, z_dot, θ, ϕ, θ_dot, ϕ_dot
        x_cont = np.concatenate((quad_xyz_ang.T, quad_xyz.T, quad_xyz_vel.T)) # Torque LQR expects (theta, phi).
        x_cont = x_cont.reshape((self.nx, 1))

        return x_cont, omega.reshape((3, 1)), accel.reshape((3, 1)), omega_timestamp

    def _takeoff_sequence(self):
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
            self.rate.sleep()
        
        self.hover_thrust = self.quad_cb.thrust

        if self.hover_thrust < 0.3:
            rospy.loginfo("Hover thrust too low. Exiting the node.")
            self.hover_thrust = 0.45

        rospy.loginfo("Recorded hover thrust: {}".format(self.hover_thrust))
        rospy.loginfo("Takeoff pose achieved!")

    def _change_goal(self):
        if self.track_type == "xyz":
            random_x = np.random.uniform(-5, 5, 1)
            random_y = np.random.uniform(-5, 5, 1)
            random_z = np.random.uniform(self.min_height+1, 10, 1)
            self.xgoal = np.array([0, 0, 0, random_x.item(), random_y.item(), random_z.item(), 0, 0, 0]).reshape((self.nx, 1))
        elif self.track_type == "orient":
            random_gamma = np.random.uniform(-np.pi/3, np.pi/3, 1)
            random_beta = np.random.uniform(-np.pi/3, np.pi/3, 1)
            random_alpha = np.random.uniform(-np.pi/3, np.pi/3, 1)
            random_z = np.random.uniform(self.min_height+3, 10, 1)
            self.xgoal = np.array([random_gamma.item(), random_beta.item(), random_alpha.item(), 0, 0, random_z.item(), 0, 0, 0]).reshape((self.nx, 1))

    def run(self, duration):
        # Log Array
        num_itr = int(duration * self.hz)
        state_log = np.zeros((self.nx, num_itr))
        input_log = np.zeros((self.nu, num_itr))
        error_log = np.zeros((self.nx, num_itr))
        omega_log = np.zeros((3, num_itr))  # Omega: rotational velocity
        accel_log = np.zeros((3, num_itr))  # Omega dot: rotational acceleration
        omega_timestamp_log = np.zeros((1, num_itr))

        ##### Takeoff Sequence #####
        self._takeoff_sequence()
        
        rospy.loginfo("Starting the data collection!")
        start_time = rospy.Time.now()

        # for itr in range(int(1e10)):  # If you want to test with console.
        for itr in range(num_itr):
            # self._time_calls()

            if rospy.is_shutdown():
                rospy.loginfo_throttle(3, "Node shutdown detected. Exiting the control loop.")
                break
            
            if (itr % 150 == 0):
                self._change_goal()

            ### Get the states ###
            x, omega, accel, omega_timestamp = self._get_states()
            print(f"{omega_timestamp=}")
            ### Get the control input ###
            u = self.controller(x)
            ### Send the control input ###
            self._send_attitude_setpoint(u)
            
            # Log the state and input
            state_log[:, itr] = x.flatten()
            input_log[:, itr] = u.flatten()
            error_log[:, itr] = (x - self.xgoal).flatten()
            omega_log[:, itr] = omega.flatten()
            accel_log[:, itr] = accel.flatten()
            omega_timestamp_log[:, itr] = omega_timestamp

            if x[5] < self.min_height:
                rospy.loginfo("Minimum height reached. Exiting the control loop.")
                break
            
            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log[:, :itr], input_log[:, :itr], error_log[:, :itr], omega_log[:, :itr], accel_log[:, :itr], omega_timestamp_log[:, :itr]


if __name__ == "__main__":

    # Get current date and time
    current_time = datetime.datetime.now()
    
    ##### Argparse #####
    parser = argparse.ArgumentParser(description="ETH Tracking Node for Constant Position")
    parser.add_argument("--mode", type=str, default="sim", help="Mode of operation (sim or real)")
    parser.add_argument("--hz", type=int, default=90, help="Frequency of the control loop")
    parser.add_argument("--track_type", type=str, default="xyz or orient", help="Type of tracking to be used")
    parser.add_argument("--takeoff_height", type=float, default=1.5, help="Height to takeoff to (in meters)")
    parser.add_argument("--cont_duration", type=int, default=20, help="Duration for which the controller should run (in seconds)")
    parser.add_argument("--vehicle", type=str)
    parser.add_argument("--cont_type", type=str, help="lqr or random")    # rs: r and s, tp: theta and phi

    args = parser.parse_args()
    mode = args.mode
    hz = args.hz
    track_type = args.track_type
    takeoff_height = args.takeoff_height
    cont_duration = args.cont_duration
    vehicle = args.vehicle
    cont_type = args.cont_type


    ##### LQR Gains #####
    Q = 1.0 * np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1])
    R = 1.0 * np.diag([1, 1, 1, 1])

    print("#####################################################")
    print("### Data Collection Node for Rate-Accel. Started ####")
    print("#####################################################")
    print("")

    eth_node = DataCollectionRateAccel(
        vehicle, cont_type, mode, 
        hz, track_type, Q, R, 
        takeoff_height=takeoff_height)
    
    state_log, input_log, error_log, omega_log, accel_log, omega_timestamp_log = eth_node.run(duration=cont_duration)

    print("#####################################################")
    print("### Data Collection Node for Rate-Accel. Started ####")
    print("#####################################################")
    print("")

    #################################
    ######### Save the logs #########
    #################################

    # curr_dir = os.getcwd()
    save_dir = "/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/data_collection"

    # Format the date and time into the desired filename format
    formatted_time = current_time.strftime("%m%d_%H%M%S-DC-Sim-Good")
    directory_path = os.path.join(save_dir, formatted_time)
    os.makedirs(directory_path, exist_ok=True)

    np.save(os.path.join(directory_path, "log.npy"),
            {
                "state": state_log,
                "input": input_log,
                "error": error_log,
                "omega": omega_log,
                "accel": accel_log,
                "omega_timestamp": omega_timestamp_log
            })
    np.save(os.path.join(directory_path, "params.npy"),
            {
                "vehicle": vehicle,
                "cont_type": cont_type,
                "mode": mode,
                "hz": hz,
                "track_type": track_type,
                "Q": Q,
                "R": R,
                "takeoff_height": takeoff_height,
                "cont_duration": cont_duration
            })
    
    print("#####################################################")
    print(f"########### LOG DATA SAVED IN {formatted_time} ###########")
    print("#####################################################")

