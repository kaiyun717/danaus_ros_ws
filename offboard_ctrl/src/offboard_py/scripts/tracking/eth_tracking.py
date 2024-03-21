"""
This script is used to track the position of the drone using the ETH controller, 
based on the paper "A Flying Inverted Pendulum" by Raffaello D'Andrea and Markus Hehn.
"""
#!/usr/bin/env python

import os
import math

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

class ETHTrackingNode:
    def __init__(self, 
                 mode,
                 hz, 
                 track_type,
                 mass,
                 L, 
                 Q, 
                 R, 
                 takeoff_height=0.5, 
                 lqr_itr=1000,
                 pend_upright_time=0.5,
                 pend_upright_tol=0.05) -> None:
        
        self.mode = mode                        # "sim" or "real"
        self.hz = hz                            # Control Loop Frequency
        self.track_type = track_type            # "constant" or "circular"
        self.takeoff_height = takeoff_height    # Takeoff Height
        
        self.mass = mass                        # Mass of the quadrotor + pendulum
        self.L = L                              # Length from pendulum base to CoM
        self.Q = Q                              # State cost matrix
        self.R = R                              # Input cost matrix
        self.dt = 1/self.hz                     # Time step
        self.lqr_itr = lqr_itr                  # Number of iterations for Infinite-Horizon LQR
        # self.cont_duration = cont_duration      # Duration for which the controller should run (in seconds)

        self.nx = 13
        self.nu = 4

        self.pend_upright_time = pend_upright_time  # Time to keep the pendulum upright
        self.pend_upright_tol = pend_upright_tol    # Tolerance for pendulum relative position [r,z] (norm in meters)

        ### Subscribers ###
        self.quad_cb = VehicleStateCB(mode=self.mode)
        self.pend_cb = PendulumCB(mode=self.mode)
        ### Services ###
        self.quad_modes = FcuModes()
        ### Publishers ###
        self.quad_pose_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.quad_att_setpoint_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)

        ### Tracking Controller ###
        if self.track_type == "constant":
            self.cont = ConstantPositionTracker(self.L, self.Q, self.R, self.dt)
            self.cont_K_inf = self.cont.infinite_horizon_LQR(self.lqr_itr)
            self.xgoal = self.cont.xgoal
            self.ugoal = self.cont.ugoal
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
            _t = 15
        else:
            _t = 25

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
            # rospy.loginfo("Takeoff sequence completed.")
            self.rate.sleep()

        if self.mode == "real":
            quad_pose = self.quad_cb.get_xyz_pose()
            while (not rospy.is_shutdown()) and (np.linalg.norm(target_pose - quad_pose) > 0.1):
                self.quad_pose_pub.publish(self.takeoff_pose)
                quad_pose = self.quad_cb.get_xyz_pose()
                rospy.loginfo("Quad pose: {}".format(quad_pose))
                self.rate.sleep()
        elif self.mode == "sim":
            pass
        
        rospy.loginfo("Takeoff pose achieved!")

    def _pend_upright_real(self, req_time=0.5, tol=0.05):
        consecutive_time = rospy.Duration(0.0)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            # Get the position of the pendulum
            pendulum_position = self.pend_cb.get_rz_pose()

            # Calculate the norm of the position
            position_norm = np.linalg.norm(pendulum_position)

            # Check if the norm is less than 0.05m
            if position_norm < tol:
                consecutive_time += rospy.Time.now() - start_time
                if consecutive_time >= rospy.Duration(req_time):
                    rospy.loginfo("Pendulum position has been less than 0.05m for 0.5 seconds straight.")
                    return True
            else:
                consecutive_time = rospy.Duration(0.0)
                start_time = rospy.Time.now()

            self.rate.sleep()
    
    def _pend_upright_sim(self, req_time=0.5, tol=0.05):
        get_link_properties_service = rospy.ServiceProxy('/gazebo/get_link_properties', GetLinkProperties)
        self.set_link_state_service = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        set_link_properties_service = rospy.ServiceProxy('/gazebo/set_link_properties', SetLinkProperties)

        consecutive_time = rospy.Duration(0.0)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            # Keep the quadrotor at this pose!
            self.quad_pose_pub.publish(self.takeoff_pose)

            link_state = LinkState()
            link_state.link_name = 'danaus12_pend::pendulum'
            link_state.reference_frame = 'base_link'
            _ = self.set_link_state_service(link_state)

            # Get the position of the pendulum
            pendulum_position = self.pend_cb.get_rz_pose()

            # Calculate the norm of the position
            position_norm = np.linalg.norm(pendulum_position)

            # Check if the norm is less than 0.05m
            if position_norm < tol:
                consecutive_time += rospy.Time.now() - start_time
                if consecutive_time >= rospy.Duration(req_time):
                    rospy.loginfo("Pendulum position has been less than 0.05m for 0.5 seconds straight.")
                    
                    # Turn gravity on!
                    curr_pend_properties = get_link_properties_service(link_name='danaus12_pend::pendulum')
                    new_pend_properties = SetLinkPropertiesRequest(
                                            link_name='danaus12_pend::pendulum',
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
                    rospy.sleep(1)
                    return True
            else:
                consecutive_time = 0
                start_time = rospy.Time.now()

            self.rate.sleep()

    def run(self, duration):
        # Log Array
        num_itr = int(duration * self.hz)
        state_log = np.zeros((self.nx, num_itr))
        input_log = np.zeros((self.nu, num_itr))

        ##### Takeoff Sequence #####
        self._takeoff_sequence()
        
        ##### Pendulum Mounting #####
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
        
        # itr = 0
        # while (not rospy.is_shutdown()) and (rospy.Time.now() - start_time) < rospy.Duration(duration):

        ##### Setting near origin & upright #####
        if self.mode == "sim":
            rospy.loginfo("Setting near origin & upright")
            for _ in range(150):
                link_state = LinkState()
                link_state.link_name = 'danaus12_pend::pendulum'
                link_state.reference_frame = 'base_link'
                _ = self.set_link_state_service(link_state)
                self.quad_pose_pub.publish(self.takeoff_pose)
        else:
            pass
        
        ##### Pendulum Position Control #####
        rospy.loginfo("Starting the constant position control!")
        start_time = rospy.Time.now()

        for itr in range(num_itr):
            if rospy.is_shutdown():
                rospy.loginfo("Node shutdown detected. Exiting the control loop.")
                break
            
            # Quad XYZ
            quad_xyz = self.quad_cb.get_xyz_pose()
            # Quad XYZ Velocity
            quad_xyz_vel = self.quad_cb.get_xyz_velocity()
            # Quad ZYX Angular Velocity
            quad_zyx_ang_vel = self.quad_cb.get_zyx_angular_velocity()
            # Pendulum RZ
            if self.mode == "sim":
                pend_rz = self.pend_cb.get_rz_pose()
            else:
                pend_rz = self.pend_cb.get_rz_pose(vehicle_pose=quad_xyz)
            # Pendulum RZ Velocity
            if self.mode == "sim":
                pend_rz_vel = self.pend_cb.get_rz_vel()
            else:
                pend_rz_vel = self.pend_cb.get_rz_vel(self.dt)

            # State Vector
            x = np.concatenate((quad_xyz.T, quad_xyz_vel.T, quad_zyx_ang_vel.T, pend_rz.T, pend_rz_vel.T))
            x = x.reshape((self.nx, 1))

            # Control Input
            u = self.ugoal - self.cont_K_inf @ (x - self.xgoal)

            # Publish the attitude setpoint
            self.att_setpoint.header.stamp = rospy.Time.now()
            self.att_setpoint.body_rate.x = u[0]
            self.att_setpoint.body_rate.y = u[1]
            self.att_setpoint.body_rate.z = u[2]
            # self.att_setpoint.thrust = (u[3] - 981) / 9.81
            self.att_setpoint.thrust = (u[3]/(9.81)) * 0.45

            self.quad_att_setpoint_pub.publish(self.att_setpoint)

            # Log the state and input
            state_log[:, itr] = x.flatten()
            input_log[:, itr] = u.flatten()

            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log, input_log


if __name__ == "__main__":
    
    ##### Argparse #####
    parser = argparse.ArgumentParser(description="ETH Tracking Node for Constant Position")
    parser.add_argument("--mode", type=str, default="sim", help="Mode of operation (sim or real)")
    parser.add_argument("--hz", type=int, default=50, help="Frequency of the control loop")
    parser.add_argument("--track_type", type=str, default="constant", help="Type of tracking to be used")
    parser.add_argument("--mass", type=float, default=0.676, help="Mass of the quadrotor + pendulum (in kg)")
    parser.add_argument("--takeoff_height", type=float, default=0.5, help="Height to takeoff to (in meters)")
    parser.add_argument("--pend_upright_time", type=float, default=0.5, help="Time to keep the pendulum upright")
    parser.add_argument("--pend_upright_tol", type=float, default=0.05, help="Tolerance for pendulum relative position [r,z] (norm in meters)")
    parser.add_argument("--lqr_itr", type=int, default=1000, help="Number of iterations for Infinite-Horizon LQR")
    parser.add_argument("--cont_duration", type=int, default=20, help="Duration for which the controller should run (in seconds)")

    args = parser.parse_args()
    mode = args.mode
    hz = args.hz
    track_type = args.track_type
    mass = args.mass
    takeoff_height = args.takeoff_height
    pend_upright_time = args.pend_upright_time
    pend_upright_tol = args.pend_upright_tol
    lqr_itr = args.lqr_itr
    cont_duration = args.cont_duration

    print("#####################################################")
    print("## ETH Tracking Node for Constant Position Started ##")
    print("#####################################################")
    
    L = 0.5
    Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    R = 1.0 * np.diag([1, 1, 1, 10])

    eth_node = ETHTrackingNode(mode, hz, track_type, mass, L, Q, R, 
                               takeoff_height=takeoff_height, 
                               lqr_itr=lqr_itr, 
                               pend_upright_time=pend_upright_time, 
                               pend_upright_tol=pend_upright_tol)
    
    state_log, input_log = eth_node.run(duration=cont_duration)

    print("#####################################################")
    print("### ETH Tracking Node for Constant Position Over  ###")
    print("#####################################################")

    # Save the logs
    print("State Log")
    print(state_log)
    print("Input Log")
    print(input_log)
    
    curr_dir = os.getcwd()
    save_dir = os.path.abspath(os.path.join(curr_dir, "../logs"))
    save_dir = curr_dir
    np.save(os.path.join(save_dir, "state_log_2.npy"), state_log)
    np.save(os.path.join(save_dir, "input_log_2.npy"), input_log)
