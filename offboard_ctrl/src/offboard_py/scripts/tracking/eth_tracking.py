"""
This script is used to track the position of the drone using the ETH controller, 
based on the paper "A Flying Inverted Pendulum" by Raffaello D'Andrea and Markus Hehn.
"""
import os
import math

import rospy
import numpy as np
import scipy

import argparse

from geometry_msgs.msg import PoseStamped, Quaternion, Vector3, TwistStamped
from std_msgs.msg import Header
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from src.controllers.eth_constant_controller import ConstantPositionTracker
from src.callbacks.fcu_state_callbacks import VehicleStateCB
from src.callbacks.pend_state_callbacks import PendulumCB
from src.callbacks.fcu_modes import FcuModes


class ETHTrackingNode:
    def __init__(self, 
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

        self.pend_upright_time = pend_upright_time  # Time to keep the pendulum upright
        self.pend_upright_tol = pend_upright_tol    # Tolerance for pendulum relative position [r,z] (norm in meters)

        ### Subscribers ###
        self.quad_cb = VehicleStateCB()
        self.pend_cb = PendulumCB()
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
        
        ### Attitude Setpoint ###
        self.att_setpoint = AttitudeTarget()
        self.att_setpoint.header = Header()
        self.att_setpoint.header.frame_id = "base_footprint"
        self.att_setpoint.type_mask = 128  # Ignore attitude/orientation

        ### Initialize the node ###
        self.rate = rospy.Rate(self.hz)
        self._init_node()


    def _init_node(self):
        rospy.init_node('eth_tracking_node', anonymous=True)

    def _takeoff_sequence(self):
        # Takeoff Pose
        takeoff_pose = PoseStamped()
        takeoff_pose.pose.position.x = 0
        takeoff_pose.pose.position.y = 0
        takeoff_pose.pose.position.z = self.takeoff_height
        target_pose = np.array([takeoff_pose.pose.position.x, takeoff_pose.pose.position.y, takeoff_pose.pose.position.z])

        # Send initial commands
        for _ in range(100):
            if rospy.is_shutdown():
                break
            
            self.quad_pose_pub.publish(takeoff_pose)
            self.rate.sleep()
        
        # Arm and Offboard Mode
        last_req = rospy.Time.now()
        start_time = rospy.Time.now()

        while (not rospy.is_shutdown()) and (rospy.Time.now() - start_time) < rospy.Duration(25):
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

            self.quad_pose_pub.publish(takeoff_pose)
            rospy.loginfo("Takeoff sequence completed.")
            self.rate.sleep()

        quad_pose = self.quad_cb.get_xyz_pose()
        while (not rospy.is_shutdown()) and (np.linalg.norm(target_pose - quad_pose) > 0.1):
            quad_pose = self.quad_cb.get_xyz_pose()
            self.rate.sleep()
        
        rospy.loginfo("Takeoff pose achieved!")

    def _pend_upright(self, req_time=0.5, tol=0.05):
        consecutive_time = 0.0

        while not rospy.is_shutdown():
            # Get the position of the pendulum
            pendulum_position = self.pend_cb.get_rz_pose()

            # Calculate the norm of the position
            position_norm = np.linalg.norm(pendulum_position)

            # Check if the norm is less than 0.05m
            if position_norm < tol:
                consecutive_time += rospy.Time.now() - start_time
                if consecutive_time >= req_time:
                    rospy.loginfo("Pendulum position has been less than 0.05m for 0.5 seconds straight.")
                    return True
            else:
                consecutive_time = 0
                start_time = rospy.Time.now()

            self.rate.sleep()

    def run(self, duration):
        # Takeoff Sequence
        self._takeoff_sequence()
        
        # Sleep for 5 seconds so that pendulum can be mounted
        rospy.loginfo("Sleeping for 5 seconds to mount the pendulum.")
        rospy.sleep(5)

        # Keep the pendulum upright
        rospy.loginfo("Keeping the pendulum upright.")
        self._pend_upright(req_time=self.pend_upright_time, tol=self.pend_upright_tol)

        # Log Array
        num_itr = int(duration * self.hz)
        state_log = np.zeros((self.nx, num_itr))
        input_log = np.zeros((self.nu, num_itr))

        # Start the constant position control!
        rospy.loginfo("Starting the constant position control!")
        start_time = rospy.Time.now()

        # itr = 0
        # while (not rospy.is_shutdown()) and (rospy.Time.now() - start_time) < rospy.Duration(duration):

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
            pend_rz = self.pend_cb.get_rz_pose()
            # Pendulum RZ Velocity
            pend_rz_vel = self.pend_cb.get_rz_vel(self.dt)

            # State Vector
            x = np.vstack((quad_xyz, quad_xyz_vel, quad_zyx_ang_vel, pend_rz, pend_rz_vel))
            x = x.reshape((self.nx, 1))

            # Control Input
            u = self.ugoal - self.cont_K_inf @ (x - self.xgoal)

            # Publish the attitude setpoint
            self.att_setpoint.header.stamp = rospy.Time.now()
            self.att_setpoint.body_rate.x = u[2]
            self.att_setpoint.body_rate.y = u[1]
            self.att_setpoint.body_rate.z = u[0]
            self.att_setpoint.thrust = u[3] * self.mass

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
    parser.add_argument("--hz", type=int, default=50, help="Frequency of the control loop")
    parser.add_argument("--track_type", type=str, default="constant", help="Type of tracking to be used")
    parser.add_argument("--takeoff_height", type=float, default=0.5, help="Height to takeoff to (in meters)")
    parser.add_argument("--pend_upright_time", type=float, default=0.5, help="Time to keep the pendulum upright")
    parser.add_argument("--pend_upright_tol", type=float, default=0.05, help="Tolerance for pendulum relative position [r,z] (norm in meters)")
    parser.add_argument("--lqr_itr", type=int, default=1000, help="Number of iterations for Infinite-Horizon LQR")
    parser.add_argument("--cont_duration", type=int, default=20, help="Duration for which the controller should run (in seconds)")

    args = parser.parse_args()
    hz = args.hz
    track_type = args.track_type
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
    R = 1.0 * np.diag([1, 1, 1, 1])

    eth_node = ETHTrackingNode(hz, track_type, L, Q, R, 
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
    # save_dir = os.path.abspath(os.path.join(curr_dir, "../../logs"))
    save_dir = curr_dir
    np.save(os.path.join(save_dir, "state_log.npy"), state_log)
    np.save(os.path.join(save_dir, "input_log.npy"), input_log)
