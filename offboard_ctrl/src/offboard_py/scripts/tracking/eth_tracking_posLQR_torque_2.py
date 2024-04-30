"""
This script is used to track the position of the drone using the ETH controller, 
based on the paper "A Flying Inverted Pendulum" by Raffaello D'Andrea and Markus Hehn.
"""
#!/usr/bin/env python

import os
import math
import datetime

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
                 takeoff_height=1.5, 
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
        self.cont_duration = cont_duration      # Duration for which the controller should run (in seconds)

        self.nx = 12
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

        self.hover_thrust = None

        # self.u_P = 0.5
        # self.u_I = 0.1
        # self.u_D = 0.001
        self.pitch_int = 0.0
        self.roll_int = 0.0
        self.yaw_int = 0.0
        self.gains_dict_px4_sim = {
            "MC_PITCHRATE_P": 0.138,
            "MC_PITCHRATE_I": 0.168,
            "MC_PITCHRATE_D": 0.0028,
            "MC_ROLLRATE_P": 0.094,
            "MC_ROLLRATE_I": 0.118,
            "MC_ROLLRATE_D": 0.0017,
            "MC_YAWRATE_P": 0.1,
            "MC_YAWRATE_I": 0.11,
            "MC_YAWRATE_D": 0.0,
        }
        

        ### Goal Position ###
        takeoff_pose = np.array([0, 0, self.takeoff_height])

        ### Takeoff Controller ###
        Q_takeoff = 1.0 * np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])      # Without pendulum
        R_takeoff = 1.0 * np.diag([1, 1, 1, 1])
        # self.takeoff_cont = ConstantPositionTracker(self.L, Q_takeoff, R_takeoff, takeoff_pose, self.dt)
        # self.takeoff_K_inf = self.takeoff_cont.infinite_horizon_LQR(self.lqr_itr)
        # self.takeoff_goal = self.takeoff_cont.xgoal
        # self.takeoff_input = self.takeoff_cont.ugoal
        self.takeoff_K_inf = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.990423659437537, 0.0, 0.0, 1.7209841208008194],
            [1.4814681367499676, 0.002995481626744845, 0.005889427337762064, 0.28201313626232954, 0.0005738532722789398, 0.005981686140109121, 0.0005315334025884319, -0.2644839556489373, 0.0, 0.000779188657396088, -0.3870845425896414, 0.0],
            [0.002995437953182237, 1.4528645588948705, 0.0034553424254151212, 0.0005738451434393372, 0.27654055987779025, 0.003509251033197913, 0.2594021552915394, -0.0005315255954541553, 0.0, 0.3796374382180433, -0.0007791772254469898, 0.0],
            [0.03160126871064939, 0.018545750101093363, 0.39799289325860154, 0.006079664965265176, 0.003567296347199587, 0.4032536566450523, 0.003278753893103592, -0.005585886845822208, 0.0, 0.004811104113305738, -0.008196880671368846, 0.0]
        ])

        self.takeoff_goal = np.array([0, 0, 0, 0, 0, 0, 1, 1, self.takeoff_height, 0, 0, 0]).reshape((self.nx, 1))
        self.takeoff_input = np.array([9.81, 0, 0, 0]).reshape((self.nu, 1))
        
        self.J_inv = np.array([
			[305.7518,  -0.6651,  -5.3547],
			[ -0.6651, 312.6261,  -3.1916],
			[ -5.3547,  -3.1916, 188.9651]])

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

        self.body_rate_accum = np.zeros(3)

    def _init_node(self):
        rospy.init_node('eth_tracking_node', anonymous=True)

    def _torque_to_body_rate(self, torque):
        body_rate = self.J_inv @ torque * self.dt
        # error = body_rate
        
        self.body_rate_accum = self.body_rate_accum * 0.9 + body_rate.flatten() * 0.1 
        # print(self.body_rate_accum)
        # error_x = error[0]
        # error_y = error[1]
        # error_z = error[2]

        # px = self.gains_dict_px4_sim["MC_ROLLRATE_P"] * error_x
        # py = self.gains_dict_px4_sim["MC_PITCHRATE_P"] * error_y
        # pz = self.gains_dict_px4_sim["MC_YAWRATE_P"] * error_z

        # self.roll_int += self.gains_dict_px4_sim["MC_ROLLRATE_I"] * error_x * self.dt
        # self.pitch_int += self.gains_dict_px4_sim["MC_PITCHRATE_I"] * error_y * self.dt
        # self.yaw_int += self.gains_dict_px4_sim["MC_YAWRATE_I"] * error_z * self.dt

        # ix = self.roll_int
        # iy = self.pitch_int
        # iz = self.yaw_int
        
        # dx = self.gains_dict_px4_sim["MC_ROLLRATE_D"] * error_x/self.dt
        # dy = self.gains_dict_px4_sim["MC_PITCHRATE_D"] * error_y/self.dt
        # dz = self.gains_dict_px4_sim["MC_YAWRATE_D"] * error_z/self.dt

        # return np.array([px + ix + dx, py + iy + dy, pz + iz + dz])
        return self.body_rate_accum


    def _quad_lqr_controller(self):
        quad_xyz = self.quad_cb.get_xyz_pose()
        quad_xyz_vel = self.quad_cb.get_xyz_velocity()
        quad_xyz_ang = self.quad_cb.get_xyz_angles()
        quad_xyz_ang_vel = self.quad_cb.get_xyz_angular_velocity()
        if self.mode == "sim":
            pend_rs = self.pend_cb.get_rs_pose(vehicle_pose=quad_xyz)
        else:
            pend_rs = np.array([0, 0])
        if self.mode == "sim":
            pend_rs_vel = self.pend_cb.get_rs_vel(vehicle_vel=quad_xyz_vel)
        else:
            pend_rs_vel = np.array([0, 0])
        
        x = np.concatenate((quad_xyz_ang.T, quad_xyz_ang_vel.T, quad_xyz.T, quad_xyz_vel.T))
        x = x.reshape((self.nx, 1))
        u = self.takeoff_input - self.takeoff_K_inf @ (x - self.takeoff_goal)  # 0 - K * dx = +ve

        self.att_setpoint.header.stamp = rospy.Time.now()
        # u_J_inv = self.J_inv @ u[1:].reshape((3,1)) * self.dt
        # self.u_int += self.u_I * u_J_inv
        # u_D = self.u_D * (u_J_inv - self.prev_u)/self.dt
        # self.prev_u = u_J_inv

        # u_body_rates = self.u_P * u_J_inv + self.u_int + u_D
        u_body_rates = self._torque_to_body_rate(u[1:].reshape((3,1)))
        
        self.att_setpoint.body_rate.x = u_body_rates[0]
        self.att_setpoint.body_rate.y = u_body_rates[1]
        self.att_setpoint.body_rate.z = u_body_rates[2]

        self.att_setpoint.thrust = (u[0]/(9.81)) * self.hover_thrust
        self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)
        self.quad_att_setpoint_pub.publish(self.att_setpoint) 

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
            self.hover_thrust = 0.39

        rospy.loginfo("Recorded hover thrust: {}".format(self.hover_thrust))

        # if self.mode == "real":
        #     quad_pose = self.quad_cb.get_xyz_pose()
        #     while (not rospy.is_shutdown()) and (np.linalg.norm(target_pose - quad_pose) > 0.1):
        #         self.quad_pose_pub.publish(self.takeoff_pose)
        #         quad_pose = self.quad_cb.get_xyz_pose()
        #         rospy.loginfo("Quad pose: {}".format(quad_pose))
        #         self.rate.sleep()
        # elif self.mode == "sim":
        #     pass
        
        rospy.loginfo("Takeoff pose achieved!")

    def _pend_upright_real(self, req_time=0.5, tol=0.05):
        consecutive_time = rospy.Duration(0.0)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            self._quad_lqr_controller()
            
            # Get the position of the pendulum
            quad_pose = self.quad_cb.get_xyz_pose()
            pendulum_position = self.pend_cb.get_rs_pose(vehicle_pose=quad_pose)

            # Calculate the norm of the position
            position_norm = np.linalg.norm(pendulum_position)

            # Check if the norm is less than 0.05m
            if position_norm < tol:
                rospy.loginfo("Pendulum upright for {} seconds.".format(consecutive_time.to_sec()))
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
            # # Keep the quadrotor at this pose!
            # self.quad_pose_pub.publish(self.takeoff_pose)  
            self._quad_lqr_controller(thrust_ratio=0.45)     # This is better than takeoff_pose
            
            link_state = LinkState()
            link_state.pose.position.x = -0.001995
            link_state.pose.position.y = 0.000135
            link_state.pose.position.z = 0.531659
            link_state.link_name = 'danaus12_pend::pendulum'
            link_state.reference_frame = 'base_link'
            _ = self.set_link_state_service(link_state)

            # Get the position of the pendulum
            quad_xyz = self.quad_cb.get_xyz_pose()
            pendulum_position = self.pend_cb.get_rs_pose(vehicle_pose=quad_xyz)

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
                    self._quad_lqr_controller(thrust_ratio=0.47) 
                    # rospy.sleep(0.5)
                    return True
            else:
                consecutive_time = rospy.Duration(0.0)
                start_time = rospy.Time.now()

            self.rate.sleep()

    def run(self, duration):
        # Log Array
        num_itr = int(duration * self.hz)
        state_log = np.zeros((self.nx, num_itr))
        input_log = np.zeros((self.nu, num_itr))
        error_log = np.zeros((self.nx, num_itr))

        ##### Takeoff Sequence #####
        self._takeoff_sequence()
        
        # ##### Pendulum Mounting #####
        # if self.mode == "real":
        #     # Sleep for 5 seconds so that pendulum can be mounted
        #     # rospy.loginfo("Sleeping for 5 seconds to mount the pendulum.")

        #     # Keep the pendulum upright
        #     rospy.loginfo("Keeping the pendulum upright.")
        #     self._pend_upright_real(req_time=self.pend_upright_time, tol=self.pend_upright_tol)
        # elif self.mode == "sim":
        #     rospy.loginfo("Swing the pendulum upright.")
        #     self._pend_upright_sim(req_time=self.pend_upright_time, tol=self.pend_upright_tol)
        
        # ##### Setting near origin & upright #####
        # if self.mode == "sim":
        #     rospy.loginfo("Setting near origin & upright")
        #     for _ in range(150):
        #         link_state = LinkState()
        #         link_state.pose.position.x = -0.001995
        #         link_state.pose.position.y = 0.000135
        #         link_state.pose.position.z = 0.531659
        #         link_state.link_name = 'danaus12_pend::pendulum'
        #         link_state.reference_frame = 'base_link'
        #         _ = self.set_link_state_service(link_state)
                
        #         # self.quad_pose_pub.publish(self.takeoff_pose)
        #         self._quad_lqr_controller(thrust_ratio=0.46)     # This is better than takeoff_pose
        # else:
        #     pass
        
        ##### Pendulum Position Control #####
        rospy.loginfo("Starting the constant position control!")
        start_time = rospy.Time.now()

        # for itr in range(int(1e10)):  # If you want to test with console.
        for itr in range(int(10e10)):
            if rospy.is_shutdown():
                rospy.loginfo_throttle(3, "Node shutdown detected. Exiting the control loop.")
                break
            
            # Quad XYZ
            # quad_xyz = self.quad_cb.get_xyz_pose()
            # # Quad XYZ Velocity
            # quad_xyz_vel = self.quad_cb.get_xyz_velocity()
            # # Quad ZYX Angles
            # quad_zyx_ang = self.quad_cb.get_zyx_angles()
            # # Pendulum RS
            # if self.mode == "sim":
            #     pend_rs = self.pend_cb.get_rs_pose(vehicle_pose=quad_xyz)
            # else:
            #     # pend_rs = self.pend_cb.get_rs_pose(vehicle_pose=quad_xyz)
            #     pend_rs = np.array([0, 0])
            # # Pendulum RS Velocity
            # x = np.concatenate((quad_xyz.T, quad_xyz_vel.T, quad_zyx_ang.T, pend_rs.T, pend_rs_vel.T))
            # x = x.reshape((self.nx, 1))

            # # Control Input
            # u = self.ugoal - self.cont_K_inf @ (x - self.xgoal)  # 0 - K * dx = +ve

            # # Publish the attitude setpoint
            # self.att_setpoint.header.stamp = rospy.Time.now()
            # self.att_setpoint.body_rate.x = u[0]    # np.clip(u[0], -20, 20)
            # self.att_setpoint.body_rate.y = u[1]    # np.clip(u[1], -20, 20)
            # self.att_setpoint.body_rate.z = u[2]    # np.clip(u[2], -20, 20)
            # self.att_setpoint.thrust = (u[3]/(9.81)) * self.hover_thrust
            # self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)
            # # rospy.loginfo_throttle(0.2, "Thrust: {}".format(self.att_setpoint.thrust))

            # rospy.loginfo_throttle(1, "u[0]: {}, u[1]: {}, u[2]: {}, u[3]: {}".format(u[0], u[1], u[2], u[3]))
            # rospy.loginfo_throttle(1, "z: {}, z_target: {}".format(quad_xyz[2], self.xgoal[2]) )
            # # rospy.loginfo_throttle(1, "wx: {}, wy: {}, wz: {}, Thrust: {}".format(self.att_setpoint.body_rate.x, self.att_setpoint.body_rate.y, self.att_setpoint.body_rate.z, self.att_setpoint.thrust))
            # self.quad_att_setpoint_pub.publish(self.att_setpoint) # Uncomment this line to publish the attitude setpoint
            # # self.quad_pose_pub.publish(s
            # if self.mode == "sim":
            #     pend_rs_vel = self.pend_cb.get_rs_vel(vehicle_vel=quad_xyz_vel)
            # else:
            #     # pend_rs_vel = self.pend_cb.get_rs_vel(vehicle_vel=quad_xyz_vel)
            #     pend_rs_vel = np.array([0, 0])

            # # State Vector
            # x = np.concatenate((quad_xyz.T, quad_xyz_vel.T, quad_zyx_ang.T, pend_rs.T, pend_rs_vel.T))
            # x = x.reshape((self.nx, 1))

            # # Control Input
            # u = self.ugoal - self.cont_K_inf @ (x - self.xgoal)  # 0 - K * dx = +ve

            # # Publish the attitude setpoint
            # self.att_setpoint.header.stamp = rospy.Time.now()
            # self.att_setpoint.body_rate.x = u[0]    # np.clip(u[0], -20, 20)
            # self.att_setpoint.body_rate.y = u[1]    # np.clip(u[1], -20, 20)
            # self.att_setpoint.body_rate.z = u[2]    # np.clip(u[2], -20, 20)
            # self.att_setpoint.thrust = (u[3]/(9.81)) * self.hover_thrust
            # self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)
            # # rospy.loginfo_throttle(0.2, "Thrust: {}".format(self.att_setpoint.thrust))

            # rospy.loginfo_throttle(1, "u[0]: {}, u[1]: {}, u[2]: {}, u[3]: {}".format(u[0], u[1], u[2], u[3]))
            # rospy.loginfo_throttle(1, "z: {}, z_target: {}".format(quad_xyz[2], self.xgoal[2]) )
            # # rospy.loginfo_throttle(1, "wx: {}, wy: {}, wz: {}, Thrust: {}".format(self.att_setpoint.body_rate.x, self.att_setpoint.body_rate.y, self.att_setpoint.body_rate.z, self.att_setpoint.thrust))
            # self.quad_att_setpoint_pub.publish(self.att_setpoint) # Uncomment this line to publish the attitude setpoint
            # # self.quad_pose_pub.publish(self.takeoff_pose)
            
            # # Log the state and input
            # state_log[:, itr] = x.flatten()
            # input_log[:, itr] = u.flatten()
            # error_log[:, itr] = (x - self.xgoal).flatten()
            
            self._quad_lqr_controller()

            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log, input_log, error_log


if __name__ == "__main__":
    
    ##### Argparse #####
    parser = argparse.ArgumentParser(description="ETH Tracking Node for Constant Position")
    parser.add_argument("--mode", type=str, default="real", help="Mode of operation (sim or real)")
    parser.add_argument("--hz", type=int, default=90, help="Frequency of the control loop")
    parser.add_argument("--track_type", type=str, default="constant", help="Type of tracking to be used")
    # parser.add_argument("--mass", type=float, default=0.73578, help="Mass of the quadrotor + pendulum (in kg)")
    parser.add_argument("--mass", type=float, default=0.700, help="Mass of the quadrotor + pendulum (in kg)")
    parser.add_argument("--takeoff_height", type=float, default=1.5, help="Height to takeoff to (in meters)")
    parser.add_argument("--pend_upright_time", type=float, default=0.5, help="Time to keep the pendulum upright")
    parser.add_argument("--pend_upright_tol", type=float, default=0.05, help="Tolerance for pendulum relative position [r,z] (norm in meters)")
    parser.add_argument("--lqr_itr", type=int, default=100000, help="Number of iterations for Infinite-Horizon LQR")
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
    print("")
    
    L = 0.69            # x  y  z  x_dot y_dot z_dot yaw pitch roll r s r_dot s_dot
    Q = 1.0 * np.diag([0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 0.4, 0.4])      # With pendulum
    # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
    R = 1.0 * np.diag([50, 50, 50, 1])

    eth_node = ETHTrackingNode(mode, hz, track_type, mass, L, Q, R, 
                               takeoff_height=takeoff_height, 
                               lqr_itr=lqr_itr, 
                               pend_upright_time=pend_upright_time, 
                               pend_upright_tol=pend_upright_tol)
    
    state_log, input_log, error_log = eth_node.run(duration=cont_duration)

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
    formatted_time = current_time.strftime("%m%d_%H%M%S-Real")
    directory_path = os.path.join(save_dir, formatted_time)
    os.makedirs(directory_path, exist_ok=True)

    np.save(os.path.join(directory_path, "state.npy"), state_log)
    np.save(os.path.join(directory_path, "input.npy"), input_log)
    np.save(os.path.join(directory_path, "error.npy"), error_log)
    np.save(os.path.join(directory_path, "gains.npy"), {"Q": Q, "R": R})

    print("#####################################################")
    print(f"########### LOG DATA SAVED IN {formatted_time} ###########")
    print("#####################################################")
