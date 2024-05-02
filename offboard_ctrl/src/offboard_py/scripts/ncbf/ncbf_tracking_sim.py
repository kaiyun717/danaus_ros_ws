"""
This script is used to track the position of the drone using the ETH controller, 
based on the paper "A Flying Inverted Pendulum" by Raffaello D'Andrea and Markus Hehn.
"""
#!/usr/bin/env python

import os
import math
import datetime

import IPython
import rospy
import torch
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

from src.controllers.torque_constant_position_tracker import TorqueConstantPositionTracker
from src.callbacks.fcu_state_callbacks import VehicleStateCB
from src.callbacks.pend_state_callbacks import PendulumCB
from src.callbacks.fcu_modes import FcuModes

from src.neural_cbf.utils import load_phi_and_params
from src.neural_cbf.ncbf_numpy_wrapper import NCBFNumpy
from src.neural_cbf.ncbf_controller import NCBFController
from src.env.deploy_flying_inv_pend import FlyingInvertedPendulumEnv
from src.controllers.torque_lqr import TorqueLQR


class NCBFTrackingNode:
    def __init__(self, 
                 exp_name,
                 ckpt_num,
                 mode,
                 hz, 
                 track_type,
                 mass,
                 L, 
                 Q, 
                 R,
                 eps_bdry=1.0,
                 eps_outside=5.0,
                 dynamics_noise_spread=0.00,
                 lqr_cont_type="with_pend",
                 takeoff_height=1.5, 
                 lqr_itr=100000,
                 pend_upright_time=0.5,
                 pend_upright_tol=0.05) -> None:
        
        ######################
        ##### Neural CBF #####
        ######################

        # if torch.cuda.is_available():
        #     os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        #     dev = "cuda:%i" % (0)
        #     print("Using GPU device: %s" % dev)
        # else:
        dev = "cpu"
        device = torch.device(dev)

        self.exp_name = exp_name
        self.ckpt_num = ckpt_num
        torch_ncbf_fn, param_dict = load_phi_and_params(exp_name, ckpt_num, device)

        self.ncbf_fn = NCBFNumpy(torch_ncbf_fn, device)
        print(f"{self.ncbf_fn.device=}")
        self.env = FlyingInvertedPendulumEnv(dt=1/hz, model_param_dict=param_dict, 
                                             dynamics_noise_spread=dynamics_noise_spread)
        self.env.dt = 1/hz
        self.ncbf_cont = NCBFController(self.env, self.ncbf_fn, param_dict, eps_bdry=eps_bdry, eps_outside=eps_outside)

        ######################
        #####   PARAMS   #####
        ######################
        self.mode = mode                        # "sim" or "real"
        self.hz = hz                            # Control Loop Frequency
        self.track_type = track_type            # "constant" or "circular"
        self.takeoff_height = takeoff_height    # Takeoff Height
        
        self.mass = mass                        # Mass of the quadrotor + pendulum
        self.L = L                              # Length from pendulum base to CoM
        self.dt = 1/self.hz                     # Time step
        self.lqr_itr = lqr_itr                  # Number of iterations for Infinite-Horizon LQR
        self.cont_duration = cont_duration      # Duration for which the controller should run (in seconds)
        self.lqr_cont_type = lqr_cont_type      # "with_pend" or "without_pend"

        self.nx = 16
        self.nu = 4

        self.pend_upright_time = pend_upright_time  # Time to keep the pendulum upright
        self.pend_upright_tol = pend_upright_tol    # Tolerance for pendulum relative position [r,z] (norm in meters)

        ######################
        #####   MAVROS   #####
        ######################

        ### Subscribers ###
        self.quad_cb = VehicleStateCB(mode=self.mode)
        self.pend_cb = PendulumCB(mode=self.mode)
        ### Services ###
        self.quad_modes = FcuModes()
        ### Publishers ###
        self.quad_pose_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)
        self.quad_att_setpoint_pub = rospy.Publisher("mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=1)

        self.hover_thrust = None

        ######################
        #####     LQR    #####
        ######################

        ### Goal Position ###
        takeoff_pose = np.array([0, 0, self.takeoff_height])

        ### Nominal Controller ###
        if self.track_type == "constant":
            if self.lqr_cont_type == "with_pend":
                K_inf = np.array([
                            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.7024504701211454,0.0,0.0,1.1852851512706413,],
                            [1.0498346060574577,0.0,0.0,0.08111887959217115,0.0,0.0,3.1249612183719218,0.0,0.8390135195024693,0.0,0.0,0.2439310793798623,0.0,0.0,0.3542763507641887,0.0,],
                            [0.0,1.0368611054298649,0.0,0.0,0.07970485761038303,0.0,0.0,-3.1048038968779004,0.0,-0.8337170169504385,-0.24373748893808928,0.0,0.0,-0.3536063529300743,0.0,0.0,],
                            [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,],])
                # self.torque_LQR = TorqueConstantPositionTracker(K_inf, takeoff_pose, self.dt)
                self.torque_LQR = TorqueLQR(L, Q, R, takeoff_pose, self.dt, lqr_cont_type, num_itr=lqr_itr, K_inf=K_inf)
            elif self.lqr_cont_type == "without_pend":
                # K_inf = np.array([
                #             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.990423659437537, 0.0, 0.0, 1.7209841208008194],
                #             [1.4814681367499676, 0.002995481626744845, 0.005889427337762064, 0.28201313626232954, 0.0005738532722789398, 0.005981686140109121, 0.0005315334025884319, -0.2644839556489373, 0.0, 0.000779188657396088, -0.3870845425896414, 0.0],
                #             [0.002995437953182237, 1.4528645588948705, 0.0034553424254151212, 0.0005738451434393372, 0.27654055987779025, 0.003509251033197913, 0.2594021552915394, -0.0005315255954541553, 0.0, 0.3796374382180433, -0.0007791772254469898, 0.0],
                #             [0.03160126871064939, 0.018545750101093363, 0.39799289325860154, 0.006079664965265176, 0.003567296347199587, 0.4032536566450523, 0.003278753893103592, -0.005585886845822208, 0.0, 0.004811104113305738, -0.008196880671368846, 0.0]
                #         ])
                # self.torque_LQR = TorqueConstantPositionTracker(K_inf, takeoff_pose, self.dt)
                self.torque_LQR = TorqueLQR(L, Q, R, takeoff_pose, self.dt, lqr_cont_type, num_itr=lqr_itr)
        else:
            raise NotImplementedError("Circular tracking not implemented.")

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

    def _quad_lqr_controller(self, x):
        u_torque = self.torque_LQR.torque_inputs(x)
        return u_torque

    def _send_attitude_setpoint(self, u):
        """ u[0]: Thrust, u[1]: Roll, u[2]: Pitch, u[3]: Yaw """
        self.att_setpoint.header.stamp = rospy.Time.now()
        self.att_setpoint.body_rate.x = u[1]
        self.att_setpoint.body_rate.y = u[2]
        self.att_setpoint.body_rate.z = u[3]
        self.att_setpoint.thrust = (u[0]/(9.81 * self.mass)) * self.hover_thrust
        self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)
        self.quad_att_setpoint_pub.publish(self.att_setpoint) 

    def _get_states(self):
        quad_xyz = self.quad_cb.get_xyz_pose()
        quad_xyz_vel = self.quad_cb.get_xyz_velocity()
        quad_xyz_ang = self.quad_cb.get_xyz_angles()
        quad_xyz_ang_vel = self.quad_cb.get_xyz_angular_velocity()   # TODO: Need to verify
        # pend_ang = self.pend_cb.get_rs_ang(vehicle_pose=quad_xyz)
        # pend_ang_vel = self.pend_cb.get_rs_ang_vel(vehicle_pose=None, vehicle_vel=quad_xyz_vel)
        pend_ang = np.array([0, 0])
        pend_ang_vel = np.array([0, 0])
        
        x_safe = np.concatenate((quad_xyz_ang.T, quad_xyz_ang_vel.T, pend_ang.T, pend_ang_vel.T, quad_xyz.T, quad_xyz_vel.T))
        x_safe = x_safe.reshape((self.nx, 1))

        x_nom = np.concatenate((quad_xyz_ang.T, quad_xyz_ang_vel.T, pend_ang.T, pend_ang_vel.T,quad_xyz.T, quad_xyz_vel.T))
        x_nom = x_nom.reshape((self.torque_LQR.nx, 1))
        return x_safe, x_nom

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

            self.rate.sleep()
        
        self.hover_thrust = self.quad_cb.thrust * 1.0

        if self.hover_thrust < 0.3:
            rospy.loginfo("Hover thrust too low. Exiting the node.")
            self.hover_thrust = 0.39

        rospy.loginfo("Recorded hover thrust: {}".format(self.hover_thrust))
        rospy.loginfo("Takeoff pose achieved!")

    def _pend_upright_real(self, req_time=1, tol=0.01):
        consecutive_time = rospy.Duration(0.0)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            x_safe, x_nom = self._get_states()
            u_nom = self._quad_lqr_controller(x_nom)
            self._send_attitude_setpoint(u_nom)
            
            # Get the position of the pendulum
            quad_pose = self.quad_cb.get_xyz_pose()
            pendulum_position = self.pend_cb.get_rs_pose(vehicle_pose=quad_pose)

            # Calculate the norm of the position
            position_norm = np.linalg.norm(pendulum_position)

            # Check if the norm is less than  tol
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
            x_safe, x_nom = self._get_states()
            u_nom = self._quad_lqr_controller(x_nom)
            self._send_attitude_setpoint(u_nom)
            
            link_state = LinkState()
            link_state.pose.position.x = -0.001995
            link_state.pose.position.y = 0.000135
            link_state.pose.position.z = 0.721659
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
                    x_safe, x_nom = self._get_states()
                    u_nom = self._quad_lqr_controller(x_nom)
                    self._send_attitude_setpoint(u_nom) 
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
        nom_input_log = np.zeros((self.nu, num_itr))
        safe_input_log = np.zeros((self.nu, num_itr))
        error_log = np.zeros((self.torque_LQR.nx, num_itr))
        status_log = np.zeros((1, num_itr))
        phi_val_log = np.zeros((1, num_itr))

        ##### Takeoff Sequence #####0411_193906-Real
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
        #      ##### Setting near origin & upright #####
        #     rospy.loginfo("Setting near origin & upright")
        #     for _ in range(350):
        #         link_state = LinkState()
        #         link_state.pose.position.x = -0.001995
        #         link_state.pose.position.y = 0.000135
        #         link_state.pose.position.z = 0.721659
        #         link_state.link_name = 'danaus12_pend::pendulum'
        #         link_state.reference_frame = 'base_link'
        #         _ = self.set_link_state_service(link_state)
                
        #         # self.quad_pose_pub.publish(self.takeoff_pose)
        #         x_safe, x_nom = self._get_states()
        #         u_nom = self._quad_lqr_controller(x_nom)     # This is better than takeoff_pose
        #         self._send_attitude_setpoint(u_nom)
        
        ##### Pendulum Position Control #####
        rospy.loginfo("Starting the constant position control!")
        start_time = rospy.Time.now()

        # for itr in range(int(1e10)):  # If you want to test with console.
        for itr in range(num_itr):
            if rospy.is_shutdown():
                rospy.loginfo_throttle(3, "Node shutdown detected. Exiting the control loop.")
                break
            
            x_safe, x_nom = self._get_states()
            # pend_rs = self.pend_cb.get_rs_pose(vehicle_pose=x_nom[6:8].flatten())
            
            # if pend_rs is None:
            #     # self.quad_pose_pub.publish(self.takeoff_pose)
            #     self.att_setpoint.header.stamp = rospy.Time.now()
            #     self.att_setpoint.body_rate.x = 0    # np.clip(u[0], -20, 20)
            #     self.att_setpoint.body_rate.y = 0    # np.clip(u[1], -20, 20)
            #     self.att_setpoint.body_rate.z = 0    # np.clip(u[2], -20, 20)
            #     self.att_setpoint.thrust = 0
            #     self.quad_att_setpoint_pub.publish(self.att_setpoint)
            #     rospy.loginfo_throttle(3,"Pendulum can't be seen. Continuing the control loop.")
            #     self.rate.sleep()
            #     continue

            # rospy.loginfo_throttle(0.2, f"RS Norm: {np.linalg.norm(pend_rs)}")
                
            # if np.linalg.norm(pend_rs) > 0.65:
            #     # self._quad_lqr_controller()
            #     # self.quad_pose_pub.publish(self.takeoff_pose)
            #     rospy.loginfo_throttle(3,"Pendulum too far away. Exiting the control loop.")
            #     self.rate.sleep()
            #     continue
            
            u_torque = self._quad_lqr_controller(x_nom) # 9.75
            
            u_safe, stat, phi_val = self.ncbf_cont.compute_control(x_safe, np.copy(u_torque))
            u_safe_body = self.torque_LQR.torque_to_body_rate(u_safe)
            rospy.loginfo(f"Itr.{itr}/{num_itr}\nU Safe: {u_safe_body.flatten()}\nU Nom: {u_torque.flatten()}\nPhi: {phi_val}, Status\n{stat}")

            self._send_attitude_setpoint(u_safe_body)
            
            # Log the state and input
            state_log[:, itr] = x_safe.flatten()
            nom_input_log[:, itr] = u_torque.flatten()
            safe_input_log[:, itr] = u_safe.flatten()
            error_log[:, itr] = (x_nom - self.torque_LQR.xgoal).flatten()
            status_log[:, itr] = stat
            phi_val_log[:, itr] = phi_val

            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log, nom_input_log, safe_input_log, error_log, status_log, phi_val_log


if __name__ == "__main__":
    
    ##### Argparse #####
    parser = argparse.ArgumentParser(description="NCBF Tracking Node")
    parser.add_argument("--mode", type=str, default="real", help="Mode of operation (sim or real)")
    parser.add_argument("--hz", type=int, default=90, help="Frequency of the control loop")
    parser.add_argument("--track_type", type=str, default="constant", help="Type of tracking to be used")
    parser.add_argument("--mass", type=float, default=0.746, help="Mass of the quadrotor + pendulum (in kg)")
    parser.add_argument("--takeoff_height", type=float, default=1.5, help="Height to takeoff to (in meters)")
    parser.add_argument("--pend_upright_time", type=float, default=0.5, help="Time to keep the pendulum upright")
    parser.add_argument("--pend_upright_tol", type=float, default=0.05, help="Tolerance for pendulum relative position [r,z] (norm in meters)")
    parser.add_argument("--lqr_itr", type=int, default=100000, help="Number of iterations for Infinite-Horizon LQR")
    parser.add_argument("--cont_duration", type=int, default=10, help="Duration for which the controller should run (in seconds)")
    parser.add_argument("--lqr_cont_type", type=str, default="with_pend", help="with or without pendulum")
    parser.add_argument("--exp_name", type=str, help="Experiment name")
    parser.add_argument("--ckpt_num", type=int, help="Checkpoint number")
    parser.add_argument("--eps_bdry", type=float, default=1.0, help="Boundary epsilon")
    parser.add_argument("--eps_outside", type=float, default=5.0, help="Outside epsilon")
    parser.add_argument("--dynamics_noise_spread", type=float, default=0.00, help="Dynamics noise spread")

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
    lqr_cont_type = args.lqr_cont_type
    exp_name = args.exp_name
    ckpt_num = args.ckpt_num
    eps_bdry = args.eps_bdry
    eps_outside = args.eps_outside
    dynamics_noise_spread = args.dynamics_noise_spread

    print("######################################################")
    print("## NCBF Tracking Node for Constant Position Started ##")
    print("######################################################")
    print("")
    

    L = 0.69            # x  y  z  x_dot y_dot z_dot yaw pitch roll r s r_dot s_dot

    if lqr_cont_type == "with_pend":
        # Q = 1.0 * np.diag([50, 50, 4, 0, 0, 0.0, 0.0, 0, 0, 10, 10, 10,  10])      # With pendulum
        # # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        # R = 1.0 * np.diag([100, 100, 0.01, 0.9])

        Q = 1.0 * np.diag([3, 3, 2, 0.005, 0.005, 0.0, 0.0, 0.0, 0.0, 10, 10, 0.0001, 0.0001])      # With pendulum
        Q = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10, 10, 0.0001, 0.0001, 3, 3, 2, 0.005, 0.005, 0.0])      # With pendulum
        R = 1.0 * np.diag([1, 10, 10, 1])

        # Q = 1.0 * np.diag([4, 4, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16, 16, 0.4, 0.4])      # With pendulum
        # # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        # R = 1.0 * np.diag([6.5, 6.5, 1, 1])

        # Q = 1.0 * np.diag([2, 2, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, 2, 0, 0])      # With pendulum
        # # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        # R = 1.0 * np.diag([70, 70, 1, 1])
        
    ncbf_node = NCBFTrackingNode(
        exp_name=exp_name, ckpt_num=ckpt_num,
        mode=mode, hz=hz, track_type=track_type, 
        mass=mass, L=L, Q=Q, R=R,
        eps_bdry=eps_bdry, eps_outside=eps_outside, dynamics_noise_spread=dynamics_noise_spread,
        lqr_cont_type=lqr_cont_type, 
        takeoff_height=takeoff_height, 
        lqr_itr=lqr_itr, 
        pend_upright_time=pend_upright_time, 
        pend_upright_tol=pend_upright_tol)
    
    state_log, nom_input_log, safe_input_log, error_log, status_log, phi_val_log = ncbf_node.run(duration=cont_duration)
    print("####################################################")
    print("## NCBF Tracking Node for Constant Position Over  ##")
    print("####################################################")
    print("")

    #################################
    ######### Save the logs #########
    #################################

    # curr_dir = os.getcwd()
    save_dir = "/home/oem/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/ncbf"

    # Get current date and time
    current_time = datetime.datetime.now()
    # Format the date and time into the desired filename format
    formatted_time = current_time.strftime("%m%d_%H%M%S-nCBF-sim")
    directory_path = os.path.join(save_dir, formatted_time)
    os.makedirs(directory_path, exist_ok=True)

    np.save(os.path.join(directory_path, "state.npy"), state_log)
    np.save(os.path.join(directory_path, "nom_input.npy"), nom_input_log)
    np.save(os.path.join(directory_path, "safe_input.npy"), safe_input_log)
    np.save(os.path.join(directory_path, "error.npy"), error_log)
    np.save(os.path.join(directory_path, "status_log.npy"), status_log)
    np.save(os.path.join(directory_path, "phi_val_log.npy"), phi_val_log)
    np.save(os.path.join(directory_path, "params.npy"), 
        {"mode": mode,
        "hz": hz,
        "track_type": track_type,
        "mass": mass,
        "takeoff_height": takeoff_height,
        "pend_upright_time": pend_upright_time,
        "pend_upright_tol": pend_upright_tol,
        "lqr_itr": lqr_itr,
        "cont_duration": cont_duration,
        "lqr_cont_type": lqr_cont_type,
        "exp_name": exp_name,
        "ckpt_num": ckpt_num,
        "eps_bdry": eps_bdry,
        "eps_outside": eps_outside,
        "dynamics_noise_spread": dynamics_noise_spread})

    print("#####################################################")
    print(f"########### LOG DATA SAVED IN {formatted_time} ###########")
    print("#####################################################")
