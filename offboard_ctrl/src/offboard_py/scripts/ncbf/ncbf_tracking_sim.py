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

# from src.controllers.torque_constant_position_tracker import TorqueConstantPositionTracker
from src.controllers.eth_constant_controller import ConstantPositionTracker
from src.callbacks.fcu_state_callbacks import VehicleStateCB
from src.callbacks.pend_state_callbacks import PendulumCB
from src.callbacks.fcu_modes import FcuModes

from src.neural_cbf.utils import load_phi_and_params
from src.neural_cbf.ncbf_numpy_wrapper import NCBFNumpy
from src.neural_cbf.ncbf_controller_bodyrate import NCBFControllerBodyRate
from src.env.deploy_flying_inv_pend import FlyingInvertedPendulumEnv
from src.controllers.torque_lqr import TorqueLQR


class NCBFTrackingNode:
    def __init__(self, 
                 vehicle,
                 cont_type,
                 exp_name,
                 ckpt_num,
                 mode,
                 hz, 
                 track_type,
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

        ### Device = CPU ###
        device = torch.device("cpu")
        ### Load the Neural CBF ###
        self.exp_name = exp_name
        self.ckpt_num = ckpt_num
        torch_ncbf_fn, param_dict = load_phi_and_params(exp_name, ckpt_num, device)
        self.ncbf_fn = NCBFNumpy(torch_ncbf_fn, device)
        ### Environment ###
        self.env = FlyingInvertedPendulumEnv(vehicle=vehicle, dt=1/hz, model_param_dict=param_dict, 
                                             dynamics_noise_spread=dynamics_noise_spread)
        self.env.dt = 1/hz
        self.ncbf_cont = NCBFControllerBodyRate(vehicle, self.env, self.ncbf_fn, param_dict, eps_bdry=eps_bdry, eps_outside=eps_outside)

        ######################
        #####   PARAMS   #####
        ######################
        self.vehicle = vehicle                  # Type of vehicle: danaus12_old, danaus12_newold
        self.mode = mode                        # "sim" or "real"
        self.hz = hz                            # Control Loop Frequency
        self.track_type = track_type            # "constant" or "circular"
        self.takeoff_height = takeoff_height    # Takeoff Height
        
        if self.vehicle == "danaus12_old":
            self.pend_z_pose = 0.531659
            self.mass = 0.67634104 + 0.03133884
            self.L = 0.5    # unlike L_p, we are using CoM in LQR.
        elif self.vehicle == "danaus12_newold":
            NotImplementedError(f"{vehicle} not implemented in NCBFTrackingNode")
            # self.pend_z_pose = 0.721659
            # # self.pend_z_pose = 0.69
            # # self.L = 0.69
            # self.L = 0.79
            # self.mass = 0.70034104 + 0.046 #0.03133884

        ### Neural Controller Params ###
        self.nx_cbf = 16    # Including vehicle angular velocities in the world frame

        ### Nominal Controller Params ###
        self.cont_type = cont_type
        self.Q = Q
        self.R = R
        self.dt = 1/self.hz                     # Time step
        self.lqr_itr = lqr_itr                  # Number of iterations for Infinite-Horizon LQR
        self.cont_duration = cont_duration      # Duration for which the controller should run (in seconds)
        self.lqr_cont_type = lqr_cont_type      # "with_pend" or "without_pend"

        self.nx = 13
        self.nu = 4

        ### Pendulum Params ###
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

        ### Takeoff Controller ###
                                # γ, β, α, x, y, z, x_dot, y_dot, z_dot, pendulum (4)
        Q_takeoff = 1.0 * np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])      # Without pendulum
        R_takeoff = 1.0 * np.diag([1, 10, 10, 10])
        self.takeoff_cont = ConstantPositionTracker(self.cont_type, self.L, Q_takeoff, R_takeoff, takeoff_pose, self.dt)
        self.takeoff_K_inf = self.takeoff_cont.infinite_horizon_LQR(self.lqr_itr)
        self.takeoff_goal = self.takeoff_cont.xgoal
        self.takeoff_input = self.takeoff_cont.ugoal

        ### Nominal Controller ###
        if (self.track_type == "constant") and (self.lqr_cont_type == "with_pend"):
            self.cont = ConstantPositionTracker(self.cont_type, self.L, self.Q, self.R, np.array([0, 0, self.takeoff_height]), self.dt)
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
        rospy.init_node('ncbf_tracking_node', anonymous=True)

    def _quad_takeoff_controller(self):
        quad_xyz = self.quad_cb.get_xyz_pose()
        quad_xyz_vel = self.quad_cb.get_xyz_velocity()
        quad_xyz_ang = self.quad_cb.get_xyz_angles()
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
    
    def _quad_lqr_controller(self, x):
        u_bodyrate = self.ugoal - self.cont_K_inf @ (x - self.xgoal)
        return u_bodyrate

    def _send_attitude_setpoint(self, u):
        """ u[0]: Thrust, u[1]: Roll, u[2]: Pitch, u[3]: Yaw """
        self.att_setpoint.header.stamp = rospy.Time.now()
        self.att_setpoint.body_rate.x = u[1]
        self.att_setpoint.body_rate.y = u[2]
        self.att_setpoint.body_rate.z = u[3]
        self.att_setpoint.thrust = (u[0]/(9.81)) * self.hover_thrust
        self.att_setpoint.thrust = np.clip(self.att_setpoint.thrust, 0.0, 1.0)
        self.quad_att_setpoint_pub.publish(self.att_setpoint) 

    def _get_states(self):
        quad_xyz = self.quad_cb.get_xyz_pose()
        quad_xyz_vel = self.quad_cb.get_xyz_velocity()
        quad_xyz_ang = self.quad_cb.get_xyz_angles()
        quad_xyz_ang_vel = self.quad_cb.get_xyz_angular_velocity()   # TODO: Need to verify
        if self.cont_type == "tp":
            pend_pos = self.pend_cb.get_rs_ang(vehicle_pose=quad_xyz)
            pend_vel = self.pend_cb.get_rs_ang_vel(vehicle_vel=quad_xyz_vel)
        else:
            NotImplementedError(f"{self.cont_type} not implemented in NCBFTrackingNode")

        x_safe = np.concatenate((quad_xyz_ang.T, quad_xyz_ang_vel.T, pend_pos.T, pend_vel.T, quad_xyz.T, quad_xyz_vel.T))
        x_safe = x_safe.reshape((self.nx_cbf, 1))

        x_nom = np.concatenate((quad_xyz_ang.T, quad_xyz.T, quad_xyz_vel.T, pend_pos.T, pend_vel.T))
        x_nom = x_nom.reshape((self.nx, 1))
        return x_safe, x_nom

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
            NotImplementedError(f"{self.mode} not implemented in NCBFTrackingNode")

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

    def _pend_upright_sim(self, req_time=0.5, tol=0.05):
        get_link_properties_service = rospy.ServiceProxy('/gazebo/get_link_properties', GetLinkProperties)
        self.set_link_state_service = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        set_link_properties_service = rospy.ServiceProxy('/gazebo/set_link_properties', SetLinkProperties)

        consecutive_time = rospy.Duration(0.0)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            # # Keep the quadrotor at this pose!
            # self.quad_pose_pub.publish(self.takeoff_pose)  
            self._quad_takeoff_controller()     # This is better than takeoff_pose
            
            link_state = LinkState()
            # link_state.pose.position.x = -0.001995
            # link_state.pose.position.y = 0.000135
            link_state.pose.position.x = 0.001995
            link_state.pose.position.y = -0.000135
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
                    self._quad_takeoff_controller() 
                    # rospy.sleep(0.5)
                    return True
            else:
                consecutive_time = rospy.Duration(0.0)
                start_time = rospy.Time.now()

            self.rate.sleep()
            
    def run(self, duration):
        # Log Array
        num_itr = int(duration * self.hz)
        state_log = np.zeros((self.nx_cbf, num_itr))    # 16
        nom_input_log = np.zeros((self.nu, num_itr))    # 4
        safe_input_log = np.zeros((self.nu, num_itr))   # 4
        # error_log = np.zeros((self.nx_cbf, num_itr))    # 16
        # xgoal_log = np.zeros((self.nx_cbf, num_itr))    # 16
        status_log = np.zeros((1, num_itr))
        phi_val_log = np.zeros((1, num_itr))

        ##### Takeoff Sequence #####
        self._takeoff_sequence()
        
        ##### Pendulum Mounting #####
        if self.mode == "sim":
            rospy.loginfo("Swing the pendulum upright.")
            self._pend_upright_sim(req_time=self.pend_upright_time, tol=self.pend_upright_tol)
        else:
            rospy.loginfo("Pendulum mounting not implemented for real mode.")

        ##### Setting near origin & upright #####
        if self.mode == "sim":
            rospy.loginfo("Setting near origin & upright")
            for _ in range(150):
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
                self._quad_takeoff_controller()     # This is better than takeoff_pose
        else:
            pass

        ##### Pendulum Position Control #####
        rospy.loginfo("Starting the constant position control!")
        start_time = rospy.Time.now()

        # for itr in range(int(1e10)):  # If you want to test with console.
        for itr in range(num_itr):
            if rospy.is_shutdown():
                rospy.loginfo_throttle(3, "Node shutdown detected. Exiting the control loop.")
                break
            
            x_safe, x_nom = self._get_states()
            u_bodyrate = self._quad_lqr_controller(x_nom)
            u_safe_bodyrate, stat, phi_val = self.ncbf_cont.compute_control(x_safe, np.copy(u_bodyrate))
            rospy.loginfo(f"Itr.{itr}/{num_itr}\nU Safe: {u_safe_bodyrate.flatten()}\nU Nom: {u_bodyrate.flatten()}\nPhi: {phi_val}, Status\n{stat}")

            self._send_attitude_setpoint(u_safe_bodyrate)
            
            # Log the state and input
            state_log[:, itr] = x_safe.flatten()
            nom_input_log[:, itr] = u_bodyrate.flatten()
            safe_input_log[:, itr] = u_safe_bodyrate.flatten()
            # error_log[:, itr] = (x_nom - self.torque_LQR.xgoal).flatten()
            # xgoal_log[:, itr] = self.torque_LQR.xgoal.flatten()
            status_log[:, itr] = stat
            phi_val_log[:, itr] = phi_val

            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log, nom_input_log, safe_input_log, status_log, phi_val_log
        # return state_log, nom_input_log, safe_input_log, error_log, xgoal_log, status_log, phi_val_log


if __name__ == "__main__":
    
    ##### Argparse #####
    parser = argparse.ArgumentParser(description="NCBF Tracking Node")
    parser.add_argument("--mode", type=str, default="real", help="Mode of operation (sim or real)")
    parser.add_argument("--hz", type=int, default=90, help="Frequency of the control loop")
    parser.add_argument("--track_type", type=str, default="constant", help="Type of tracking to be used")
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

    parser.add_argument("--vehicle", type=str, default="danaus12_old", help="Type of vehicle (danaus12_old, danaus12_newold)")
    parser.add_argument("--cont_type", type=str, default="tp", help="Either rs or tp")    # rs: r and s, tp: theta and phi

    args = parser.parse_args()
    mode = args.mode
    hz = args.hz
    track_type = args.track_type
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
    vehicle = args.vehicle
    cont_type = args.cont_type

    print("######################################################")
    print("## NCBF Tracking Node for Constant Position Started ##")
    print("######################################################")
    print("")
    

    if cont_type == "rs":
                           # γ, β, α, x, y, z, x_dot, y_dot, z_dot, r, s, r_dot, s_dot
        Q = 1.0 * np.diag([0.0, 0.0, 0.0, 2, 2, 2, 0.0, 0.0, 0.0, 2.0, 2.0, 0.4, 0.4])
        R = 1.0 * np.diag([1, 7, 7, 7])
    elif cont_type == "tp":
                           # γ, β, α, x, y, z, x_dot, y_dot, z_dot, θ, ϕ, θ_dot, ϕ_dot
        Q = 1.0 * np.diag([0.0, 0.0, 0.0, 2, 2, 2, 0.0, 0.0, 0.0, 4.0, 4.0, 0.4, 0.4])
        R = 1.0 * np.diag([1, 7, 7, 7])
        
    ncbf_node = NCBFTrackingNode(
        vehicle=vehicle, cont_type=cont_type,
        exp_name=exp_name, ckpt_num=ckpt_num,
        mode=mode, hz=hz, track_type=track_type, 
        Q=Q, R=R,
        eps_bdry=eps_bdry, eps_outside=eps_outside, dynamics_noise_spread=dynamics_noise_spread,
        lqr_cont_type=lqr_cont_type, 
        takeoff_height=takeoff_height, 
        lqr_itr=lqr_itr, 
        pend_upright_time=pend_upright_time, 
        pend_upright_tol=pend_upright_tol)
    
    state_log, nom_input_log, safe_input_log, error_log, xgoal_log, status_log, phi_val_log = ncbf_node.run(duration=cont_duration)
    print("####################################################")
    print("## NCBF Tracking Node for Constant Position Over  ##")
    print("####################################################")
    print("")

    #################################
    ######### Save the logs #########
    #################################

    # curr_dir = os.getcwd()
    save_dir = "/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/ncbf"

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
    np.save(os.path.join(directory_path, "xgoal.npy"), xgoal_log)
    np.save(os.path.join(directory_path, "status_log.npy"), status_log)
    np.save(os.path.join(directory_path, "phi_val_log.npy"), phi_val_log)
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
            "lqr_cont_type": lqr_cont_type,
            "exp_name": exp_name,
            "ckpt_num": ckpt_num,
            "eps_bdry": eps_bdry,
            "eps_outside": eps_outside,
            "dynamics_noise_spread": dynamics_noise_spread,
            "vehicle": vehicle,
            "cont_type": cont_type,
            "Q": Q,
            "R": R
        })

    print("#####################################################")
    print(f"########### LOG DATA SAVED IN {formatted_time} ###########")
    print("#####################################################")
