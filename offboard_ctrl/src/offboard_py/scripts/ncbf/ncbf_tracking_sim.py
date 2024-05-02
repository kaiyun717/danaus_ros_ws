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
                self.torque_LQR = TorqueConstantPositionTracker(K_inf, takeoff_pose, self.dt)
                # self.torque_LQR = TorqueLQR(L, Q, R, takeoff_pose, self.dt, lqr_cont_type, num_itr=lqr_itr)
            elif self.lqr_cont_type == "without_pend":
                K_inf = np.array([
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9904236594375369, 0.0, 0.0, 1.7209841208008192],
                            [1.4816729161221946, 0.0, 0.0, 0.2820508569413449, 0.0, 0.0, 0.0, -0.2645216194466015, 0.0, 0.0, -0.38713923609434187, 0.0],
                            [0.0, 1.452935447187532, 0.0, 0.0, 0.2765536175628985, 0.0, 0.2594151935758526, 0.0, 0.0, 0.37965637166446786, 0.0, 0.0],
                            [0.0, 0.0, 0.3980386245466367, 0.0, 0.0, 0.4032993897561105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        ])
                self.torque_LQR = TorqueConstantPositionTracker(K_inf, takeoff_pose, self.dt)
                # self.torque_LQR = TorqueLQR(L, Q, R, takeoff_pose, self.dt, lqr_cont_type, num_itr=lqr_itr)
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

        x_nom = np.concatenate((quad_xyz_ang.T, quad_xyz_ang_vel.T, quad_xyz.T, quad_xyz_vel.T))
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

    def run(self, duration):
        # Log Array
        num_itr = int(duration * self.hz)
        state_log = np.zeros((self.nx, num_itr))
        nom_input_log = np.zeros((self.nu, num_itr))
        safe_input_log = np.zeros((self.nu, num_itr))
        error_log = np.zeros((self.torque_LQR.nx, num_itr))
        xgoal_log = np.zeros((self.torque_LQR.nx, num_itr))
        status_log = np.zeros((1, num_itr))
        phi_val_log = np.zeros((1, num_itr))

        ##### Takeoff Sequence #####0411_193906-Real
        self._takeoff_sequence()
        
        ##### Pendulum Position Control #####
        rospy.loginfo("Starting the constant position control!")
        start_time = rospy.Time.now()

        traj_goals = np.array([[2, 2, self.takeoff_height], [2, -2, self.takeoff_height], [-2, -2, self.takeoff_height], [-2, 2, self.takeoff_height]])

        # for itr in range(int(1e10)):  # If you want to test with console.
        for itr in range(num_itr):
            if rospy.is_shutdown():
                rospy.loginfo_throttle(3, "Node shutdown detected. Exiting the control loop.")
                break
            
            # Update goal
            if itr % 100 == 0:
                self.torque_LQR.update_goal(traj_goals[(itr//100) % len(traj_goals)])
                rospy.loginfo(f"Goal updated to: {self.torque_LQR.xgoal.flatten()}")

            x_safe, x_nom = self._get_states()
            
            u_torque = self._quad_lqr_controller(x_nom)
            u_body = self.torque_LQR.torque_to_body_rate(u_torque)
            self._send_attitude_setpoint(u_body)
            rospy.loginfo(f"Itr.{itr}/{num_itr}\nU Nom: {u_torque.flatten()}")

            u_safe, stat, phi_val = self.ncbf_cont.compute_control(x_safe, np.copy(u_torque))
            # u_safe_body = self.torque_LQR.torque_to_body_rate(u_safe)
            # rospy.loginfo(f"Itr.{itr}/{num_itr}\nU Safe: {u_safe.flatten()}\nU Nom: {u_torque.flatten()}\nPhi: {phi_val}, Status\n{stat}")

            # self._send_attitude_setpoint(u_safe_body)
            
            # Log the state and input
            state_log[:, itr] = x_safe.flatten()
            nom_input_log[:, itr] = u_torque.flatten()
            # safe_input_log[:, itr] = u_safe.flatten()
            error_log[:, itr] = (x_nom - self.torque_LQR.xgoal).flatten()
            xgoal_log[:, itr] = self.torque_LQR.xgoal.flatten()
            # status_log[:, itr] = stat
            phi_val_log[:, itr] = phi_val

            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log, nom_input_log, safe_input_log, error_log, xgoal_log, status_log, phi_val_log


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

        # Q = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10, 10, 0.0001, 0.0001, 3, 3, 2, 0.005, 0.005, 0.0])      # With pendulum
        Q = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.000, 0.000, 10, 10, 10, 0.005, 0.005, 0.0])      # Without pendulum
        
        R = 1.0 * np.diag([10, 10, 10, 1])

        # Q = 1.0 * np.diag([4, 4, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16, 16, 0.4, 0.4])      # With pendulum
        # # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        # R = 1.0 * np.diag([6.5, 6.5, 1, 1])

        # Q = 1.0 * np.diag([2, 2, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, 2, 0, 0])      # With pendulum
        # # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        # R = 1.0 * np.diag([70, 70, 1, 1])
    else:
        Q = None
        R = None
        
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
    
    state_log, nom_input_log, safe_input_log, error_log, xgoal_log, status_log, phi_val_log = ncbf_node.run(duration=cont_duration)
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
    np.save(os.path.join(directory_path, "xgoal.npy"), xgoal_log)
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
        "dynamics_noise_spread": dynamics_noise_spread,
        "K_inf": np.array([
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9904236594375369, 0.0, 0.0, 1.7209841208008192],
                            [1.4816729161221946, 0.0, 0.0, 0.2820508569413449, 0.0, 0.0, 0.0, -0.2645216194466015, 0.0, 0.0, -0.38713923609434187, 0.0],
                            [0.0, 1.452935447187532, 0.0, 0.0, 0.2765536175628985, 0.0, 0.2594151935758526, 0.0, 0.0, 0.37965637166446786, 0.0, 0.0],
                            [0.0, 0.0, 0.3980386245466367, 0.0, 0.0, 0.4032993897561105, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                        ])})

    print("#####################################################")
    print(f"########### LOG DATA SAVED IN {formatted_time} ###########")
    print("#####################################################")
