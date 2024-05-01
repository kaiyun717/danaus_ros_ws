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
                 device,
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

        
        self.exp_name = exp_name
        self.ckpt_num = ckpt_num
        torch_ncbf_fn, param_dict = load_phi_and_params(exp_name, ckpt_num, device)
        torch_ncbf_fn.eval()

        self.ncbf_fn = NCBFNumpy(torch_ncbf_fn, device)
        print(f"{self.ncbf_fn.device=}")
        self.env = FlyingInvertedPendulumEnv(dt=1/hz, model_param_dict=param_dict, 
                                             dynamics_noise_spread=dynamics_noise_spread)
        self.env.dt = 1/hz

        self.ncbf_cont = NCBFController(self.env, self.ncbf_fn, param_dict, eps_bdry=eps_bdry, eps_outside=eps_outside)

        ############# Torch vs Numpy #############
        import time
        x_torch = torch.rand(1,10).to(device)
        x_np = np.random.rand(1,10)

        torch_start_time = time.time()
        phi_torch = torch_ncbf_fn(x_torch)
        torch_end_time = time.time()
        print(f"Time taken for torch: {torch_end_time - torch_start_time}")

        numpy_start_time = time.time()
        phi_numpy = self.ncbf_fn.phi_fn(x_np)
        numpy_end_time = time.time()
        print(f"Time taken for numpy: {numpy_end_time - numpy_start_time}")

        inside_x = np.zeros((16,1))
        inside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        compute_start_time = time.time()
        u_safe, stat, phi_val = self.ncbf_cont.compute_control(inside_x, inside_u)
        compute_end_time = time.time()
        print(f"Time taken for compute control for inside: {compute_end_time - compute_start_time}")

        outside_x = np.zeros((16,1))
        outside_x[0] = np.pi/6
        outside_x[1] = np.pi/6
        outside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        compute_start_time = time.time()
        u_safe, stat, phi_val = self.ncbf_cont.compute_control(outside_x, outside_u)
        compute_end_time = time.time()
        print(f"Time taken for compute control for outside: {compute_end_time - compute_start_time}")




        IPython.embed()

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
                gains_dict_px4 = {"MC_PITCHRATE_P": 0.138,
                                  "MC_PITCHRATE_I": 0.168,
                                  "MC_PITCHRATE_D": 0.0028,
                                  "MC_ROLLRATE_P": 0.094,
                                  "MC_ROLLRATE_I": 0.118,
                                  "MC_ROLLRATE_D": 0.0017,
                                  "MC_YAWRATE_P": 0.1,
                                  "MC_YAWRATE_I": 0.11,
                                  "MC_YAWRATE_D": 0.0}
                self.torque_LQR = TorqueConstantPositionTracker(K_inf, takeoff_pose, gains_dict_px4, self.dt)
            elif self.lqr_cont_type == "without_pend":
                K_inf = np.array([
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.990423659437537, 0.0, 0.0, 1.7209841208008194],
                            [1.4814681367499676, 0.002995481626744845, 0.005889427337762064, 0.28201313626232954, 0.0005738532722789398, 0.005981686140109121, 0.0005315334025884319, -0.2644839556489373, 0.0, 0.000779188657396088, -0.3870845425896414, 0.0],
                            [0.002995437953182237, 1.4528645588948705, 0.0034553424254151212, 0.0005738451434393372, 0.27654055987779025, 0.003509251033197913, 0.2594021552915394, -0.0005315255954541553, 0.0, 0.3796374382180433, -0.0007791772254469898, 0.0],
                            [0.03160126871064939, 0.018545750101093363, 0.39799289325860154, 0.006079664965265176, 0.003567296347199587, 0.4032536566450523, 0.003278753893103592, -0.005585886845822208, 0.0, 0.004811104113305738, -0.008196880671368846, 0.0]
                        ])
                gains_dict_px4 = {"MC_PITCHRATE_P": 0.138,
                                  "MC_PITCHRATE_I": 0.168,
                                  "MC_PITCHRATE_D": 0.0028,
                                  "MC_ROLLRATE_P": 0.094,
                                  "MC_ROLLRATE_I": 0.118,
                                  "MC_ROLLRATE_D": 0.0017,
                                  "MC_YAWRATE_P": 0.1,
                                  "MC_YAWRATE_I": 0.11,
                                  "MC_YAWRATE_D": 0.0}
                self.torque_LQR = TorqueConstantPositionTracker(K_inf, takeoff_pose, gains_dict_px4, self.dt)
                
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
        if self.device.type == "cuda":
            u_nom = torch.tensor(np.zeros((self.nu, 1)), device=self.device, dtype=torch.float32)
        else:
            u_nom = np.zeros((self.nu, 1))
        return u_nom

    def _send_attitude_setpoint(self, u):
        print("Attitude setpoint publishing.")

    def _get_states(self):
        if self.device.type == "cuda":
            x_safe = torch.tensor(np.zeros((self.env.nx, 1)), device=self.device, dtype=torch.float32)
            x_nom = torch.tensor(np.zeros((self.torque_LQR.nx, 1)), device=self.device, dtype=torch.float32)
        else:
            x_safe = np.zeros((self.env.nx, 1))
            x_nom = np.zeros((self.torque_LQR.nx, 1))

        return x_safe, x_nom

    def _takeoff_sequence(self):
        print("Taking off.")

    def _pend_upright_sim(self, req_time=0.5, tol=0.05):
        print("Swinging the pendulum upright.")

    def run(self, duration):
        # Log Array
        num_itr = int(duration * self.hz)
        state_log = np.zeros((self.nx, num_itr))
        nom_input_log = np.zeros((self.nu, num_itr))
        safe_input_log = np.zeros((self.nu, num_itr))
        error_log = np.zeros((self.torque_LQR.nx, num_itr))
        status_log = np.zeros((1, num_itr))

        ##### Takeoff Sequence #####0411_193906-Real
        self._takeoff_sequence()
        
        ##### Pendulum Position Control #####
        rospy.loginfo("Starting the constant position control!")
        start_time = rospy.Time.now()

        # for itr in range(int(1e10)):  # If you want to test with console.
        for itr in range(num_itr):
            if rospy.is_shutdown():
                rospy.loginfo_throttle(3, "Node shutdown detected. Exiting the control loop.")
                break
            
            x_safe, x_nom = self._get_states()
            
            u_nom = self._quad_lqr_controller(x_nom)
            # IPython.embed()
            u_safe, stat = self.ncbf_cont.compute_control(x_safe, u_nom)
            # IPython.embed()
            u_safe = self.torque_LQR.torque_to_body_rate(u_safe)
            rospy.loginfo(f"Itr.{itr}/{num_itr}, U Safe: {u_safe}")
            
            self._send_attitude_setpoint(u_safe)
            
            # Log the state and input
            state_log[:, itr] = x_safe.flatten()
            nom_input_log[:, itr] = u_nom.flatten()
            safe_input_log[:, itr] = u_safe.flatten()
            error_log[:, itr] = (x_nom - self.torque_LQR.xgoal).flatten()
            status_log[:, itr] = stat

            self.rate.sleep()

        rospy.loginfo("Constant position control completed.")
        return state_log, nom_input_log, safe_input_log, error_log, status_log


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

  

    L = 0.69            # x  y  z  x_dot y_dot z_dot yaw pitch roll r s r_dot s_dot

    if lqr_cont_type == "with_pend":
        # Q = 1.0 * np.diag([50, 50, 4, 0, 0, 0.0, 0.0, 0, 0, 10, 10, 10,  10])      # With pendulum
        # # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        # R = 1.0 * np.diag([100, 100, 0.01, 0.9])

        Q = 1.0 * np.diag([3, 3, 2, 0.005, 0.005, 0.0, 0.0, 0.0, 0.0, 10, 10, 0.0001, 0.0001])      # With pendulum
        # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        R = 1.0 * np.diag([10, 10, 1, 1])

        # Q = 1.0 * np.diag([4, 4, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 16, 16, 0.4, 0.4])      # With pendulum
        # # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        # R = 1.0 * np.diag([6.5, 6.5, 1, 1])

        # Q = 1.0 * np.diag([2, 2, 2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2, 2, 0, 0])      # With pendulum
        # # Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])      # Without pendulum
        # R = 1.0 * np.diag([70, 70, 1, 1])
    else:
        # Qx = np.diag([0.3, 0, 0, 3, 0])
        # Rx = np.diag([6.5])
        # Qy = np.diag([0.3, 0, 0, 3, 0])
        # Ry = np.diag([6.5])
        # Qz = np.diag([2, 0])
        # Rz = np.diag([1])

        Qx = np.diag([2, 0, 0, 2, 0])
        Rx = np.diag([7])
        Qy = np.diag([2, 0, 0, 2, 0])
        Ry = np.diag([7])
        Qz = np.diag([2, 0])
        Rz = np.diag([1])

        Q = [Qx, Qy, Qz]
        R = [Rx, Ry, Rz]

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
        dev = "cuda:%i" % (0)
        print("Using GPU device: %s" % dev)
    else:
        dev = "cpu"
    device = torch.device(dev)

        
    ncbf_node = NCBFTrackingNode(
        exp_name=exp_name, ckpt_num=ckpt_num, device=device,
        mode=mode, hz=hz, track_type=track_type, 
        mass=mass, L=L, Q=Q, R=R,
        eps_bdry=eps_bdry, eps_outside=eps_outside, dynamics_noise_spread=dynamics_noise_spread,
        lqr_cont_type=lqr_cont_type, 
        takeoff_height=takeoff_height, 
        lqr_itr=lqr_itr, 
        pend_upright_time=pend_upright_time, 
        pend_upright_tol=pend_upright_tol)
    
    state_log, nom_input_log, safe_input_log, error_log, status_log = ncbf_node.run(duration=cont_duration)
    