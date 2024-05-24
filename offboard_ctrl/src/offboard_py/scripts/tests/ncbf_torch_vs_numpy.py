"""
This script is used to track the position of the drone using the ETH controller, 
based on the paper "A Flying Inverted Pendulum" by Raffaello D'Andrea and Markus Hehn.
"""
#!/usr/bin/env python

import os
import math
import datetime

import IPython
import torch
import numpy as np
import scipy

import argparse

from src.neural_cbf.utils import load_phi_and_params
from src.neural_cbf.ncbf_numpy_wrapper import NCBFNumpy
from src.neural_cbf.ncbf_controller import NCBFController
from src.env.deploy_flying_inv_pend import FlyingInvertedPendulumEnv


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
        import time, pickle

        # print("######################################################")
        # outside_x = np.zeros((16,1))
        # outside_x[0] = np.pi/4
        # outside_x[1] = np.pi/4
        # outside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        # compute_start_time = time.time()
        # u_safe, stat, phi_val = self.ncbf_cont.compute_control(outside_x, outside_u)
        # compute_end_time = time.time()
        # print(f"Time taken for compute control for outside: {(compute_end_time - compute_start_time)*1000}")
        # print("######################################################")

        
        # num_samples = 0
        # total_time = 0

        # for _ in range(1000):
        #     outside_x = np.zeros((16,1))
        #     outside_x[0] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        #     outside_x[1] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        #     outside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        #     compute_start_time = time.time()
        #     u_safe, stat, phi_val = self.ncbf_cont.compute_control(outside_x, outside_u)
        #     compute_end_time = time.time()
        #     if stat == 0:
        #         num_samples += 1
        #         total_time += compute_end_time - compute_start_time
        
        # print(f"Average time taken for compute control for outside: {(total_time/num_samples)*1000}")

        # IPython.embed()


        print("######################################################")
        outside_x = np.zeros((16,1))
        outside_x[0] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        outside_x[1] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        outside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        compute_start_time = time.time()
        u_safe, stat, phi_val = self.ncbf_cont.compute_control(outside_x, outside_u)
        compute_end_time = time.time()
        print(f"Time taken for compute control for outside: {(compute_end_time - compute_start_time)*1000}")
        print("######################################################")

        print("\n######################################################")
        outside_x = np.zeros((16,1))
        outside_x[0] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        outside_x[1] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        outside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        compute_start_time = time.time()
        u_safe, stat, phi_val = self.ncbf_cont.compute_control(outside_x, outside_u)
        compute_end_time = time.time()
        print(f"Time taken for compute control for outside: {(compute_end_time - compute_start_time)*1000}")
        print("######################################################")

        print("\n######################################################")
        outside_x = np.zeros((16,1))
        outside_x[0] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        outside_x[1] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        outside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        compute_start_time = time.time()
        u_safe, stat, phi_val = self.ncbf_cont.compute_control(outside_x, outside_u)
        compute_end_time = time.time()
        print(f"Time taken for compute control for outside: {(compute_end_time - compute_start_time)*1000}")
        print("######################################################")

        print("\n######################################################")
        outside_x = np.zeros((16,1))
        outside_x[0] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        outside_x[1] = np.random.uniform(np.pi/4-0.1, np.pi/4+0.1)
        outside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        compute_start_time = time.time()
        u_safe, stat, phi_val = self.ncbf_cont.compute_control(outside_x, outside_u)
        compute_end_time = time.time()
        print(f"Time taken for compute control for outside: {(compute_end_time - compute_start_time)*1000}")
        print("######################################################")

        IPython.embed()

        ######## MULTIPLE ########
        timing_data = []

        # batch_sizes = [1, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]
        batch_sizes = [1]#, 2**3, 2**6, 2**9, 2**12, 2**15, 2**18, 2**21]

        for i in batch_sizes:

            x_torch = torch.rand(i,10).to(device)
            x_np = np.random.rand(i,10)

            _ = torch_ncbf_fn(torch.rand(1,10).to(device))
            _ = self.ncbf_fn.phi_fn(np.random.rand(1,10))

            print("######### TORCH #########")
            torch.cuda.synchronize()
            torch_start_time = time.time()
            phi_torch = torch_ncbf_fn(x_torch)
            torch.cuda.synchronize()
            torch_end_time = time.time()
            print(f"Time taken for torch: {torch_end_time - torch_start_time}")
            torch_time = torch_end_time - torch_start_time
            print(f"{phi_torch.shape=}")
            
            print("######### NUMPY #########")
            torch.cuda.synchronize()
            numpy_start_time = time.time()
            phi_numpy = self.ncbf_fn.phi_fn(x_np)
            torch.cuda.synchronize()
            numpy_end_time = time.time()
            print(f"Time taken for numpy: {numpy_end_time - numpy_start_time}")
            numpy_time = numpy_end_time - numpy_start_time
            print(f"{phi_numpy.shape=}")

            IPython.embed()
            
            timing_data.append([torch_time, numpy_time])

        # timing_array = np.array(timing_data)

        # with open('cpu_timing_data_3.pkl', 'wb') as f:
        #     pickle.dump({"timing_array": timing_array, "batch_sizes": batch_sizes}, f)


        # inside_x = np.zeros((16,1))
        # inside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        # compute_start_time = time.time()
        # u_safe, stat, phi_val = self.ncbf_cont.compute_control(inside_x, inside_u)
        # compute_end_time = time.time()
        # print(f"Time taken for compute control for inside: {compute_end_time - compute_start_time}")

        # outside_x = np.zeros((16,1))
        # outside_x[0] = np.pi/4
        # outside_x[1] = np.pi/4
        # outside_u = np.array([9.81, 0, 0, 0]).reshape((4,1))
        # compute_start_time = time.time()
        # u_safe, stat, phi_val = self.ncbf_cont.compute_control(outside_x, outside_u)
        # compute_end_time = time.time()
        # print(f"Time taken for compute control for outside: {compute_end_time - compute_start_time}")

        # IPython.embed()


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

    # if torch.cuda.is_available():
    #     os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    #     dev = "cuda:%i" % (0)
    #     print("Using GPU device: %s" % dev)
    # else:
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
    