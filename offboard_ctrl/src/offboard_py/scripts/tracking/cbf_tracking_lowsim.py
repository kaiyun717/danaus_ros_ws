"""
This script is used to track the position of the drone using the ETH controller, 
based on the paper "A Flying Inverted Pendulum" by Raffaello D'Andrea and Markus Hehn.
"""
#!/usr/bin/env python

import IPython

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
from src.controllers.cbf_controller import CBFController
from src.callbacks.fcu_state_callbacks import VehicleStateCB
from src.callbacks.pend_state_callbacks import PendulumCB
from src.callbacks.fcu_modes import FcuModes

g = 9.81

class LowFidelitySim:
    def __init__(self, L_p, qp_weight, u_max, u_min, dt=1/90, delta_max=np.pi/4, rs_max=0.35,
                 kappa=8.7899304e3, n1=2.15029699, n2=1, k=0.01, 
                 takeoff_height=1.5, 
                 lqr_itr=10000) -> None:
        
        self.L = L_p        
        self.takeoff_height = takeoff_height
        
        self.cbf_filter = CBFController(qp_weight=qp_weight, u_max=u_max, u_min=u_min, L_p=self.L, delta_max=delta_max, rs_max=rs_max, kappa=kappa, n1=n1, n2=n2, k=k)

        self.dt = dt
        
        self.target_pose = np.array([0, 0, takeoff_height])
        Q_nom = 1.0 * np.diag([1, 1, 1, 0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0])
        R_nom = 1.0 * np.diag([50, 50, 50, 1])
        self.nom_cont = ConstantPositionTracker(self.L, Q_nom, R_nom, self.target_pose, self.dt)
        self.nom_K_inf = self.nom_cont.infinite_horizon_LQR(lqr_itr)
        self.nom_goal = self.nom_cont.xgoal
        self.nom_input = self.nom_cont.ugoal

    def dynamics(self, x, u):
        
        # IPython.embed()

        x, y, z, xdot, ydot, zdot, alpha, beta, gamma, r_pend, s_pend, dr_pend, ds_pend = x.flatten()
        alpha = self._convert_angle_to_negpi_pi_interval(alpha)
        beta = self._convert_angle_to_negpi_pi_interval(beta)
        gamma = self._convert_angle_to_negpi_pi_interval(gamma)

        R = np.zeros((3, 3)) # Rz*Ry*Rx
        R[0, 0] = np.cos(alpha)*np.cos(beta)
        R[0, 1] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
        R[0, 2] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
        R[1, 0] = np.sin(alpha)*np.cos(beta)
        R[1, 1] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
        R[1, 2] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
        R[2, 0] = -np.sin(beta)
        R[2, 1] = np.cos(beta)*np.sin(gamma)
        R[2, 2] = np.cos(beta)*np.cos(gamma)
        
        A = np.zeros((3, 3))	# Inv(anglular_velocity_matrix)
        A[0, 0] = np.cos(gamma)/(np.cos(beta)*np.cos(gamma)**2 + np.cos(beta)*np.sin(gamma)**2)
        A[0, 1] = np.sin(gamma)/(np.cos(beta)*np.cos(gamma)**2 + np.cos(beta)*np.sin(gamma)**2)
        A[0, 2] = 0.0
        A[1, 0] = -np.sin(gamma)#/(np.cos(gamma)**2 + np.sin(gamma)**2)
        A[1, 1] = np.cos(gamma)#/(np.cos(gamma)**2 + np.sin(gamma)**2)
        A[1, 2] = 0.0
        A[2, 0] = (np.cos(gamma)*np.sin(beta))/(np.cos(beta)*np.cos(gamma)**2 + np.cos(beta)*np.sin(gamma)**2)
        A[2, 1] = (np.sin(gamma)*np.sin(beta))/(np.cos(beta)*np.cos(gamma)**2 + np.cos(beta)*np.sin(gamma)**2)
        A[2, 2] = 1.0

        # IPython.embed()

        xyz_ddot = R@np.array([0, 0, u[3][0]]).reshape((3,1)) - np.array([0, 0, g]).reshape((3,1))
        ang_dot = A@u[:3].reshape((3,1))

        # IPython.embed()

        x_ddot = xyz_ddot[0][0]
        y_ddot = xyz_ddot[1][0]
        z_ddot = xyz_ddot[2][0]

        # print("X_ddot: ", x_ddot)
        # print("Y_ddot: ", y_ddot)
        # print("Z_ddot: ", z_ddot)
        # IPython.embed()

        L_p = self.L
        r_ddot = (L_p**4*x_ddot - r_pend**3*ds_pend**2 + r_pend**4*x_ddot + r_pend*s_pend**3*y_ddot 
					+ r_pend**3*s_pend*y_ddot + L_p**2*r_pend*dr_pend**2 + L_p**2*r_pend*ds_pend**2 
					- 2*L_p**2*r_pend**2*x_ddot - L_p**2*s_pend**2*x_ddot + g*r_pend**3*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
					- r_pend*dr_pend**2*s_pend**2 + r_pend**3*z_ddot*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
					+ r_pend**2*s_pend**2*x_ddot - L_p**2*r_pend*s_pend*y_ddot + 2*r_pend**2*dr_pend*s_pend*ds_pend 
					- L_p**2*g*r_pend*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
					- L_p**2*r_pend*z_ddot*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
					+ g*r_pend*s_pend**2*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
					+ r_pend*s_pend**2*z_ddot*(L_p**2 - r_pend**2 - s_pend**2)**(1/2)) \
				/(L_p**2*(- L_p**2 + r_pend**2 + s_pend**2))
        s_ddot = (L_p**4*y_ddot - dr_pend**2*s_pend**3 + s_pend**4*y_ddot + r_pend*s_pend**3*x_ddot 
                    + r_pend**3*s_pend*x_ddot + L_p**2*dr_pend**2*s_pend + L_p**2*s_pend*ds_pend**2 
                    - L_p**2*r_pend**2*y_ddot - 2*L_p**2*s_pend**2*y_ddot + g*s_pend**3*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
                    - r_pend**2*s_pend*ds_pend**2 + s_pend**3*z_ddot*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
                    + r_pend**2*s_pend**2*y_ddot - L_p**2*r_pend*s_pend*x_ddot + 2*r_pend*dr_pend*s_pend**2*ds_pend 
                    - L_p**2*g*s_pend*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
                    - L_p**2*s_pend*z_ddot*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
                    + g*r_pend**2*s_pend*(L_p**2 - r_pend**2 - s_pend**2)**(1/2) 
                    + r_pend**2*s_pend*z_ddot*(L_p**2 - r_pend**2 - s_pend**2)**(1/2)) \
                /(- L_p**4 + L_p**2*r_pend**2 + L_p**2*s_pend**2)
        

        return np.array([xdot, ydot, zdot, x_ddot, y_ddot, z_ddot, ang_dot[2][0], ang_dot[1][0], ang_dot[0][0], dr_pend, ds_pend, r_ddot, s_ddot]).reshape((13,1))

    def _convert_angle_to_negpi_pi_interval(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def rk4(self, x, u):
        k1 = self.dt*self.dynamics(x, u)
        k2 = self.dt*self.dynamics(x + 0.5*k1, u)
        k3 = self.dt*self.dynamics(x + 0.5*k2, u)
        k4 = self.dt*self.dynamics(x + k3, u)
        return x + (1/6)*(k1 + 2*k2 + 2*k3 + k4)
        # return x + k1
    
    def run(self, duration):
        num_itr = int(duration * 90)
        state_log = np.zeros((13, num_itr))
        u_nom_log = np.zeros((4, num_itr-1))
        u_safe_log = np.zeros((4, num_itr-1))
        error_log = np.zeros((13, num_itr))
        
        # x = np.hstack((self.target_pose, np.array([0,0,0,0,0,0,0,0,0,0]))).reshape(13,1)
        x = np.hstack((np.array([0.1, 0.1, self.takeoff_height]), np.array([0,0,0,0,0,0,0.0,0.0,0,0]))).reshape(13,1)
        state_log[:, 0] = x.flatten()
        
        for itr in range(num_itr-1):
            
            # if itr == 104:
            #     IPython.embed()

            x[6] = self._convert_angle_to_negpi_pi_interval(x[6])
            x[7] = self._convert_angle_to_negpi_pi_interval(x[7])
            x[8] = self._convert_angle_to_negpi_pi_interval(x[8])

            u_nom = self.nom_input - self.nom_K_inf @ (x - self.nom_goal)
            
            if np.abs(self.cbf_filter.cbf.phi_fn(x[6:].reshape(-1))) < 1e-1:
                print("###################")
                print("## Near boundary ##")
                print("###################")
                print("h(x) = ", self.cbf_filter.cbf.phi_fn(x[6:].reshape(-1)))
                try:
                    u_safe = self.cbf_filter.solve_qp(x[6:].reshape(-1), np.roll(u_nom, 1))
                    u_safe = np.roll(u_safe, -1)
                except ValueError as e:
                    print("Error: ", e)
                    return state_log[:,:itr], u_nom_log[:,:itr], u_safe_log[:,:itr], error_log[:,:itr]
            else:
                u_safe = u_nom

            # IPython.embed()
            x = self.rk4(x, u_safe)

            if np.isnan(x).any():
                print("NAN in x")
                return state_log[:,:itr], u_nom_log[:,:itr], u_safe_log[:,:itr], error_log[:,:itr]
            
            state_log[:, itr+1] = x.flatten()
            u_nom_log[:, itr] = u_nom.flatten()
            u_safe_log[:, itr] = u_safe.flatten()
            error_log[:, itr+1] = (x - self.nom_goal).flatten()

            print("Iteration: ", itr)
            print("State: ", x.flatten())
            print("Nominal Input: ", u_nom.flatten())
            print("Safe Input: ", u_safe.flatten())
            print("Error: ", (x - self.nom_goal).flatten())
        
        return state_log, u_nom_log, u_safe_log, error_log


if __name__ == "__main__":
    L_p = 0.69

    qp_weight = np.diag([50, 10, 10, 50]) * 10.0
    # qp_weight = np.diag([1, 10, 10, 10]) * 10.0
    # kappa = 8.7899304e3
    # # kappa = 1000
    # # n1 = 2.15029699
    # n1 = 3.76
    # n2 = 1.0
    # k = 0.01
    kappa = 4.91654058e+01 
    n1 = 1.97677007e+00 
    n2 = 1.03804057e+00 
    k = 1.00000000e-02

    delta_max = np.pi/4
    rs_max = 0.15

    u_max = np.array([2*g, 15, 15, 15]).reshape((4,1))
    u_min = np.array([0, -15, -15, -15]).reshape((4,1))
    hz = 90

    low_sim = LowFidelitySim(L_p, qp_weight, u_max, u_min, dt=1/hz, 
                             delta_max=delta_max, rs_max=rs_max,
                             kappa=kappa, n1=n1, n2=n2, k=k, 
                             takeoff_height=1.5, 
                             lqr_itr=10000)
    
    state_log, u_nom_log, u_safe_log, error_log = low_sim.run(15)

    #################################
    ######### Save the logs #########
    #################################

    # curr_dir = os.getcwd()
    save_dir = "/home/oem/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/low_sim"

    # Get current date and time
    current_time = datetime.datetime.now()
    # Format the date and time into the desired filename format
    formatted_time = current_time.strftime("%m%d_%H%M%S-Sim")
    directory_path = os.path.join(save_dir, formatted_time+"-og-kappa_{:.2f}_n1_{:.2f}_n2_{:.2f}_k_{:.2f}".format(kappa, n1, n2, k))
    os.makedirs(directory_path, exist_ok=True)

    np.save(os.path.join(directory_path, "state.npy"), state_log)
    np.save(os.path.join(directory_path, "u_nom.npy"), u_nom_log)
    np.save(os.path.join(directory_path, "u_safe.npy"), u_safe_log)
    np.save(os.path.join(directory_path, "error.npy"), error_log)
    np.save(os.path.join(directory_path, "gains.npy"), {"kappa": kappa, "n1": n1, "n2": n2, "k": k, "qp_weight": qp_weight, "u_max": u_max, "u_min": u_min, "delta_max": delta_max, "rs_max": rs_max, "L_p": L_p})

    print("#####################################################")
    print("################# LOG DATA SAVED ####################")
    print("#####################################################")

    
    