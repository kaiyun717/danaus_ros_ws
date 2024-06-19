"""
Script for running the ETH constant position tracking controller.
"""
#! /usr/bin/env python

import numpy as np
import scipy
import scipy.linalg
import IPython


class QuadLQRController:
    def __init__(self, Q, R, xyz_goal, dt) -> None:
        """
        States: [gamma, beta, alpha, x, y, z, xdot, ydot, zdot]
        Inputs: [a, wx, wy, wz]
        """

        self.Q = Q      # State cost matrix
        self.R = R      # Input cost matrix
        self.dt = dt    # Time step
        g = 9.81   # Gravity
        self.nx = 9    # Number of states
        self.nu = 4     # Number of inputs

        self.Ac = np.array([[ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 1, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 1, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [ 0, g, 0, 0, 0, 0, 0, 0, 0],
                            [-g, 0, 0, 0, 0, 0, 0, 0, 0],
                            [ 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        self.Bc = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [1, 0, 0, 0],])
        exp_matrix = scipy.linalg.expm(np.block([[self.Ac, self.Bc], [np.zeros((self.nu, self.nx + self.nu))]]) * self.dt)
        self.Ad = exp_matrix[:self.nx, :self.nx]
        self.Bd = exp_matrix[:self.nx, self.nx:]
        self.xgoal = (np.array([0, 0, 0, xyz_goal[0], xyz_goal[1], xyz_goal[2], 0, 0, 0]))   
        self.ugoal = (np.array([g, 0, 0, 0]))

        self.xgoal = self.xgoal.reshape((self.nx, 1))
        self.ugoal = self.ugoal.reshape((self.nu, 1))
    
    def infinite_horizon_LQR(self, num_itr=100000):
        P = np.copy(self.Q)
        K_old = np.linalg.inv(self.R + self.Bd.T @ P @ self.Bd) @ self.Bd.T @ P @ self.Ad
        P_old = self.Q + self.Ad.T @ P @ (self.Ad - self.Bd @ K_old)

        for i in range(num_itr):
            K_new = np.linalg.inv(self.R + self.Bd.T @ P_old @ self.Bd) @ self.Bd.T @ P_old @ self.Ad
            P_new = self.Q + self.Ad.T @ P_old @ (self.Ad - self.Bd @ K_new)
            if np.linalg.norm(K_new - K_old) < 1e-9:
                print("Infinite horizon LQR converged at iteration ", i)
                print("LQR Gain: \n", K_new)
                # print("Kx_wx: ", K_new[1, 0], " Kr_wx: ", K_new[1, -4])
                # print("Ky_wy: ", K_new[0, 1], " Ks_wy: ", K_new[0, -3])
                # print("Krd_wx: ", K_new[1, -2], " Ksd_wy: ", K_new[0, -1])
                return K_new
            else:
                K_old = K_new
                P_old = P_new

        print("LQR did not converge")


if __name__ == "__main__":
    
    ## Check that the LQR is working
    Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    R = 1.0 * np.diag([1, 1, 1, 1])

    pos_tracker = QuadLQRController("rs_og", 0.5, Q, R, np.zeros((3,1)), 0.02)
    K = pos_tracker.infinite_horizon_LQR()