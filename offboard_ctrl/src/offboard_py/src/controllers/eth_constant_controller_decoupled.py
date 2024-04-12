"""
Script for running the ETH constant position tracking controller.
"""
#! /usr/bin/env python

import numpy as np
import scipy


class ConstantPositionTrackerDecoupled:
    def __init__(self, L, Q: list, R: list, xyz_goal, dt) -> None:
        self.L = L      # length from the base of the pendulum to the center of mass
        self.Qx = Q[0]
        self.Rx = R[0]
        self.Qy = Q[1]
        self.Ry = R[1]
        self.Qz = Q[2]
        self.Rz = R[2]
        self.dt = dt    # Time step
        self.g = 9.81   # Gravity
        self.nx = 13    # Number of states
        self.nu = 4     # Number of inputs
        
        # Lateral: Ac: 5 x 5; Bc 5 x 1
        self.Acx = np.array([
            [0, 1,       0,             0, 0],
            [0, 0,  self.g,             0, 0],
            [0, 0,       0,             0, 0],
            [0, 0,       0,             0, 1],
            [0, 0, -self.g, self.g/self.L, 0]
        ])

        self.Bcx = np.array([
            [0],
            [0],
            [1],
            [0],
            [0]
        ])

        self.Acy = np.array([
            [0, 1,       0,             0, 0],
            [0, 0, -self.g,             0, 0],
            [0, 0,       0,             0, 0],
            [0, 0,       0,             0, 1],
            [0, 0,  self.g, self.g/self.L, 0]
        ])

        self.Bcy = np.array([
            [0],
            [0],
            [1],
            [0],
            [0]
        ])
        
        # Vertical Ac: 2x2, Bc: 2 x 1
        self.Acz = np.array([
            [0, 1],
            [0, 0]
        ])
        
        self.Bcz = np.array([
            [0],
            [1],
        ])
        
        exp_matrix_x = scipy.linalg.expm(np.block([[self.Acx, self.Bcx], [np.zeros((1, 6))]]) * self.dt)
        self.Adx = exp_matrix_x[:5, :5]
        self.Bdx = exp_matrix_x[:5, 5:]

        exp_matrix_y = scipy.linalg.expm(np.block([[self.Acy, self.Bcy], [np.zeros((1, 6))]]) * self.dt)
        self.Ady = exp_matrix_y[:5, :5]
        self.Bdy = exp_matrix_y[:5, 5:]

        exp_matrix_z = scipy.linalg.expm(np.block([[self.Acz, self.Bcz], [np.zeros((1, 3))]]) * self.dt)
        self.Adz = exp_matrix_z[:2, :2]
        self.Bdz = exp_matrix_z[:2, 2:]

        self.xgoal_x = (np.array([xyz_goal[0], 0, 0, 0, 0]))
        self.xgoal_y = (np.array([xyz_goal[1], 0, 0, 0, 0]))
        self.xgoal_z = (np.array([xyz_goal[2], 0]))   

        self.xgoal_x = self.xgoal_x.reshape((5,1))
        self.xgoal_y = self.xgoal_y.reshape((5,1))
        self.xgoal_z = self.xgoal_z.reshape((2,1))
        
        self.ugoal_x = (np.array([0])).reshape((1,1))
        self.ugoal_y = (np.array([0])).reshape((1,1))
        self.ugoal_z = (np.array([self.g])).reshape((1,1))
    
    def infinite_horizon_LQR(self, num_itr=10000):
        K_new_x = self._infinite_horizon_LQR(self.Qx, self.Rx, self.Adx, self.Bdx, num_itr)
        K_new_y = self._infinite_horizon_LQR(self.Qy, self.Ry, self.Ady, self.Bdy, num_itr)
        K_new_z = self._infinite_horizon_LQR(self.Qz, self.Rz, self.Adz, self.Bdz, num_itr)

        return K_new_x, K_new_y, K_new_z
    
    def _infinite_horizon_LQR(self, Q, R, Ad, Bd, num_itr=10000):
        P = np.copy(Q)
        K_old = np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
        P_old = Q + Ad.T @ P @ (Ad - Bd @ K_old)

        for i in range(num_itr):
            K_new = np.linalg.inv(R + Bd.T @ P_old @ Bd) @ Bd.T @ P_old @ Ad
            P_new = Q + Ad.T @ P_old @ (Ad - Bd @ K_new)
            if np.linalg.norm(K_new - K_old) < 1e-9:
                print("Infinite horizon LQR converged at iteration ", i)
                print("LQR Gain: \n", K_new)
                # print("Kx_wx: ", K_new[1, 0], " Kr_wx: ", K_new[1, -4])
                # print("Ky_wy: ", K_new[0, 1], " Ks_wy: ", K_new[0, -3])
                return K_new
            else:
                K_old = K_new
                P_old = P_new

        print("LQR did not converge")


if __name__ == "__main__":
    
    ## Check that the LQR is working
    Q = 1.0 * np.diag([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0])
    R = 1.0 * np.diag([1, 1, 1, 1])

    pos_tracker = ConstantPositionTracker(1.0, Q, R, 0.02)
    K = pos_tracker.infinite_horizon_LQR()