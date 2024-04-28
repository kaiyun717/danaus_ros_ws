"""
Script for running the ETH constant position tracking controller.
"""
#! /usr/bin/env python

import numpy as np
import scipy


class ConstantPositionTracker:
    def __init__(self, L, Q, R, xyz_goal, dt) -> None:
        self.L = L      # length from the base of the pendulum to the center of mass
        self.Q = Q      # State cost matrix
        self.R = R      # Input cost matrix
        self.dt = dt    # Time step
        self.g = 9.81   # Gravity
        self.nx = 12    # Number of states
        self.nu = 4     # Number of inputs

        # self.Ac = np.array([[0, 0, 0, 1, 0, 0, 0,  0,       0,       0,             0,             0, 0],
        #                     [0, 0, 0, 0, 1, 0, 0,  0,       0,       0,             0,             0, 0],
        #                     [0, 0, 0, 0, 0, 1, 0,  0,       0,       0,             0,             0, 0],
        #                     [0, 0, 0, 0, 0, 0, 0,  self.g,  0,       0,             0,             0, 0],
        #                     [0, 0, 0, 0, 0, 0, 0,  0,       -self.g, 0,             0,             0, 0],
        #                     [0, 0, 0, 0, 0, 0, 0,  0,       0,       0,             0,             0, 0],
        #                     [0, 0, 0, 0, 0, 0, 0,  0,       0,       0,             0,             0, 0],
        #                     [0, 0, 0, 0, 0, 0, 0,  0,       0,       0,             0,             0, 0],
        #                     [0, 0, 0, 0, 0, 0, 0,  0,       0,       0,             0,             0, 0],
        #                     [0, 0, 0, 0, 0, 0, 0,  0,       0,       0,             0,             1, 0],
        #                     [0, 0, 0, 0, 0, 0, 0,  0,       0,       0,             0,             0, 1],
        #                     [0, 0, 0, 0, 0, 0, 0, -self.g,  0,       self.g/self.L, 0,             0, 0],
        #                     [0, 0, 0, 0, 0, 0, 0,  0,       self.g,  0,             self.g/self.L, 0, 0]])
        # self.Bc = np.array([[0, 0, 0, 0],
        #                     [0, 0, 0, 0],
        #                     [0, 0, 0, 0],
        #                     [0, 0, 0, 0],
        #                     [0, 0, 0, 0],
        #                     [0, 0, 0, 1],
        #                     [0, 0, 1, 0],
        #                     [0, 1, 0, 0],
        #                     [1, 0, 0, 0],
        #                     [0, 0, 0, 0],
        #                     [0, 0, 0, 0],
        #                     [0, 0, 0, 0],
        #                     [0, 0, 0, 0]])
        
        # exp_matrix = scipy.linalg.expm(np.block([[self.Ac, self.Bc], [np.zeros((self.nu, self.nx + self.nu))]]) * self.dt)
        # self.Ad = exp_matrix[:self.nx, :self.nx]
        # self.Bd = exp_matrix[:self.nx, self.nx:]
        
        self.Ad = np.array([
            [1.0, 0.0, 0.0, 0.02, 0.0,  0.0, 0.0, 0.001962, 0.0, 0.0, 1.308e-5, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.02,  0.0, 0.0, 0.0, -0.001962, 0.0, 0.0, -1.308e-5],
            [0.0, 0.0, 1.0, 0.0,  0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0,  0.0,  0.0, 0.0, 0.19619999999999999, 0.0, 0.0, 0.001962, 0.0],
            [0.0, 0.0, 0.0, 0.0,  1.0,  0.0, 0.0, 0.0, -0.19619999999999999, 0.0, 0.0, -0.001962],
            [0.0, 0.0, 0.0, 0.0,  0.0,  1.0, 0.0, 0.0, 0.0, 0.0, 0.0 , 0.0],
            [0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 1.0, 0.0, 0.0, 0.02, 0.0,  0.0],
            [0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 1.0, 0.0, 0.0, 0.02,  0.0],
            [0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0, 1.0, 0.0, 0.0 , 0.02],
            [0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 1.0, 0.0 , 0.0],
            [0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 1.0 , 0.0],
            [0.0, 0.0, 0.0, 0.0,  0.0,  0.0, 0.0, 0.0, 0.0, 0.0, 0.0 , 1.0]
        ])
        
        self.Bd = np.array([
            [0.0        , 2.04421e-5,  0.0      ,  0.0],
            [-1.99861e-5,  0.0      ,   0.0     ,   0.0],
            [0.0        , 0.0       ,  0.0      ,  0.0002],
            [0.0        , 0.00408842,  0.0      ,  0.0],
            [-0.00399723,  0.0      ,   0.0     ,   0.0],
            [0.0        , 0.0       ,  0.0      ,  0.02],
            [0.0        , 0.0       ,  0.0377677,  0.0],
            [0.0        , 0.0625141 ,  0.0      ,  0.0],
            [0.0611197  , 0.0       ,  0.0      ,  0.0],
            [0.0        , 0.0       ,  3.77677  ,  0.0],
            [0.0        , 6.25141   ,  0.0      ,  0.0],
            [6.11197    , 0.0       ,  0.0      ,  0.0],
        ])

        # self.Ad = np.array([
        #     [1.0, 0.0, 0.0, 0.01,  0.0,  0.0],
        #     [0.0, 1.0, 0.0, 0.0 , 0.01,  0.0],
        #     [0.0, 0.0, 1.0, 0.0 , 0.0 , 0.01],
        #     [0.0, 0.0, 0.0, 1.0 , 0.0 , 0.0],
        #     [0.0, 0.0, 0.0, 0.0 , 1.0 , 0.0],
        #     [0.0, 0.0, 0.0, 0.0 , 0.0 , 1.0]
        # ])

        # self.Bd = np.array([
        #     [0.0, 0.0, 0.009441925548528666, 0.0],
        #     [0.0, 0.015628516416193644, 0.0, 0.0],
        #     [0.015279912721138537, 0.0, 0.0, 0.0],
        #     [0.0, 0.0, 1.8883851097057331, 0.0],
        #     [0.0, 3.1257032832387286, 0.0, 0.0],
        #     [3.0559825442277075, 0.0, 0.0, 0.0]
        # ])

        self.xgoal = (np.array([xyz_goal[0], xyz_goal[1], xyz_goal[2], 0, 0, 0, 0, 0, 0, 0, 0, 0]))   
        # self.xgoal = (np.array([0, 0, 0, 0, 0, 0]))   
        # Goal: (1, 1, 1) --> # Pos Pitch (wy), Neg Roll (wx), Pos Thr --> No, No, Yes
        # Goal: (-1, -1, -1) --> # Neg Pitch (wy), Pos Roll (wx), Neg Thr --> No, No, Yes

        self.xgoal = self.xgoal.reshape((self.nx, 1))
        self.ugoal = (np.array([0, 0, 0, self.g]))
        self.ugoal = self.ugoal.reshape((self.nu, 1))
    
    def infinite_horizon_LQR(self, num_itr=10000):
        print(f"{self.Q.shape=}, {self.R.shape=}, {self.Ad.shape=}, {self.Bd.shape=}")
        P = np.copy(self.Q)
        K_old = np.linalg.inv(self.R + self.Bd.T @ P @ self.Bd) @ self.Bd.T @ P @ self.Ad
        P_old = self.Q + self.Ad.T @ P @ (self.Ad - self.Bd @ K_old)

        for i in range(num_itr):
            K_new = np.linalg.inv(self.R + self.Bd.T @ P_old @ self.Bd) @ self.Bd.T @ P_old @ self.Ad
            P_new = self.Q + self.Ad.T @ P_old @ (self.Ad - self.Bd @ K_new)
            if np.linalg.norm(K_new - K_old) < 1e-9:
                print("Infinite horizon LQR converged at iteration ", i)
                print("LQR Gain: \n", K_new)
                print("Kx_wx: ", K_new[1, 0], " Kr_wx: ", K_new[1, -4])
                print("Ky_wy: ", K_new[0, 1], " Ks_wy: ", K_new[0, -3])
                print("Krd_wx: ", K_new[1, -2], " Ksd_wy: ", K_new[0, -1])
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