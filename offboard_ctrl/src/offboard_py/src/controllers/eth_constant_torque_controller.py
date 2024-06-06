"""
Script for running the ETH constant position tracking controller.
"""
#! /usr/bin/env python

import numpy as np
import scipy
import IPython

g = 9.81

class ConstantPositionTorqueTracker:
    def __init__(self, cont_type, ang_vel_type, J, L, Q, R, xyz_goal, dt) -> None:
        self.cont_type = cont_type
        self.ang_vel_type = ang_vel_type
        self.J = J      # Inertia of the pendulum
        self.L = L      # length from the base of the pendulum to the center of mass
        self.Q = Q      # State cost matrix
        self.R = R      # Input cost matrix
        self.dt = dt    # Time step
        self.g = 9.81   # Gravity
        self.nx = 16    # Number of states
        self.nu = 4     # Number of inputs

        Jxx = self.J[0][0]
        Jyy = self.J[1][1]
        Jzz = self.J[2][2]
        Jxy = self.J[0][1]
        Jxz = self.J[0][2]
        Jyz = self.J[1][2]
        
        assert L == 0.5, "L should be 0.5 for this controller"

        if self.ang_vel_type == "euler":
            ### Original ###
            # self.Ad = np.array([[         1,          0, 0,       0.01,          0,    0, 0, 0, 0,    0,    0,    0,        0,        0,         0,         0],
            #                     [         0,          1, 0,          0,       0.01,    0, 0, 0, 0,    0,    0,    0,        0,        0,         0,         0],
            #                     [         0,          0, 1,          0,          0, 0.01, 0, 0, 0,    0,    0,    0,        0,        0,         0,         0],
            #                     [         0,          0, 0,          1,          0,    0, 0, 0, 0,    0,    0,    0,        0,        0,         0,         0],
            #                     [         0,          0, 0,          0,          1,    0, 0, 0, 0,    0,    0,    0,        0,        0,         0,         0],
            #                     [         0,          0, 0,          0,          0,    1, 0, 0, 0,    0,    0,    0,        0,        0,         0,         0],
            #                     [         0,  0.0004905, 0,          0,  1.635e-06,    0, 1, 0, 0, 0.01,    0,    0,        0,        0,         0,         0],
            #                     [-0.0004905,          0, 0, -1.635e-06,          0,    0, 0, 1, 0,    0, 0.01,    0,        0,        0,         0,         0],
            #                     [         0,          0, 0,          0,          0,    0, 0, 0, 1,    0,    0, 0.01,        0,        0,         0,         0],
            #                     [         0,     0.0981, 0,          0,  0.0004905,    0, 0, 0, 0,    1,    0,    0,        0,        0,         0,         0],
            #                     [   -0.0981,          0, 0, -0.0004905,          0,    0, 0, 0, 0,    0,    1,    0,        0,        0,         0,         0],
            #                     [         0,          0, 0,          0,          0,    0, 0, 0, 0,    0,    0,    1,        0,        0,         0,         0],
            #                     [         0, -0.00073584,0,          0, -2.4525e-06,   0, 0, 0, 0,    0,    0,    0,  1.00074,        0, 0.0100025,         0],
            #                     [-0.00073584,         0, 0, -2.4525e-06,         0,    0, 0, 0, 0,    0,    0,    0,        0,  1.00074,         0, 0.0100025],
            #                     [         0,  -0.147186, 0,          0, -0.00073584,   0, 0, 0, 0,    0,    0,    0, 0.147186,        0,   1.00074,         0],
            #                     [ -0.147186,          0, 0, -0.00073584,         0,    0, 0, 0, 0,    0,    0,    0,        0, 0.147186,         0,   1.00074]])
            # self.Bd = np.array([[     0,    0.0156079, -0.000629628, -0.000146218],
            #                     [     0, -0.000629628,     0.017349,  -0.00021998],
            #                     [     0, -0.000146218,  -0.00021998,    0.0101024],
            #                     [     0,      3.12159,    -0.125926,   -0.0292435],
            #                     [     0,    -0.125926,      3.46981,   -0.0439959],
            #                     [     0,   -0.0292435,   -0.0439959,      2.02049],
            #                     [     0, -5.14721e-08,  1.41828e-06, -1.79833e-08],
            #                     [     0, -1.27595e-06,  5.14721e-08,  1.19533e-08],
            #                     [ 5e-05,            0,            0,            0],
            #                     [     0, -2.05888e-05,  0.000567314, -7.19333e-06],
            #                     [     0,  -0.00051038,  2.05888e-05,  4.78132e-06],
            #                     [  0.01,            0,            0,            0],
            #                     [     0,  7.72082e-08, -2.12743e-06,   2.6975e-08],
            #                     [     0, -1.91392e-06,  7.72082e-08,    1.793e-08],
            #                     [     0,  3.08833e-05, -0.000850971,    1.079e-05],
            #                     [     0, -0.000765569,  3.08833e-05,  7.17198e-06]])
            ### Diff. version 1 ###
            self.Ac = np.array([[                     0,                      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,                     0,                     0, 0, 0],
                                [                     0,                 self.g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 0],
                                [               -self.g,                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 0],
                                [                     0,                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 1, 0],
                                [                     0,                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0,                     0, 0, 1],
                                [                     0, -(3*self.g)/(4*self.L), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (3*self.g)/(4*self.L),                     0, 0, 0],
                                [-(3*self.g)/(4*self.L),                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,                     0, (3*self.g)/(4*self.L), 0, 0]])
            self.Bc = np.array([[0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,    (Jyz**2 - Jyy*Jzz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz), -(Jxz*Jyz - Jxy*Jzz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz), -(Jxy*Jyz - Jxz*Jyy)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz)],
                                [0, -(Jxz*Jyz - Jxy*Jzz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz),    (Jxz**2 - Jxx*Jzz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz), -(Jxy*Jxz - Jxx*Jyz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz)],
                                [0, -(Jxy*Jyz - Jxz*Jyy)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz), -(Jxy*Jxz - Jxx*Jyz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz),    (Jxy**2 - Jxx*Jyy)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz)],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [1,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0],
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]])
        elif self.ang_vel_type == "body":
            self.Ac = np.array([[           0,            0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,           0,           0, 0, 0]
                                [           0,            g, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0,           0, 0, 0]
                                [          -g,            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0,           0, 0, 0]
                                [           0,            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0,           0, 1, 0]
                                [           0,            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0,           0, 0, 1]
                                [           0, -(3*g)/(4*L), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (3*g)/(4*L),           0, 0, 0]
                                [-(3*g)/(4*L),            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,           0, (3*g)/(4*L), 0, 0]])
            self.Bc = np.array([[0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,    (Jyz**2 - Jyy*Jzz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz), -(Jxz*Jyz - Jxy*Jzz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz), -(Jxy*Jyz - Jxz*Jyy)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz)]
                                [0, -(Jxz*Jyz - Jxy*Jzz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz),    (Jxz**2 - Jxx*Jzz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz), -(Jxy*Jxz - Jxx*Jyz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz)]
                                [0, -(Jxy*Jyz - Jxz*Jyy)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz), -(Jxy*Jxz - Jxx*Jyz)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz),    (Jxy**2 - Jxx*Jyy)/(Jzz*Jxy**2 - 2*Jxy*Jxz*Jyz + Jyy*Jxz**2 + Jxx*Jyz**2 - Jxx*Jyy*Jzz)]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [1,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]
                                [0,                                                                                      0,                                                                                      0,                                                                                      0]])


        exp_matrix = scipy.linalg.expm(np.block([[self.Ac, self.Bc], [np.zeros((self.nu, self.nx + self.nu))]]) * self.dt)
        self.Ad = exp_matrix[:self.nx, :self.nx]
        self.Bd = exp_matrix[:self.nx, self.nx:]

        self.xgoal = (np.array([0, 0, 0, 0, 0, 0, xyz_goal[0], xyz_goal[1], xyz_goal[2], 0, 0, 0, 0, 0, 0, 0])) 
        self.ugoal = (np.array([self.g, 0, 0, 0]))
        
        self.xgoal = self.xgoal.reshape((self.nx, 1))
        self.ugoal = self.ugoal.reshape((self.nu, 1))
    
    def infinite_horizon_LQR(self, num_itr=10000):
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

    pos_tracker = ConstantPositionTorqueTracker("tp", 0.5, Q, R, np.zeros((3,1)), 0.02)
    # K = pos_tracker.infinite_horizon_LQR()