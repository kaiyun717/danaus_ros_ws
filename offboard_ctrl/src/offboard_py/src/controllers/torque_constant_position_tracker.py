"""
Script for running the ETH constant position tracking controller.
"""
#! /usr/bin/env python
import IPython
import numpy as np


class TorqueConstantPositionTracker:
    def __init__(self, K_inf, xyz_goal, gains_dict_px4, dt) -> None:
        self.K_inf = K_inf

        self.J_inv = np.array([
			[305.7518,  -0.6651,  -5.3547],
			[ -0.6651, 312.6261,  -3.1916],
			[ -5.3547,  -3.1916, 188.9651]])

        # self.pitch_int = 0.0
        # self.roll_int = 0.0
        # self.yaw_int = 0.0
        # self.gains_dict_px4 = gains_dict_px4

        self.body_rate_accum = np.zeros((3,1))
        
        self.dt = dt    # Time step
        self.g = 9.81   # Gravity
        self.nx = self.K_inf.shape[1]    # Number of states
        self.nu = self.K_inf.shape[0]     # Number of inputs

        self.xgoal = (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, xyz_goal[0], xyz_goal[1], xyz_goal[2], 0, 0, 0]))   

        self.xgoal = self.xgoal.reshape((self.nx, 1))
        self.ugoal = (np.array([self.g, 0, 0, 0]))
        self.ugoal = self.ugoal.reshape((self.nu, 1))
    
    def torque_body_rate_inputs(self, x):
        """
        Calculate the body rate inputs from the torques.
        State, x = [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, x, y, z, x_dot, y_dot, z_dot]
        """
        u = self.ugoal - self.K_inf @ (x - self.xgoal)

        torques = u[1:4]
        body_rates = self.J_inv @ torques * self.dt
        
        # px = self.gains_dict_px4["MC_ROLLRATE_P"] * body_rates[0]
        # py = self.gains_dict_px4["MC_PITCHRATE_P"] * body_rates[1]
        # pz = self.gains_dict_px4["MC_YAWRATE_P"] * body_rates[2]

        # self.roll_int += self.gains_dict_px4["MC_ROLLRATE_I"] * body_rates[0] * self.dt
        # self.pitch_int += self.gains_dict_px4["MC_PITCHRATE_I"] * body_rates[1] * self.dt
        # self.yaw_int += self.gains_dict_px4["MC_YAWRATE_I"] * body_rates[2] * self.dt

        # ix = self.roll_int
        # iy = self.pitch_int
        # iz = self.yaw_int
        
        # dx = self.gains_dict_px4["MC_ROLLRATE_D"] * body_rates[0]/self.dt
        # dy = self.gains_dict_px4["MC_PITCHRATE_D"] * body_rates[1]/self.dt
        # dz = self.gains_dict_px4["MC_YAWRATE_D"] * body_rates[2]/self.dt
        # return np.array([u[0], px + ix + dx, py + iy + dy, pz + iz + dz])

        self.body_rate_accum = self.body_rate_accum * 0.5 + body_rates * 0.5
        u_body = np.array([u[0], self.body_rate_accum[0], self.body_rate_accum[1], self.body_rate_accum[2]])
        return u_body, u
    
    def torque_inputs(self, x):
        """
        Calculate the body rate inputs from the torques.
        State, x = [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, x, y, z, x_dot, y_dot, z_dot]
        """
        u = self.ugoal - self.K_inf @ (x - self.xgoal)

        return u
    
    def torque_to_body_rate(self, u):
        torques = u[1:4].reshape((3,1))
        body_rates = self.J_inv @ torques * self.dt
        
        # px = self.gains_dict_px4["MC_ROLLRATE_P"] * body_rates[0]
        # py = self.gains_dict_px4["MC_PITCHRATE_P"] * body_rates[1]
        # pz = self.gains_dict_px4["MC_YAWRATE_P"] * body_rates[2]

        # self.roll_int += self.gains_dict_px4["MC_ROLLRATE_I"] * body_rates[0] * self.dt
        # self.pitch_int += self.gains_dict_px4["MC_PITCHRATE_I"] * body_rates[1] * self.dt
        # self.yaw_int += self.gains_dict_px4["MC_YAWRATE_I"] * body_rates[2] * self.dt

        # ix = self.roll_int
        # iy = self.pitch_int
        # iz = self.yaw_int
        
        # dx = self.gains_dict_px4["MC_ROLLRATE_D"] * body_rates[0]/self.dt
        # dy = self.gains_dict_px4["MC_PITCHRATE_D"] * body_rates[1]/self.dt
        # dz = self.gains_dict_px4["MC_YAWRATE_D"] * body_rates[2]/self.dt
        # return np.array([u[0], px + ix + dx, py + iy + dy, pz + iz + dz])
        self.body_rate_accum = self.body_rate_accum * 0.5 + body_rates * 0.5
        # IPython.embed()
        try:
            u = np.array([u[0].flatten(), self.body_rate_accum[0], self.body_rate_accum[1], self.body_rate_accum[2]])
        except:
            IPython.embed()
        return u
