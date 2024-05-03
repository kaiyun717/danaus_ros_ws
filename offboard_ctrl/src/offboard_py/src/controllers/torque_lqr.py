import IPython
import numpy as np
import scipy


class TorqueLQR:
    def __init__(self, L, Q, R, xyz_goal, dt, type="with_pend", K_inf=None, num_itr=10000) -> None:
        self.L = L      # length from the base of the pendulum to the center of mass
        self.Q = Q      # State cost matrix
        self.R = R      # Input cost matrix
        self.dt = dt    # Time step
        self.g = 9.81
        self.num_itr = num_itr
        self.xyz_goal = xyz_goal
        
        self.body_rate_accum = np.zeros((3,1))
        
        self.J_inv = np.array([
			[305.7518,  -0.6651,  -5.3547],
			[ -0.6651, 312.6261,  -3.1916],
			[ -5.3547,  -3.1916, 188.9651]])
        
        self.xgoal = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.xyz_goal[0], self.xyz_goal[1], self.xyz_goal[2], 0, 0, 0]).reshape((16, 1))
        self.ugoal = np.array([self.g, 0, 0, 0]).reshape((4, 1))
        
        if type == "with_pend":
            self.nx = 16
            self.nu = 4
            if K_inf is None:
                self.K_inf = self._infinite_horizon_LQR(num_itr)
                # IPython.embed()
            else:
                self.K_inf = K_inf
        else:   # type == "without_pend"
            self.nx = 12
            self.nu = 4
            if K_inf is None:
                K_inf = np.array([
                            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.990423659437537, 0.0, 0.0, 1.7209841208008194],
                            [1.4814681367499676, 0.002995481626744845, 0.005889427337762064, 0.28201313626232954, 0.0005738532722789398, 0.005981686140109121, 0.0005315334025884319, -0.2644839556489373, 0.0, 0.000779188657396088, -0.3870845425896414, 0.0],
                            [0.002995437953182237, 1.4528645588948705, 0.0034553424254151212, 0.0005738451434393372, 0.27654055987779025, 0.003509251033197913, 0.2594021552915394, -0.0005315255954541553, 0.0, 0.3796374382180433, -0.0007791772254469898, 0.0],
                            [0.03160126871064939, 0.018545750101093363, 0.39799289325860154, 0.006079664965265176, 0.003567296347199587, 0.4032536566450523, 0.003278753893103592, -0.005585886845822208, 0.0, 0.004811104113305738, -0.008196880671368846, 0.0]
                        ])
            else:
                K_inf = K_inf
        
    def _infinite_horizon_LQR(self, num_itr):
        A = np.zeros((16, 16))  # 16 x 16
        ######## Pendulum Control ########
        A[0:3, 3:6] = np.eye(3)
        A[6:8, 8:10] = np.eye(2)
        A[8, 0] = -3 * self.g / (2 * self.L)
        A[9, 1] = -3 * self.g / (2 * self.L)
        A[8, 6] = 3 * self.g / (2 * self.L)
        A[9, 7] = 3 * self.g / (2 * self.L)
        ######## Position Control ########
        A[13,1] = self.g
        A[14,0] = -self.g
        A[10:13, 13:16] = np.eye(3)

        B = np.zeros((16, 4))			
        ######## Pendulum Control ########
        B[3:6, 1:4] = self.J_inv
        ######## Position Control ########
        B[15,0] = 1		# TODO: need mass division???

        exp_matrix = scipy.linalg.expm(np.block([[A, B], [np.zeros((4, 16 + 4))]]) * self.dt)
        Ad = exp_matrix[:16, :16]
        Bd = exp_matrix[:16, 16:]

        P = np.copy(self.Q)
        K_old = np.linalg.inv(self.R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad
        P_old = self.Q + Ad.T @ P @ (Ad - Bd @ K_old)

        for i in range(num_itr):
            K_new = np.linalg.inv(self.R + Bd.T @ P_old @ Bd) @ Bd.T @ P_old @ Ad
            P_new = self.Q + Ad.T @ P_old @ (Ad - Bd @ K_new)
            if np.linalg.norm(K_new - K_old) < 1e-9:
                print("Infinite horizon LQR converged at iteration ", i)
                print("LQR Gain: \n", K_new)
                return K_new
            else:
                K_old = K_new
                P_old = P_new
        print("LQR did not converge")

    def torque_body_rate_inputs(self, x):
        """
        Calculate the body rate inputs from the torques.
        State, x = [roll, pitch, yaw, roll_rate, pitch_rate, yaw_rate, x, y, z, x_dot, y_dot, z_dot]
        """
        u = self.ugoal - self.K_inf @ (x - self.xgoal)

        torques = u[1:4]
        body_rates = self.J_inv @ torques * self.dt
        
        self.body_rate_accum = self.body_rate_accum * 0.9 + body_rates * 0.1
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
        
        self.body_rate_accum = self.body_rate_accum * 0.9 + body_rates * 0.1
        try:
            u = np.array([u[0].flatten(), self.body_rate_accum[0], self.body_rate_accum[1], self.body_rate_accum[2]])
        except:
            IPython.embed()
        return u


if __name__ == "__main__":
    Q = np.eye(16)
    R = np.eye(4)
    L = 0.69
    xyz_goal = np.array([0, 0, 1.5])
    dt = 1/90
    tracker = TorqueLQR(L, Q, R, xyz_goal, dt, type="with_pend")
    print(tracker.K_inf)
    