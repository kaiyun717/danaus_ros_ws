import numpy as np
import IPython
import sys, os


class FlyingInvertedPendulumEnv:
    def __init__(self, 
                 dt,
                 model_param_dict, 
                 real_param_dict=None, 
                 dynamics_noise_spread=0.0):
    
        self.model_param_dict = model_param_dict

        self.__dict__.update(self.model_param_dict)  # __dict__ holds and object's attributes
        
        assert real_param_dict is None or dynamics_noise_spread == 0.0
        self.real_param_dict = real_param_dict
        self.dynamics_noise_spread = dynamics_noise_spread
        
        state_index_names = [
            "gamma", "beta", "alpha", 
            "dgamma", "dbeta", "dalpha", 
            "phi", "theta", 
            "dphi", "dtheta", 
            "x", "y", "z",                  # Including the translational positions
            "dx", "dy", "dz"                # Including the translational velocities
        ]
        state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))
        self.i = state_index_dict
        self.dt = dt
        # self.dt = 1e-6
        self.g = 9.81

        # self.init_visualization() # TODO: simplifying out viz, for now

        self.control_lim_verts = self.compute_control_lim_vertices()
        
    def compute_control_lim_vertices(self):
        angle = self.angle
        r1 = self.r1
        r2 = self.r2
        m_s = self.m_s
        max_thrust = self.max_thrust
        min_thrust = self.min_thrust

        M = np.array([
            [1, 1, 1, 1], 
            [-r1*np.sin(angle), r2*np.sin(angle), r1*np.sin(angle), -r2*np.sin(angle)],
            [-r1*np.cos(angle), r2*np.cos(angle), -r1*np.cos(angle), r2*np.cos(angle)],
            [-m_s, -m_s, m_s, m_s]]) # mixer matrix
        M = M * max_thrust		# normalize to max thrust so that it has max_thrust when v = 1
        
        self.mixer = M
        self.mixer_inv = np.linalg.inv(self.mixer)

        r1 = np.concatenate((np.zeros(8), np.ones(8)))
        r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
        r3 = np.concatenate((np.zeros(2), np.ones(2),np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),np.zeros(2), np.ones(2)))
        r4 = np.zeros(16)
        r4[1::2] = 1.0
        impulse_vert = np.concatenate((r1[None], r2[None], r3[None], r4[None]), axis=0) # 16 vertices in the impulse control space

        force_vert = M@impulse_vert - np.array([[self.M*self.g], [0.0], [0.0], [0.0]]) # Fixed bug: was subtracting self.M*g (not just in the first row)
        force_vert = force_vert.T.astype("float32")
        return force_vert

    def _f_model(self, x):

        # IPython.embed()

        # if len(x.shape) == 1:
        #     x = x[None] # (1, 16)
        # # print("Inside f")
        # # IPython.embed()
        # bs = x.shape[0]

        gamma = x[:, self.i["gamma"]]
        beta = x[:, self.i["beta"]]
        alpha = x[:, self.i["alpha"]]

        phi = x[:, self.i["phi"]]
        theta = x[:, self.i["theta"]]
        dphi = x[:, self.i["dphi"]]
        dtheta = x[:, self.i["dtheta"]]

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_alpha = np.sin(alpha)
        sin_beta = np.sin(beta)
        sin_gamma = np.sin(gamma)

        R = np.zeros((3, 3))
        R[0, 0] = cos_alpha*cos_beta
        R[0, 1] = cos_alpha*sin_beta*sin_gamma - sin_alpha*cos_gamma
        R[0, 2] = cos_alpha*sin_beta*cos_gamma + sin_alpha*sin_gamma
        R[1, 0] = sin_alpha*cos_beta
        R[1, 1] = sin_alpha*sin_beta*sin_gamma + cos_alpha*cos_gamma
        R[1, 2] = sin_alpha*sin_beta*cos_gamma - cos_alpha*sin_gamma
        R[2, 0] = -sin_beta
        R[2, 1] = cos_beta*sin_gamma
        R[2, 2] = cos_beta*cos_gamma

        # R = np.zeros((3, 3))
        # R[0, 0] = np.cos(alpha)*np.cos(beta)
        # R[0, 1] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
        # R[0, 2] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
        # R[1, 0] = np.sin(alpha)*np.cos(beta)
        # R[1, 1] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
        # R[1, 2] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
        # R[2, 0] = -np.sin(beta)
        # R[2, 1] = np.cos(beta)*np.sin(gamma)
        # R[2, 2] = np.cos(beta)*np.cos(gamma)

        k_x = R[0, 2]
        k_y = R[1, 2]
        k_z = R[2, 2]

        ###### Computing state derivatives
        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        sin_theta = np.sin(theta)

        ddphi = (3.0) * (k_y * cos_phi + k_z * sin_phi) * (self.M * self.g) / (
                    2 * self.M * self.L_p * cos_theta) + 2 * dtheta * dphi * np.tan(theta)
        
        ddtheta = (3.0*(-k_x*cos_theta-k_y*sin_phi*sin_theta + k_z*cos_phi*sin_theta)*(self.M*self.g)/(2.0*self.M*self.L_p)) \
                - np.square(dphi)*sin_theta*cos_theta

        # ddphi = (3.0) * (k_y * np.cos(phi) + k_z * np.sin(phi)) * (self.M * self.g) / (
        #             2 * self.M * self.L_p * np.cos(theta)) + 2 * dtheta * dphi * np.tan(theta)

        # ddtheta = (3.0*(-k_x*np.cos(theta)-k_y*np.sin(phi)*np.sin(theta) + k_z*np.cos(phi)*np.sin(theta))*(self.M*self.g)/(2.0*self.M*self.L_p)) \
        #         - np.square(dphi)*np.sin(theta)*np.cos(theta)

        ddx = k_x*self.g
        ddy = k_y*self.g
        ddz = k_z*self.g - self.g

        # Including translational motion
        f = np.vstack([x[:,self.i["dgamma"]], 
                       x[:,self.i["dbeta"]], 
                       x[:,self.i["dalpha"]], 
                       0, 
                       0, 
                       0, 
                       dphi, 
                       dtheta, 
                       ddphi, 
                       ddtheta, 
                       x[:,self.i["dx"]], 
                       x[:,self.i["dy"]], 
                       x[:,self.i["dz"]], 
                       ddx, 
                       ddy, 
                       ddz]).T
        return f.reshape((16,1))

    def _g_model(self, x):
        # if len(x.shape) == 1:
        #     x = x[None] # (1, 16)
        # # print("g: returns matrix")
        # # IPython.embed()
        # bs = x.shape[0]

        gamma = x[:,self.i["gamma"]]
        beta = x[:,self.i["beta"]]
        alpha = x[:,self.i["alpha"]]

        phi = x[:,self.i["phi"]]
        theta = x[:,self.i["theta"]]

        cos_alpha = np.cos(alpha)
        cos_beta = np.cos(beta)
        cos_gamma = np.cos(gamma)
        sin_alpha = np.sin(alpha)
        sin_beta = np.sin(beta)
        sin_gamma = np.sin(gamma)

        R = np.zeros((3, 3))
        R[0, 0] = cos_alpha*cos_beta
        R[0, 1] = cos_alpha*sin_beta*sin_gamma - sin_alpha*cos_gamma
        R[0, 2] = cos_alpha*sin_beta*cos_gamma + sin_alpha*sin_gamma
        R[1, 0] = sin_alpha*cos_beta
        R[1, 1] = sin_alpha*sin_beta*sin_gamma + cos_alpha*cos_gamma
        R[1, 2] = sin_alpha*sin_beta*cos_gamma - cos_alpha*sin_gamma
        R[2, 0] = -sin_beta
        R[2, 1] = cos_beta*sin_gamma
        R[2, 2] = cos_beta*cos_gamma

        # R = np.zeros((3, 3))
        # R[0, 0] = np.cos(alpha)*np.cos(beta)
        # R[0, 1] = np.cos(alpha)*np.sin(beta)*np.sin(gamma) - np.sin(alpha)*np.cos(gamma)
        # R[0, 2] = np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma)
        # R[1, 0] = np.sin(alpha)*np.cos(beta)
        # R[1, 1] = np.sin(alpha)*np.sin(beta)*np.sin(gamma) + np.cos(alpha)*np.cos(gamma)
        # R[1, 2] = np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma)
        # R[2, 0] = -np.sin(beta)
        # R[2, 1] = np.cos(beta)*np.sin(gamma)
        # R[2, 2] = np.cos(beta)*np.cos(gamma)

        k_x = R[0, 2]
        k_y = R[1, 2]
        k_z = R[2, 2]

        ###### Computing state derivatives
        # J_inv = np.diag([(1.0/self.J_x), (1.0/self.J_y), (1.0/self.J_z)])
        # J = np.array([
		# 	[self.J_xx, self.J_xy, self.J_xz],
		# 	[self.J_xy, self.J_yy, self.J_yz],
		# 	[self.J_xz, self.J_yz, self.J_zz]]).to(self.device)
		# norm_torques = u[:, 1:]@torch.inverse(J)
        J_inv = np.array([
            [305.7518,  -0.6651,  -5.3547],
            [ -0.6651, 312.6261,  -3.1916],
            [ -5.3547,  -3.1916, 188.9651]])
        
        dd_drone_angles = R@J_inv

        # print(J_inv, R)

        cos_phi = np.cos(phi)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        sin_theta = np.sin(theta)

        ddphi = (3.0) * (k_y * cos_phi + k_z * sin_phi) / (2 * self.M * self.L_p * cos_theta)
        ddtheta = (3.0*(-k_x*cos_theta-k_y*sin_phi*sin_theta + k_z*cos_phi*sin_theta)/(2.0*self.M*self.L_p))

        # ddphi = (3.0)*(k_y*np.cos(phi) + k_z*np.sin(phi))/(2*self.M*self.L_p*np.cos(theta))
        # ddtheta = (3.0*(-k_x*np.cos(theta)-k_y*np.sin(phi)*np.sin(theta) + k_z*np.cos(phi)*np.sin(theta))/(2.0*self.M*self.L_p))

        # Including translational motion
        g = np.zeros((16, 4))
        g[3:6, 1:] = dd_drone_angles
        g[8, 0] = ddphi
        g[9, 0] = ddtheta
        g[13:, 0] = (1.0/self.M)*np.array([k_x, k_y, k_z]).T

        # print(g)
        return g

    def x_dot_open_loop_model(self, x, u):
        # if u.ndim == 1:
        #     u = u[None]
        # if x.ndim == 1:
        #     x = x[None]

        # Batched
        f = self._f_model(x)
        g = self._g_model(x)
        # IPython.embed()

        # u_clamped, debug_dict = self.clip_u(u)
        # print("in x_dot_open_loop")
        # IPython.embed()
        # rv = f + (g@(u_clamped[:, :, None]))[:, :, 0]
        rv = f + g@u
        # rv = np.squeeze(rv)
        return rv

    def rk4_x_dot_open_loop_model(self, x, u):
        k1 = (self.dt * self.x_dot_open_loop_model(x, u)).reshape((1,-1))
        k2 = (self.dt * self.x_dot_open_loop_model(x + k1/2, u)).reshape((1,-1))
        k3 = (self.dt * self.x_dot_open_loop_model(x + k2/2, u)).reshape((1,-1))
        k4 = (self.dt * self.x_dot_open_loop_model(x + k3, u)).reshape((1,-1))
        return x + (k1 + 2*k2 + 2*k3 + k4)/6
        # return self.dt * self.x_dot_open_loop_model(x, u)

    def x_dot_open_loop(self, x, u):
        # TODO: this is very hacky
        if self.real_param_dict is not None:
            # print("applying model mismatch in env, ln 177")
            # IPython.embed()
            self.__dict__.update(self.real_param_dict)
            # print("Changed to real params:", self.__dict__)
            rv = self.x_dot_open_loop_model(x, u)
            self.__dict__.update(self.model_param_dict)
            # print("Changed back to model params:", self.__dict__)
        elif self.dynamics_noise_spread != 0:
            # print("applying stochastic dynamics in env, ln 177")
            # IPython.embed()
            x_dot = self.x_dot_open_loop_model(x, u)
            rv = x_dot + np.random.normal(scale=self.dynamics_noise_spread, size=x_dot.shape)
        else:
            rv = self.x_dot_open_loop_model(x, u)
        return rv

    def _smooth_clamp(self, motor_impulses):
        # Batched
        # clamps to 0, 1
        # rv = 1.0/(1.0 + np.exp(-8*motor_impulses+4))
        rv = np.clip(motor_impulses, 0, 1)
        return rv

    def clip_u(self, u):
        # if u.ndim == 1:
        #     u = u[None]
        # Batched: u is bs x u_dim
        # Assumes u is raw
        # print("in clip_u")
        # IPython.embed()
        
        u_gravity_comp = u + np.array([self.M*self.g, 0, 0, 0]) # u with gravity compensation
        # motor_impulses = np.linalg.solve(self.mixer, u_gravity_comp) # low-level inputs
        motor_impulses = u_gravity_comp@self.mixer_inv.T
        smooth_clamped_motor_impulses = self._smooth_clamp(motor_impulses)

        smooth_clamped_u_gravity_comp = smooth_clamped_motor_impulses@self.mixer.T
        rv = smooth_clamped_u_gravity_comp - np.array([self.M*self.g, 0, 0, 0])

        return rv, {"motor_impulses": motor_impulses, 
                    "smooth_clamped_motor_impulses": smooth_clamped_motor_impulses}
