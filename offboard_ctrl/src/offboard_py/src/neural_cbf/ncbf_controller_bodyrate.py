import torch
import time 
import numpy as np
import math
from cvxopt import matrix, solvers

# from quadprog import solve_qp

solvers.options['show_progress'] = False
import IPython
import control

g = 9.81


class NCBFControllerBodyRate:
    def __init__(self, 
                    vehicle,
                    env, 
                    cbf_fn, 
                    param_dict, 
                    eps_bdry=1.0, 
                    eps_outside=5.0):
        super().__init__()
        variables = locals()  
        self.__dict__.update(variables)
        del self.__dict__["self"]
        self.__dict__.update(self.param_dict)

        ####################
        ### Mixer Matrix ###
        ####################
        angle = self.angle
        r1 = self.r1
        r2 = self.r2
        m_s = self.m_s
        max_thrust = self.max_thrust
        min_thrust = self.min_thrust

        mixer = np.array([
            [1, 1, 1, 1], 
            [-r1*np.sin(angle), r2*np.sin(angle), r1*np.sin(angle), -r2*np.sin(angle)],
            [-r1*np.cos(angle), r2*np.cos(angle), -r1*np.cos(angle), r2*np.cos(angle)],
            [-m_s, -m_s, m_s, m_s]]) # mixer matrix
        self.mixer = mixer * max_thrust		# Normalize to max thrust so that it has max_thrust when v = 1.
        self.mixer_inv = np.array([     # Not used, only if you want to give initial impulses for QP.
            [ 0.06300014, -0.70380981, -0.92851892, -3.4426304 ],
            [ 0.06199986,  0.70380981,  0.92851892, -3.3879707 ],
            [ 0.06300014,  0.70380981, -0.92851892,  3.4426304 ],
            [ 0.06199986, -0.70380981,  0.92851892,  3.3879707 ]])
        
        if vehicle == 'danaus12_old':
            self.J = np.array([	# NOTE: danaus12_old
                [0.00320868, 0.00011707,  0.00004899],
                [0.00011707, 0.00288707,  0.00006456],
                [0.00004899, 0.00006456,  0.00495141]])
        else:
            NotImplementedError(f"{vehicle} not implemented in NCBFController")
        
        self.dt = self.env.dt
        self.control_conversion = np.vstack((
                            np.array([1, 0, 0, 0]),
                            np.hstack((np.zeros((3,1)), self.J/self.dt))))
        self.control_conversion_inv = np.linalg.inv(self.control_conversion)

        #############################
        ### Mass of the Quadrotor ###
        #############################
        self.M = self.M
        print(f"Mass of quadrotor in controller: {self.M}")

    def compute_control(self, x, u_ref):
        u_ref_old = np.copy(u_ref)
        
        u_ref[0] = (u_ref[0] - g) * self.M 
        x = np.reshape(x, (1, -1))
        
        ####################################
        ### Determine boundary condition ###
        ####################################
        phi_vals = self.cbf_fn.phi_fn(x)  # This is an array of (1, r+1), where r is the degree
        x_next = self.env.rk4_x_dot_open_loop_model(x, u_ref)    # RK4 Implementation
        next_phi_val = self.cbf_fn.phi_fn(x_next)

        ### Outside ###
        if phi_vals[0, -1] > 1e-2:  # Outside
            print("STATUS: Outside") # TODO
            eps = self.eps_outside
            stat = 0
        ### On boundary ###
        elif phi_vals[0, -1] < 0 and next_phi_val[0, -1] >= 0:  # On boundary. Note: cheating way to convert DT to CT
            print("STATUS: On") # TODO
            eps = self.eps_bdry
            stat = 1
        # elif phi_vals[0, -1] < 1e-3 and phi_vals[0, -1] > -1e-3:  # On boundary. Note: cheating way to convert DT to CT
        #     print("STATUS: On") # TODO
        #     eps = self.eps_bdry
        #     stat = 1
        ### Inside ###
        else:
            print("STATUS: Inside") # TODO
            stat = 2
            return u_ref_old, stat, phi_vals[0, -1]

        ### Control constraints ###
        f_x = self.env._f_model(x)
        f_x = np.reshape(f_x, (16, 1))
        g_x = self.env._g_model(x)
        phi_grad = self.cbf_fn.phi_grad(x)
        phi_grad = np.reshape(phi_grad, (16, 1))
        lhs = phi_grad.T @ g_x  # 1 x 4
        rhs = -phi_grad.T @ f_x - eps
        rhs = rhs.item()  # scalar, not numpy array

        ### QP Safety Filter ###
        # Note, constraint may not always be satisfied, so we include a slack variable on the CBF input constraint
        w = 1000.0  # slack weight

        P = np.zeros((9, 9))
        P[:4, :4] = 2 *np.eye(4)
        q = np.zeros((9, 1))
        q[:4, 0] = -2*u_ref.flatten()
        q[-1, 0] = w

        # G <= h
        G = np.zeros((10,9))
        G[0, :4] = lhs @ self.control_conversion
        G[0, -1] = -1.0
        G[1:5, 4:8] = -np.eye(4)
        G[5:9, 4:8] = np.eye(4)
        G[9, -1] = -1.0

        h = np.zeros((10, 1))
        h[0, 0] = rhs
        h[5:9, 0] = 1.0

        A = np.zeros((4, 9))
        A[:4, :4] = -self.control_conversion
        A[:4, 4:8] = self.mixer
        b = np.array([self.M*g, 0, 0, 0])[:, None]

        try:
            # init_impulses = self.mixer_inv @ (u_ref - np.array([self.M*g, 0, 0, 0])[:, None])
            # initvals = {"x": matrix(np.concatenate((u_ref.flatten(), init_impulses.flatten(), np.array([0]))))}
            sol_obj = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))#, initvals=initvals)
        except:
            # IPython.embed()
            print("QP solve was unsuccessful, with status: %s " % sol_obj["status"])
            print("Go to line 96 in flying_cbf_controller")
            IPython.embed()
            print("exiting")
            exit(0)

        sol_var = np.array(sol_obj['x'])

        u_safe = sol_var[0:4]
        u_safe[0] = u_safe[0]/self.M + g
        u_safe = np.reshape(u_safe, (4))
        
        return u_safe, stat, phi_vals[0, -1]
