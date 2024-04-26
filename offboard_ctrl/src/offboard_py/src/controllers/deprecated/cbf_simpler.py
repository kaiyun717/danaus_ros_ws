import IPython

import math
import torch
import numpy as np

g = 9.81


class CBF:
    def __init__(self, L_p=0.69, delta_max=np.pi/4, rs_max=0.35,
                 kappa=8.7899304e3, n1=2.15029699, n2=1, k=0.01):
        
        ################################
        ###### System parameters #######
        ################################
        self.L_p = L_p
        self.delta_max = delta_max
        self.rs_max = rs_max
        
        ################################
        ######## CBF parameters ########
        ################################
        self.kappa = kappa
        self.n1 = n1
        self.n2 = n2
        self.k = k

    def _convert_angle_to_negpi_pi(self, angle):
        """ Converts angle to the range [-pi, pi] """
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def _x_numpy_to_x_torch(self, x):
        """ Converts numpy array to torch.tensor """
        ind_cyclic = [0, 1, 2]
        for i in ind_cyclic:
            x[i] = self._convert_angle_to_negpi_pi(x[i])
        x_torch = torch.from_numpy(x.astype("float32"))
        return x_torch

    def torch_phi(self, x):
        """ Takes in torch.tensor x and returns the value of the CBF function """
        L_p = self.L_p
        
        ##########################################################################
        ### x = [alpha (0), beta (1), gamma (2), r (3), s (4), dr (5), ds (6)] ###
        ##########################################################################
        alpha, beta, gamma, r, s, dr, ds = x
        
        xi = torch.sqrt(L_p**2 - r**2 - s**2)
        cos_cos = xi/L_p
        eps = 1e-4
        signed_eps = -torch.sign(cos_cos)*eps
        delta = torch.acos(cos_cos + signed_eps)

        # ######################################
        # ############ INTERSECTION ############
        # ######################################
        # h1 = (self.delta_max**2)**self.n1 - (delta**2 + beta**2 + gamma**2)**self.n1
        h1 = (self.delta_max**2)**self.n1 - (beta**2 + gamma**2)**self.n1
        h2 = (self.rs_max**2)**self.n2 - (r**2 + s**2)**self.n2 - self.k*(2*r*dr + 2*s*ds)

        h_complex = -1/self.kappa * torch.log(torch.exp(-self.kappa*h1) + torch.exp(-self.kappa*h2)) + math.log(2)/self.kappa

        ######################################
        ############### UNION ################
        ######################################
        # # h1 = (self.delta_max**2)**self.n1 - (beta**2 + gamma**2)**self.n1
        # h1 = (self.delta_max**2)**self.n1 - (delta**2 + beta**2 + gamma**2)**self.n1
        # h2 = (self.rs_max**2)**self.n2 - (r**2 + s**2)**self.n2 - self.k*(2*r*dr + 2*s*ds)

        # h_complex = 1/self.kappa * torch.log(torch.exp(self.kappa*h1) + torch.exp(self.kappa*h2)) - math.log(2)/self.kappa

        return h_complex
    
    # def phi_fn(self, x):
    #     """ Takes in numpy array x and returns the value of the CBF function in numpy """
    #     x_torch = self._x_numpy_to_x_torch(x)
    #     phi_torch = self.torch_phi(x_torch)
    #     phi_numpy = phi_torch.detach().cpu().numpy()
    #     return phi_numpy
    
    # def phi_grad(self, x):
    #     """ Takes in numpy array x and returns the gradient of the CBF function in numpy """
    #     # IPython.embed()
    #     x_torch = self._x_numpy_to_x_torch(x)
    #     x_torch.requires_grad = True
    #     phi_torch = self.torch_phi(x_torch)
    #     phi_grad = torch.autograd.grad(phi_torch, x_torch)[0]
        
    #     x_torch.requires_grad = False
    #     phi_grad = phi_grad.detach().cpu().numpy()
    #     return phi_grad

    # def Lfh(self, x):
    #     """ Takes in numpy array x and returns the value of Lfh in numpy """
    #     L_f_h = self.phi_grad(x).dot(self.f(x))
    #     return L_f_h

    # def Lgh(self, x):
    #     """ Takes in numpy array x and returns the value of Lfg in numpy """
    #     L_g_h = self.phi_grad(x).dot(self.g(x))
    #     return L_g_h

    def h1_fn(self, x):
        """ Takes in numpy array x and returns the value of the CBF function in numpy """
        L_p = self.L_p
        alpha, beta, gamma, r, s, dr, ds = x

        xi = np.sqrt(L_p**2 - r**2 - s**2)
        cos_cos = xi/L_p
        eps = 1e-4
        signed_eps = -np.sign(cos_cos)*eps
        delta = np.arccos(cos_cos + signed_eps)

        h1 = (self.delta_max**2)**self.n1 - (delta**2 + beta**2 + gamma**2)**self.n1
        # h1 = (self.delta_max**2)**self.n1 - (beta**2 + gamma**2)**self.n1
        return h1
    
    def h2_fn(self, x):
        """ Takes in numpy array x and returns the value of the CBF function in numpy """
        L_p = self.L_p
        alpha, beta, gamma, r, s, dr, ds = x

        h2 = (self.rs_max**2)**self.n2 - (r**2 + s**2)**self.n2 - self.k*(2*r*dr + 2*s*ds)
        return h2
    
    def phi_fn(self, x):
        """ Takes in torch.tensor x and returns the value of the CBF function """

        h_complex = -1/self.kappa * np.log(np.exp(-self.kappa*self.h1_fn(x)) + np.exp(-self.kappa*self.h2_fn(x))) #+ math.log(2)/self.kappa
        h_complex = -np.log(np.exp(-self.h1_fn(x)) + np.exp(-self.h2_fn(x)))

        return h_complex
    
    def phi_grad(self, x):
        """ Takes in numpy array x and returns the gradient of the CBF function in numpy """
        alpha, beta, gamma, r, s, dr, ds = x
        L = self.L_p
        # dh1_dx = np.array([0, 
        #                    -2*self.n1*beta*(beta**2+gamma**2)**(self.n1-1),
        #                    -2*self.n1*gamma*(beta**2+gamma**2)**(self.n1-1),
        #                    0, 0, 0, 0]).reshape((1,7))
        xi = np.sqrt(L**2 - r**2 - s**2)
        cos_cos = xi/L
        eps = 1e-4
        eps = -np.sign(cos_cos)*eps
        
        dh1_dx = np.array([
            0, 
            -2*beta*self.n1*(np.arccos(eps + (L**2 - r**2 - s**2)**(1/2)/L)**2 + beta**2 + gamma**2)**(self.n1 - 1), 
            -2*gamma*self.n1*(np.arccos(eps + (L**2 - r**2 - s**2)**(1/2)/L)**2 + beta**2 + gamma**2)**(self.n1 - 1), 
            -(2*self.n1*r*np.arccos(eps + (L**2 - r**2 - s**2)**(1/2)/L)*(np.arccos(eps + (L**2 - r**2 - s**2)**(1/2)/L)**2 + beta**2 + gamma**2)**(self.n1 - 1))/(L*(1 - (eps + (L**2 - r**2 - s**2)**(1/2)/L)**2)**(1/2)*(L**2 - r**2 - s**2)**(1/2)), 
            -(2*self.n1*s*np.arccos(eps + (L**2 - r**2 - s**2)**(1/2)/L)*(np.arccos(eps + (L**2 - r**2 - s**2)**(1/2)/L)**2 + beta**2 + gamma**2)**(self.n1 - 1))/(L*(1 - (eps + (L**2 - r**2 - s**2)**(1/2)/L)**2)**(1/2)*(L**2 - r**2 - s**2)**(1/2)), 
            0, 
            0])
        dh2_dx = np.array([0, 0, 0, 
                           -2*self.k*dr - 2*self.n2*r*(r**2+s**2)**(self.n2-1),
                           -2*self.k*ds - 2*self.n2*s*(r**2+s**2)**(self.n2-1),
                           -2*self.k*r, -2*self.k*s]).reshape((1,7))
        return dh1_dx, dh2_dx
   
    def Lfh(self, x):
        """ Takes in numpy array x and returns the value of Lfh in numpy """
        h_complex = self.phi_fn(x)
        
        dh1_dx, dh2_dx = self.phi_grad(x)
        
        h1 = self.h1_fn(x)
        h2 = self.h2_fn(x)

        lambda_1 = np.exp(-self.kappa*(h1 - h_complex))
        lambda_2 = np.exp(-self.kappa*(h2 - h_complex))

        L_f_h = lambda_1*(dh1_dx.dot(self.f(x))) + lambda_2*(dh2_dx.dot(self.f(x)))
        return L_f_h

    def Lgh(self, x):
        """ Takes in numpy array x and returns the value of Lfg in numpy """
        h_complex = self.phi_fn(x)
        
        dh1_dx, dh2_dx = self.phi_grad(x)
        
        h1 = self.h1_fn(x)
        h2 = self.h2_fn(x)

        lambda_1 = np.exp(-self.kappa*(h1 - h_complex))
        lambda_2 = np.exp(-self.kappa*(h2 - h_complex))

        L_g_h = lambda_1*(dh1_dx.dot(self.g(x))) + lambda_2*(dh2_dx.dot(self.g(x)))
        return L_g_h

    def f(self, x):
        alpha, beta, gamma, r, s, dr, ds = x
        L_p = self.L_p

        z_ddot = -g

        f_val = np.zeros((7,1))
        f_val[0] = 0
        f_val[1] = 0
        f_val[2] = 0
        f_val[3] = dr
        f_val[4] = ds
        f_val[5] = (- r**3*ds**2 + L_p**2*r*dr**2 + L_p**2*r*ds**2 
                    + g*r**3*(L_p**2 - r**2 - s**2)**(1/2) 
                    - r*dr**2*s**2 + r**3*z_ddot*(L_p**2 - r**2 - s**2)**(1/2) 
                    + 2*r**2*dr*s*ds - L_p**2*g*r*(L_p**2 - r**2 - s**2)**(1/2) 
                    - L_p**2*r*z_ddot*(L_p**2 - r**2 - s**2)**(1/2) 
                    + g*r*s**2*(L_p**2 - r**2 - s**2)**(1/2) 
                    + r*s**2*z_ddot*(L_p**2 - r**2 - s**2)**(1/2)) \
                /(L_p**2*(- L_p**2 + r**2 + s**2))
        f_val[6] = (- dr**2*s**3 + L_p**2*dr**2*s + L_p**2*s*ds**2 
                    + g*s**3*(L_p**2 - r**2 - s**2)**(1/2) 
                    - r**2*s*ds**2 + s**3*z_ddot*(L_p**2 - r**2 - s**2)**(1/2) 
                    + 2*r*dr*s**2*ds - L_p**2*g*s*(L_p**2 - r**2 - s**2)**(1/2) 
                    - L_p**2*s*z_ddot*(L_p**2 - r**2 - s**2)**(1/2) 
                    + g*r**2*s*(L_p**2 - r**2 - s**2)**(1/2) 
                    + r**2*s*z_ddot*(L_p**2 - r**2 - s**2)**(1/2)) \
                /(- L_p**4 + L_p**2*r**2 + L_p**2*s**2)

        return f_val

    def g(self, x):
        alpha, beta, gamma, r, s, dr, ds = x
        L_p = self.L_p

        x_ddot = np.array([np.cos(alpha)*np.sin(beta)*np.cos(gamma) + np.sin(alpha)*np.sin(gamma), 0, 0, 0])
        y_ddot = np.array([np.sin(alpha)*np.sin(beta)*np.cos(gamma) - np.cos(alpha)*np.sin(gamma), 0, 0, 0])
        z_ddot = np.array([np.cos(beta)*np.cos(gamma), 0, 0, 0])

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

        g_alpha_dot = np.hstack((0, A[2,:]))
        g_beta_dot = np.hstack((0, A[1,:]))
        g_gamma_dot = np.hstack((0, A[0,:]))
        
        g_r_dot = np.array([0, 0, 0, 0])
        g_s_dot = np.array([0, 0, 0, 0])

        g_r_ddot = (L_p**4*x_ddot + r**4*x_ddot + r*s**3*y_ddot + r**3*s*y_ddot 
                    - 2*L_p**2*r**2*x_ddot - L_p**2*s**2*x_ddot 
                    + r**3*z_ddot*(L_p**2 - r**2 - s**2)**(1/2) 
					+ r**2*s**2*x_ddot - L_p**2*r*s*y_ddot 
                    - L_p**2*r*z_ddot*(L_p**2 - r**2 - s**2)**(1/2) 
					+ r*s**2*z_ddot*(L_p**2 - r**2 - s**2)**(1/2)) \
				/(L_p**2*(- L_p**2 + r**2 + s**2))

        g_s_ddot = (L_p**4*y_ddot + s**4*y_ddot + r*s**3*x_ddot + r**3*s*x_ddot 
                    - L_p**2*r**2*y_ddot - 2*L_p**2*s**2*y_ddot 
                    + s**3*z_ddot*(L_p**2 - r**2 - s**2)**(1/2) 
                    + r**2*s**2*y_ddot - L_p**2*r*s*x_ddot
                    - L_p**2*s*z_ddot*(L_p**2 - r**2 - s**2)**(1/2) 
                    + r**2*s*z_ddot*(L_p**2 - r**2 - s**2)**(1/2)) \
                /(- L_p**4 + L_p**2*r**2 + L_p**2*s**2)

        g_val = np.vstack((g_alpha_dot, g_beta_dot, g_gamma_dot, g_r_dot, g_s_dot, g_r_ddot, g_s_ddot))
        return g_val


if __name__ == "__main__":
    controller = CBF()
    
    IPython.embed()
    
    x = np.array([0, 0, 0, 0, 0, 0, 0])
    # print(controller.phi_fn(x))
    # print(controller.phi_grad(x))
    # print(controller.Lfh(x))
    # print(controller.Lfg(x))
    print(controller.f(x))
    print(controller.g(x))