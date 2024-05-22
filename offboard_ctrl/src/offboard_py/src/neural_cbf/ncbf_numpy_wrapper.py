import time
import scipy as sp
import numpy as np
import IPython
import torch
import math
from torch.autograd import grad


class NCBFNumpy:
    """ Originally `PhiNumpy` """
    def __init__(self, torch_phi_fn, device):
        self.torch_phi_fn = torch_phi_fn
        self.device = device

    def set_params(self, state_dict):
        # TODO: probably should have some checks on this
        self.torch_phi_fn.load_state_dict(state_dict, strict=False)

    def _convert_angle_to_negpi_pi_interval(self, angle):
        new_angle = np.arctan2(np.sin(angle), np.cos(angle))
        return new_angle

    def _x_numpy_to_x_torch(self, x):
        # Slice off translational states, if they are present

        # if len(x.shape) == 2: # NOTE: just commented out for test purposes. NEED!
        #     x = np.reshape(x, (1, -1))
        x = x[:, :10]

        # Wrap-around on cyclical angles
        # IPython.embed()
        
        ind_cyclical = [0, 1, 2, 6, 7]
        for i in ind_cyclical:
            x[:, i] = self._convert_angle_to_negpi_pi_interval(x[:, i])

        # torch.cuda.synchronize()
        start_time = time.time()

        x_torch = torch.from_numpy(x.astype("float32")).to(self.device)
        
        # torch.cuda.synchronize()
        end_time = time.time()
        print(f"Numpy to torch: {(end_time - start_time)*1000}")
        # Q: how come we don't have to involve device = gpu?
        # A: because it is set as CPU elsewhere? Yes

        return x_torch      

    def phi_fn(self, x):
        """
        :param x: (16)
        :return: (r+1) where r is relative degree
        """
        x_torch = self._x_numpy_to_x_torch(x)
        phi_torch = self.torch_phi_fn(x_torch)

        # torch.cuda.synchronize()
        start_time = time.time()
			
        # phi_numpy = phi_torch.detach().cpu().numpy()
        phi_numpy = phi_torch.detach().numpy()

        # torch.cuda.synchronize()
        end_time = time.time()
        print(f"Torch to numpy: {(end_time - start_time)*1000}")

        return phi_numpy
    
    # def phi_fn_and_grad(self, x):
    #     """
    #     :param x: (16)
    #     :return: (r+1) where r is relative degree
    #     """
    #     x_torch = self._x_numpy_to_x_torch(x)
    #     bs = x_torch.shape[0]
    #     x_torch.requires_grad = True

    #     phi_torch = self.torch_phi_fn(x_torch, grad_x=True)
    #     phi_val = torch.sum(phi_torch[:, -1])
    #     phi_val.backward()
    #     phi_grad = x_torch.grad

    #     x_torch.requires_grad = False
    #     phi_grad = phi_grad.detach().numpy()
    #     phi_grad = np.concatenate((phi_grad, np.zeros((bs, 6))), axis=1)
        
    #     phi_numpy = phi_torch.detach().numpy()

    #     return phi_numpy, phi_grad

    def phi_grad(self, x):
        """
        :param x: (16)
        :return: (16)
        """
        self.torch_phi_fn.zero_grad()
        
        x_torch = self._x_numpy_to_x_torch(x)
        bs = x_torch.shape[0]
        x_torch.requires_grad = True

        # Compute phi grad
        phi_vals = self.torch_phi_fn(x_torch)
        phi_val = torch.sum(phi_vals[:, -1])
        phi_grad = grad([phi_val], x_torch)[0]    # NOTE: Original

        # Post operation
        x_torch.requires_grad = False

        # phi_grad = phi_grad.detach().cpu().numpy()
        phi_grad = phi_grad.detach().numpy()
        phi_grad = np.concatenate((phi_grad, np.zeros((bs, 6))), axis=1)

        return phi_grad