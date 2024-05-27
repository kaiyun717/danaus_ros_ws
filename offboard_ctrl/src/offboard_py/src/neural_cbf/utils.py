import os
import json
import math
from dotmap import DotMap
import torch
import torch.nn as nn
import pickle
import time, os, glob, re
import numpy as np

from src.neural_cbf.neural_cbf import NeuralCBF
from src.env.flying_inv_pend import HSum, XDot, ULimitSetVertices

def load_phi_and_params(exp_name, checkpoint_number, device):
    

    ###############################
    ##### Load the parameters #####
    ###############################
    fnm = "/home/oem/danaus_ros_ws/offboard_ctrl/src/trained_ncbf/log/%s/args.txt" % exp_name
    # fnm = "/home/dvij/ncbf-drone/cbf_synthesis_small/log/%s/args.txt" % exp_name
    # args = load_args(fnm) # can't use, args conflicts with args in outer scope
    with open(fnm, 'r') as f:
        json_data = json.load(f)
    args = DotMap(json_data)
    param_dict = pickle.load(open("/home/oem/danaus_ros_ws/offboard_ctrl/src/trained_ncbf/log/%s/param_dict.pkl" % exp_name, "rb"))
    # param_dict = pickle.load(open("/home/dvij/ncbf-drone/cbf_synthesis_small/log/%s/param_dict.pkl" % exp_name, "rb"))

    r = param_dict["r"]
    x_dim = param_dict["x_dim"]
    u_dim = param_dict["u_dim"]
    x_lim = param_dict["x_lim"]

    ###############################
    #####  Load the functions #####
    ###############################
    h_fn = HSum(param_dict)
    xdot_fn = XDot(param_dict, device)
    uvertices_fn = ULimitSetVertices(param_dict, device)

    if args.phi_include_xe:
        x_e = torch.zeros(1, x_dim)
    else:
        x_e = None

    state_index_dict = param_dict["state_index_dict"]
    if args.phi_nn_inputs == "spherical":
        nn_input_modifier = None
    elif args.phi_nn_inputs == "euc":
        nn_input_modifier = TransformEucNNInput(state_index_dict)

    ###############################
    #####   Send to device    #####
    ###############################
    h_fn = h_fn.to(device)
    xdot_fn = xdot_fn.to(device)
    uvertices_fn = uvertices_fn.to(device)
    if x_e is not None:
        x_e = x_e.to(device)

    ###############################
    #####   Create the nCBF   #####
    ###############################
    phi_fn = NeuralCBF(h_fn, xdot_fn, 
                        r, x_dim, u_dim, 
                        device, args, 
                        x_e=x_e, nn_input_modifier=nn_input_modifier)
    phi_fn = phi_fn.to(device)

    ###############################
    ##### Load the checkpoint #####
    ###############################
    print("=====================================================")
    print("Loading checkpoint %d of experiment %s" % (checkpoint_number, exp_name))
    print("=====================================================")
    phi_load_fpth = "/home/oem/danaus_ros_ws/offboard_ctrl/src/trained_ncbf/checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
    # phi_load_fpth = "/home/dvij/ncbf-drone/cbf_synthesis_small/checkpoint/%s/checkpoint_%i.pth" % (exp_name, checkpoint_number)
    phi_fn.load_state_dict(torch.load(phi_load_fpth, map_location=lambda storage, loc: storage))

    return phi_fn, param_dict


class TransformEucNNInput(nn.Module):
# Note: this is specific to FlyingInvPend
    def __init__(self, state_index_dict):
        """
        :param which_ind: flat numpy array
        """
        super().__init__()
        self.state_index_dict = state_index_dict
        self.output_dim = 12

    def forward(self, x):
        alpha = x[:, self.state_index_dict["alpha"]]
        beta = x[:, self.state_index_dict["beta"]]
        gamma = x[:, self.state_index_dict["gamma"]]

        dalpha = x[:, self.state_index_dict["dalpha"]]
        dbeta = x[:, self.state_index_dict["dbeta"]]
        dgamma = x[:, self.state_index_dict["dgamma"]]

        phi = x[:, self.state_index_dict["phi"]]
        theta = x[:, self.state_index_dict["theta"]]

        dphi = x[:, self.state_index_dict["dphi"]]
        dtheta = x[:, self.state_index_dict["dtheta"]]

        cos_alpha = torch.cos(alpha)
        cos_beta = torch.cos(beta)
        cos_gamma = torch.cos(gamma)
        cos_theta = torch.cos(theta)
        cos_phi = torch.cos(phi)
        
        sin_alpha = torch.sin(alpha)
        sin_beta = torch.sin(beta)
        sin_gamma = torch.sin(gamma)
        sin_theta = torch.sin(theta)
        sin_phi = torch.sin(phi)

        # x_quad = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)
        # y_quad = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
        # z_quad = torch.cos(beta)*torch.cos(gamma)

        # d_x_quad_d_alpha = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
        # d_x_quad_d_beta = -torch.cos(alpha)*torch.cos(beta)*torch.cos(gamma)
        # d_x_quad_d_gamma = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma)
        # v_x_quad = dalpha*d_x_quad_d_alpha + dbeta*d_x_quad_d_beta + dgamma*d_x_quad_d_gamma

        # d_y_quad_d_alpha = -torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.sin(alpha)*torch.sin(gamma)
        # d_y_quad_d_beta = -torch.sin(alpha)*torch.cos(beta)*torch.cos(gamma)
        # d_y_quad_d_gamma = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma)
        # v_y_quad = dalpha*d_y_quad_d_alpha + dbeta*d_y_quad_d_beta + dgamma*d_y_quad_d_gamma

        # v_z_quad = dbeta*torch.sin(beta)*torch.cos(gamma) + dgamma*torch.cos(beta)*torch.sin(gamma)

        # x_pend = torch.sin(theta)*torch.cos(phi)
        # y_pend = -torch.sin(phi)
        # z_pend = torch.cos(theta)*torch.cos(phi)

        # v_x_pend = -dtheta*torch.cos(theta)*torch.cos(phi) + dphi*torch.sin(theta)*torch.sin(phi)
        # v_y_pend = dphi*torch.cos(phi)
        # v_z_pend = dtheta*torch.sin(theta)*torch.cos(phi) + dphi*torch.cos(theta)*torch.sin(phi)

        x_quad = cos_alpha*sin_beta*cos_gamma + sin_alpha*sin_gamma
        y_quad = sin_alpha*sin_beta*cos_gamma - cos_alpha*sin_gamma
        z_quad = cos_beta*cos_gamma

        d_x_quad_d_alpha = sin_alpha*sin_beta*cos_gamma - cos_alpha*sin_gamma
        d_x_quad_d_beta = -cos_alpha*cos_beta*cos_gamma
        d_x_quad_d_gamma = cos_alpha*sin_beta*sin_gamma - sin_alpha*cos_gamma
        v_x_quad = dalpha*d_x_quad_d_alpha + dbeta*d_x_quad_d_beta + dgamma*d_x_quad_d_gamma

        d_y_quad_d_alpha = -cos_alpha*sin_beta*cos_gamma - sin_alpha*sin_gamma
        d_y_quad_d_beta = -sin_alpha*cos_beta*cos_gamma
        d_y_quad_d_gamma = sin_alpha*sin_beta*sin_gamma + cos_alpha*cos_gamma
        v_y_quad = dalpha*d_y_quad_d_alpha + dbeta*d_y_quad_d_beta + dgamma*d_y_quad_d_gamma

        v_z_quad = dbeta*sin_beta*cos_gamma + dgamma*cos_beta*sin_gamma

        x_pend = sin_theta*cos_phi
        y_pend = -sin_phi
        z_pend = cos_theta*cos_phi

        v_x_pend = -dtheta*cos_theta*cos_phi + dphi*sin_theta*sin_phi
        v_y_pend = dphi*cos_phi
        v_z_pend = dtheta*sin_theta*cos_phi + dphi*cos_theta*sin_phi

        rv = torch.cat([x_quad[:, None], y_quad[:, None], z_quad[:, None], v_x_quad[:, None], v_y_quad[:, None], v_z_quad[:, None], x_pend[:, None], y_pend[:, None], z_pend[:, None], v_x_pend[:, None], v_y_pend[:, None], v_z_pend[:, None]], dim=1)
        return rv
    