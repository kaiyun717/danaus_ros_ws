import torch
import numpy as np

from torch import nn
import os, sys
import IPython
import math

g = 9.81

def create_flying_param_dict(args=None):
	# Args: for modifying the defaults through args
	param_dict = {
		"m": 0.700,
		"J_xx": 0.00327227,
		"J_xy": 0.00000791,
		"J_xz": 0.00009286,
		"J_yy": 0.00319928,
		"J_yz": 0.00005426,
		"J_zz": 0.00529553,
		"angle": 0.9222,	# 52.84 degrees	(rotor arm to x-axis)
		"r1": 0.11053858,	# Rotor arm length for rotors 1 (FR) and 3 (RL)
		"r2": 0.11232195, 	# Rotor arm length for rotors 2 (FL) and 4 (RR)
		"m_s": 0.0183,		# Moment scale for 2204-2300KV motors
		"m_p": 0.046,		# Mass of pendulum
		"L_p": 0.69, 		# This is the CoM. Total length: 1.085m
		"max_thrust": 1.20,	# Max thrust for each motor - set to 1200g at 100% throttle
		"min_thrust": 0.00,	# Min thrust for each motor - set to 0g at 0% throttle
		'delta_safety_limit': math.pi / 4  # should be <= math.pi/4
	}
	param_dict["M"] = param_dict["m"] + param_dict["m_p"]
	state_index_names = [
		"gamma", "beta", "alpha", 
		"dgamma", "dbeta", "dalpha", 
		"phi", "theta", 
		"dphi", "dtheta"
	]  # excluded x, y, z
	state_index_dict = dict(zip(state_index_names, np.arange(len(state_index_names))))

	r = 2
	x_dim = len(state_index_names)
	u_dim = 4
	ub = args.box_ang_vel_limit		# default is 20 rad/s, which is reasonable for a drone
	thresh = np.array([math.pi / 3, 
					   math.pi / 3, 
					   math.pi, 
					   ub, 
					   ub, 
					   ub, 
					   math.pi / 3, 
					   math.pi / 3, 
					   ub, 
					   ub],
	                  dtype=np.float32) # angular velocities bounds probably much higher in reality (~10-20 for drone, which can do 3 flips in 1 sec).

	x_lim = np.concatenate((-thresh[:, None], thresh[:, None]), axis=1)  # (13, 2)

	# Save stuff in param dict
	param_dict["state_index_dict"] = state_index_dict
	param_dict["r"] = r
	param_dict["x_dim"] = x_dim
	param_dict["u_dim"] = u_dim
	param_dict["x_lim"] = x_lim

	# write args into the param_dict
	param_dict["L_p"] = args.pend_length

	return param_dict


class HSum(nn.Module):
	def __init__(self, param_dict):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.i = self.state_index_dict

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# Return value is size (bs, 1)

		# print("Inside HSum forward")
		# IPython.embed()
		theta = x[:, [self.i["theta"]]]
		phi = x[:, [self.i["phi"]]]
		gamma = x[:, [self.i["gamma"]]]
		beta = x[:, [self.i["beta"]]]

		cos_cos = torch.cos(theta)*torch.cos(phi)
		eps = 1e-4 # prevents nan when cos_cos = +/- 1 (at x = 0)
		with torch.no_grad():
			signed_eps = -torch.sign(cos_cos)*eps
		delta = torch.acos(cos_cos + signed_eps)
		rv = delta**2 + gamma**2 + beta**2 - self.delta_safety_limit**2

		return rv
	

class XDot(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device
		self.i = self.state_index_dict

	def forward(self, x, u):
		# x: bs x 10, u: bs x 4
		# The way these are implemented should be batch compliant

		# Pre-computations
		# Compute the rotation matrix from quad to global frame
		# Extract the k_{x,y,z}
		gamma = x[:, self.i["gamma"]]
		beta = x[:, self.i["beta"]]
		alpha = x[:, self.i["alpha"]]

		phi = x[:, self.i["phi"]]
		theta = x[:, self.i["theta"]]
		dphi = x[:, self.i["dphi"]]
		dtheta = x[:, self.i["dtheta"]]

		R = torch.zeros((x.shape[0], 3, 3), device=self.device) # is this the correct rotation?
		R[:, 0, 0] = torch.cos(alpha)*torch.cos(beta)
		R[:, 0, 1] = torch.cos(alpha)*torch.sin(beta)*torch.sin(gamma) - torch.sin(alpha)*torch.cos(gamma)
		R[:, 0, 2] = torch.cos(alpha)*torch.sin(beta)*torch.cos(gamma) + torch.sin(alpha)*torch.sin(gamma)
		R[:, 1, 0] = torch.sin(alpha)*torch.cos(beta)
		R[:, 1, 1] = torch.sin(alpha)*torch.sin(beta)*torch.sin(gamma) + torch.cos(alpha)*torch.cos(gamma)
		R[:, 1, 2] = torch.sin(alpha)*torch.sin(beta)*torch.cos(gamma) - torch.cos(alpha)*torch.sin(gamma)
		R[:, 2, 0] = -torch.sin(beta)
		R[:, 2, 1] = torch.cos(beta)*torch.sin(gamma)
		R[:, 2, 2] = torch.cos(beta)*torch.cos(gamma)

		k_x = R[:, 0, 2]
		k_y = R[:, 1, 2]
		k_z = R[:, 2, 2]

		F = (u[:, 0] + self.M*g)

		###### Computing state derivatives
		# IPython.embed()
  
		# J = torch.tensor([
		# 	[self.J_xx, self.J_xy, self.J_xz],
		# 	[self.J_xy, self.J_yy, self.J_yz],
		# 	[self.J_xz, self.J_yz, self.J_zz]]).to(self.device)
		# norm_torques = u[:, 1:]@torch.inverse(J)
		J_inv = torch.tensor([
			[305.7518,  -0.6651,  -5.3547],
			[ -0.6651, 312.6261,  -3.1916],
			[ -5.3547,  -3.1916, 188.9651]]).to(self.device)	# for faster computation
		norm_torques = u[:, 1:]@J_inv
		
		ddquad_angles = torch.bmm(R, norm_torques[:, :, None]) # (N, 3, 1)
		ddquad_angles = ddquad_angles[:, :, 0]
		# ddgamma = (1.0/self.J_x)*ddquad_angles[:, 0]
		# ddbeta = (1.0/self.J_y)*ddquad_angles[:, 1]
		# ddalpha = (1.0/self.J_z)*ddquad_angles[:, 2]

		ddgamma = ddquad_angles[:, 0]
		ddbeta = ddquad_angles[:, 1]
		ddalpha = ddquad_angles[:, 2]

		ddphi = (3.0)*(k_y*torch.cos(phi) + k_z*torch.sin(phi))/(2*self.M*self.L_p*torch.cos(theta))*F + 2*dtheta*dphi*torch.tan(theta)
		ddtheta = (3.0*(-k_x*torch.cos(theta)-k_y*torch.sin(phi)*torch.sin(theta) + k_z*torch.cos(phi)*torch.sin(theta))/(2.0*self.M*self.L_p))*F - torch.square(dphi)*torch.sin(theta)*torch.cos(theta)

		# Excluding translational motion
		rv = torch.cat([x[:, [self.i["dgamma"]]], x[:, [self.i["dbeta"]]], x[:, [self.i["dalpha"]]], ddgamma[:, None], ddbeta[:, None], ddalpha[:, None], dphi[:, None], dtheta[:, None], ddphi[:, None], ddtheta[:, None]], axis=1)
		return rv


class ULimitSetVertices(nn.Module):
	def __init__(self, param_dict, device):
		super().__init__()
		self.__dict__.update(param_dict)  # __dict__ holds and object's attributes
		self.device = device

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
		
		r1 = np.concatenate((np.zeros(8), np.ones(8)))
		r2 = np.concatenate((np.zeros(4), np.ones(4), np.zeros(4), np.ones(4)))
		r3 = np.concatenate((np.zeros(2), np.ones(2),np.zeros(2), np.ones(2), np.zeros(2), np.ones(2),np.zeros(2), np.ones(2)))
		r4 = np.zeros(16)
		r4[1::2] = 1.0
		impulse_vert = np.concatenate((r1[None], r2[None], r3[None], r4[None]), axis=0) # 16 vertices in the impulse control space
		impulse_vert = impulse_vert * max_thrust # scale to max thrust
		
		force_vert = M@impulse_vert - np.array([[self.M*g], [0.0], [0.0], [0.0]]) # Fixed bug: was subtracting self.M*g (not just in the first row)
		force_vert = force_vert.T.astype("float32")
		self.vert = torch.from_numpy(force_vert).to(self.device)

	def forward(self, x):
		# The way these are implemented should be batch compliant
		# (bs, n_vertices, u_dim) or (bs, 16, 4)

		rv = self.vert
		rv = rv.unsqueeze(dim=0)
		rv = rv.expand(x.shape[0], -1, -1)
		return rv