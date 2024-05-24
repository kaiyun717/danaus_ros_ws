import time
import torch
from torch import nn
from torch.autograd import grad

class NeuralCBF(nn.Module):
	# Note: currently, we have a implementation which is generic to any r. May be slow
	def __init__(self, h_fn, xdot_fn, 
			           r, x_dim, u_dim, 
					   device, args, 
					   nn_input_modifier=None, x_e=None):
		"""
		:param h_fn:
		:param xdot_fn:
		:param r:
		:param x_dim:
		:param u_dim:
		:param device:
		:param args:
		:param nn_input_modifier:
		:param x_e:
		"""
		# Later: args specifying how beta is parametrized
		super().__init__()
		variables = locals()  # dict of local names
		self.__dict__.update(variables)  # __dict__ holds and object's attributes
		del self.__dict__["self"]  # don't need `self`
		assert r>=0

		# turn Namespace into dict
		args_dict = vars(args)

		# Note: by default, it registers parameters by their variable name
		self.ci = nn.Parameter(args.phi_ci_init_range*torch.rand(r-1, 1)) # if ci in small range, ki will be much larger
		# rng = args.phi_k0_init_max - args.phi_k0_init_min
		# self.k0 = nn.Parameter(rng*torch.rand(1, 1) + args.phi_k0_init_min)
		self.k0 = nn.Parameter(args.phi_ci_init_range*torch.rand(1, 1))

		# IPython.embed()s
		# To enforce strict positivity for both
		self.ci_min = 1e-2
		self.k0_min = 1e-2

		# print("At initialization: k0 is %f" % self.k0.item())
		#############################################################
		self.net_reshape_h = self._create_net()

		# if phi_reshape_dh:
		# 	self.net_reshape_dh = self._create_net()
		self.pos_param_names = ["ci", "k0"]
		self.exclude_from_gradient_param_names = ["ci", "k0"]

	def _create_net(self):
		# print("create_net")
		# IPython.embed()

		hidden_dims = self.args.phi_nn_dimension.split("-")
		hidden_dims = [int(h) for h in hidden_dims]
		hidden_dims.append(1)

		# Input dim:
		if self.nn_input_modifier is None:
			prev_dim = self.x_dim
		else:
			prev_dim = self.nn_input_modifier.output_dim

		# phi_nnl = args_dict.get("phi_nnl", "relu") # return relu if var "phi_nnl" not on namespace
		phi_nnls = self.args.phi_nnl.split("-")
		assert len(phi_nnls) == len(hidden_dims)

		net_layers = []
		for hidden_dim, phi_nnl in zip(hidden_dims, phi_nnls):
			net_layers.append(nn.Linear(prev_dim, hidden_dim))
			if phi_nnl == "relu":
				net_layers.append(nn.ReLU())
			elif phi_nnl == "tanh":
				net_layers.append(nn.Tanh())
			elif phi_nnl == "softplus":
				net_layers.append(nn.Softplus())
			prev_dim = hidden_dim
		net = nn.Sequential(*net_layers)
		return net

	def forward(self, x, grad_x=False):
		# The way these are implemented should be batch compliant
		# Assume x is (bs, x_dim)
		# RV is (bs, r+1)

		# print("inside phi's forward")
		# IPython.embed()
		k0 = self.k0 + self.k0_min
		ci = self.ci + self.ci_min

		# Convert ci to ki
		ki = torch.tensor([[1.0]])
		ki_all = torch.zeros(self.r, self.r).to(self.device) # phi_i coefficients are in row i
		ki_all[0, 0:ki.numel()] = ki
		for i in range(self.r-1): # A is current coeffs
			A = torch.zeros(torch.numel(ki)+1, 2)
			A[:-1, [0]] = ki
			A[1:, [1]] = ki

			# Note: to preserve gradient flow, have to assign mat entries to ci not create with ci (i.e. torch.tensor([ci[0]]))
			binomial = torch.ones((2, 1))
			binomial[1] = ci[i]
			ki = A.mm(binomial)

			ki_all[i+1, 0:ki.numel()] = ki.view(1, -1)
			# Ultimately, ki should be r x 1

		# Compute higher-order Lie derivatives
		#####################################################################
		# Turn gradient tracking on for x
		bs = x.size()[0]
		if grad_x == False:
			orig_req_grad_setting = x.requires_grad # Basically only useful if x.requires_grad was False before
			x.requires_grad = True

		if self.x_e is None:
			if self.nn_input_modifier is None:
				beta_net_value = self.net_reshape_h(x)
			else:
				beta_net_value = self.net_reshape_h(self.nn_input_modifier(x))
			new_h = nn.functional.softplus(beta_net_value) + k0*self.h_fn(x)
		else:
			if self.nn_input_modifier is None:
				beta_net_value = self.net_reshape_h(x)
				beta_net_xe_value = self.net_reshape_h(self.x_e)
			else:
				# torch.cuda.synchronize()
				# start_time = time.time()
				modified_input = self.nn_input_modifier(x)
				# torch.cuda.synchronize()
				# end_time = time.time()
				# print(f"Input modifier: {(end_time - start_time)*1000}")
				
				# torch.cuda.synchronize()
				# start_time = time.time()
				beta_net_value = self.net_reshape_h(modified_input)
				# torch.cuda.synchronize()
				# end_time = time.time()
				# print(f"NN time: {(end_time - start_time)*1000}")
				
				# beta_net_value = self.net_reshape_h(self.nn_input_modifier(x))
				beta_net_xe_value = self.net_reshape_h(self.nn_input_modifier(self.x_e))

			new_h = torch.square(beta_net_value - beta_net_xe_value) + k0*self.h_fn(x)

		# torch.cuda.synchronize()
		# start_time = time.time()
		h_ith_deriv = self.h_fn(x) # bs x 1, the zeroth derivative
		# torch.cuda.synchronize()
		# end_time = time.time()
		# print(f"Rho: {(end_time - start_time)*1000}")

		h_derivs = h_ith_deriv # bs x 1
		# torch.cuda.synchronize()
		# start_time = time.time()
		f_val = self.xdot_fn(x, torch.zeros(bs, self.u_dim).to(self.device)) # bs x x_dim
		# torch.cuda.synchronize()
		# end_time = time.time()
		# print(f"xdot_fn: {(end_time - start_time)*1000}")

		for i in range(self.r-1):
			
			# torch.cuda.synchronize()
			# start_time = time.time()
			grad_h_ith = grad([torch.sum(h_ith_deriv)], x, create_graph=True)[0] # bs x x_dim; create_graph ensures gradient is computed through the gradient operation
			# torch.cuda.synchronize()
			# end_time = time.time()
			# print(f"Rho grad: {(end_time - start_time)*1000}")

			h_ith_deriv = (grad_h_ith.unsqueeze(dim=1)).bmm(f_val.unsqueeze(dim=2)) # bs x 1 x 1
			h_ith_deriv = h_ith_deriv[:, :, 0] # bs x 1
			h_derivs = torch.cat((h_derivs, h_ith_deriv), dim=1)

		if grad_x == False:
			x.requires_grad = orig_req_grad_setting
		#####################################################################
		# Turn gradient tracking off for x
		result = h_derivs.mm(ki_all.t())
		phi_r_minus_1_star = result[:, [-1]] - result[:, [0]] + new_h

		result = torch.cat((result, phi_r_minus_1_star), dim=1)

		# print("inside Phi's forward")
		# IPython.embed()
		return result
