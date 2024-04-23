import IPython

import numpy as np
from cvxopt import matrix, solvers
from src.controllers.cbf import CBF

g = 9.81

class CBFController:
    def __init__(self, qp_weight: np.ndarray, u_max: np.ndarray, u_min: np.ndarray,
                 L_p=0.69, delta_max=np.pi/4, rs_max=0.35,
                 kappa=8.7899304e3, n1=2.15029699, n2=1, k=0.01):
        
        ################################
        ##### QP solver parameters #####
        ################################
        self.qp_weight = qp_weight  # Diagonal (4x4) cost matrix: F, wx, wy, wz
        self.u_max = u_max
        self.u_min = u_min

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
        self.cbf = CBF(L_p, delta_max, rs_max, kappa, n1, n2, k)

    def solve_qp(self, x, u_ref):
        # IPython.embed()
        
        Lgh = self.cbf.Lgh(x)
        Lfh = self.cbf.Lfh(x)
        h = self.cbf.phi_fn(x)

        ######################################
        ############## USUAL QP ##############
        ######################################
        H = self.qp_weight
        f = -self.qp_weight @ u_ref
        A = np.vstack((
            -Lgh,
            -np.identity(4),
            np.identity(4)
        ))
        B = np.vstack((
            Lfh, # + 0.01*h,
            -self.u_min,
            self.u_max
        ))
        try:
            u_safe = solvers.qp(matrix(H*1.0), matrix(f*1.0), matrix(A*1.0), matrix(B*1.0))
        except ValueError as e:
            IPython.embed()
        return np.array(u_safe["x"])
        
        # if np.linalg.norm(Lgh) < 1e-6:
        #     return u_ref
        # else:
        #     nu = -Lfh - Lgh@u_ref
        #     u_safe = u_ref + np.max([0, nu[0][0]]) * Lgh.reshape((4,1))/(np.linalg.norm(Lgh))**2
        #     # new_u_min = np.copy(self.u_min)
        #     # new_u_min[0] = 9.81
        #     # return np.clip(u_safe, new_u_min, self.u_max)
        #     return np.clip(u_safe, self.u_min, self.u_max)


if __name__ == "__main__":
    qp_weight = np.diag([0.1, 1, 1, 1]) * 10
    u_max = np.array([2*g, 15, 15, 15]).reshape((4,1))
    u_min = np.array([0, -15, -15, -15]).reshape((4,1))
    kappa = 10
    cbf_controller = CBFController(qp_weight, u_max, u_min, kappa=kappa)
    x = np.array([[-3.41025840e-02],
                  [ 7.97659728e-02],
                  [ 1.59336866e+00],
                  [ 1.59271568e-02],
                  [-4.09141079e-02],
                  [-1.41304152e-02],
                  [ 7.11225925e-02],
                  [ 7.06574339e-04],
                  [ 2.57196482e-03],
                  [-1.69401132e-03],
                  [-9.14248315e-04],
                  [-2.70682937e-02],
                  [ 7.13906035e-02]])
    u_nom = np.array([[ 9.73726712e+00],
                      [-1.51274229e-04],
                      [ 1.74296839e-03],
                      [ 0.00000000e+00]])

    u_safe = cbf_controller.solve_qp(x[6:].reshape(-1), u_nom)
    print(u_safe)