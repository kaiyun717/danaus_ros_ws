import IPython
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def find_all_folders(logs_dir):
    folders = os.listdir(logs_dir)
    folders = [folder for folder in folders if os.path.isdir(os.path.join(logs_dir, folder)) and "Good" in folder]
    folders.sort()
    folders = [os.path.join(logs_dir, folder) for folder in folders]
    return folders

def load_data(folder_path, drop_idx_from_end=None):
    """
    Load data from the folder.
    """
    log = np.load(os.path.join(folder_path, 'log.npy'), allow_pickle=True).item()
    
    state_log = log['state']
    input_log = log['input']
    error_log = log['error']
    omega_log = log['omega']
    # accel_log = log['accel']
    omega_timestamp_log = log['omega_timestamp']
    
    if drop_idx_from_end is not None:
        state_log = state_log[:, drop_idx_from_end:-drop_idx_from_end]
        input_log = input_log[:, drop_idx_from_end:-drop_idx_from_end]
        error_log = error_log[:, drop_idx_from_end:-drop_idx_from_end]
        omega_log = omega_log[:, drop_idx_from_end:-drop_idx_from_end]
        omega_timestamp_log = omega_timestamp_log[:, drop_idx_from_end:-drop_idx_from_end]
        
    params = np.load(os.path.join(folder_path, 'params.npy'), allow_pickle=True).item()
    
    return state_log, input_log, error_log, omega_log, omega_timestamp_log, params

def get_ang_values(input_log, omega_log, omega_timestamp_log):
    """
    Preprocess the data for angular values.
    """
    ### Omega measurements ###
    omega_meas = omega_log       # gamma, beta, alpha
    ### Omega inputs ###
    omega_des = input_log[1:,:]  # wx, wy, wz
    
    ### Change in omega ###
    omega_delta = omega_meas[:,1:] - omega_meas[:,:-1]  # (N-1,3)
    delta_time = omega_timestamp_log[:,1:] - omega_timestamp_log[:,:-1]     # (N-1,1)
    ### Find the nonzero-time indices ###
    nonzero_idx = np.nonzero(delta_time.squeeze())  # This is from the 0 index of `delta_time`, which corresponds to the 0 index of other arrays.
    omega_delta = omega_delta[:,nonzero_idx[0]].squeeze()
    delta_time = delta_time[:,nonzero_idx[0]].squeeze()
    omega_dot = omega_delta / delta_time
    # omega_dot = omega_delta / 0.01

    ### Remove the zero-time indices ###
    omega_meas = omega_meas[:,nonzero_idx[0]].squeeze()
    omega_des = omega_des[:,nonzero_idx[0]].squeeze()

    assert (omega_des.shape == omega_dot.shape)
    assert (omega_meas.shape == omega_dot.shape)

    return omega_meas, omega_des, omega_dot

def compile_all_data(logs_dir, drop_idx_from_end):
    """
    Compiling all the data from dc_log_folders.
    """
    dc_log_folders = find_all_folders(logs_dir)
    # dc_log_folders = ["/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/data_collection/0610_173119-DC-Sim-Good"]

    omega_meas_logs = []
    omega_des_logs = []
    omega_dot_logs = []

    for folder_path in dc_log_folders:
        state_log, input_log, error_log, omega_log, omega_timestamp_log, params = load_data(folder_path, drop_idx_from_end)
        omega_meas, omega_des, omega_dot = get_ang_values(input_log, omega_log, omega_timestamp_log)
        omega_meas_logs.append(omega_meas.T)
        omega_des_logs.append(omega_des.T)
        omega_dot_logs.append(omega_dot.T)

    omega_meas_logs = np.concatenate(omega_meas_logs, axis=0)
    omega_des_logs = np.concatenate(omega_des_logs, axis=0)
    omega_dot_logs = np.concatenate(omega_dot_logs, axis=0)

    return omega_meas_logs, omega_des_logs, omega_dot_logs

def get_big_ang_values(omega_meas_logs, omega_des_logs, omega_dot_logs, thres_mag=1):
    """
    Get the angular values that are above a certain magnitude.
    """
    mag = np.linalg.norm(omega_dot_logs[:,:2], axis=1)
    idx = np.where(mag >= thres_mag)[0]

    omega_meas_logs = omega_meas_logs[idx]
    omega_des_logs = omega_des_logs[idx]
    omega_dot_logs = omega_dot_logs[idx]

    return omega_meas_logs, omega_des_logs, omega_dot_logs

def lstsq_fit(omega_meas_logs, omega_des_logs, omega_dot_logs):
    """
    All the input arrays are of shape (X,3).
    Fit a line to the data.
    """
    N = omega_meas_logs.shape[0]
    X = omega_des_logs - omega_meas_logs    # (N,3)
    Y = omega_dot_logs                      # (N,3)

    C, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    Y_pred = X @ C

    Y_mean = np.mean(Y, axis=0)
    SS_tot = np.sum((Y - Y_mean) ** 2, axis=0)
    SS_res = np.sum((Y - Y_pred) ** 2, axis=0)
    R2 = 1 - (SS_res / SS_tot)
    
    IPython.embed()

    return C, R2

def lstsq_fit_xyz(omega_meas_logs, omega_des_logs, omega_dot_logs):
    """
    Fit a line to the data for x,y,z separately.
    """

    def lstsq_fit_one(omega_meas, omega_des, omega_dot):
        C, residuals, _, _ = np.linalg.lstsq(omega_des - omega_meas, omega_dot, rcond=-1)
        Y_pred = (omega_des - omega_meas) @ C
        Y_mean = np.mean(omega_dot)
        SS_tot = np.sum((omega_dot - Y_mean) ** 2)
        SS_res = np.sum((omega_dot - Y_pred) ** 2)
        R2 = 1 - (SS_res / SS_tot)
        return C, R2

    C_list = []
    R2_list = []
    for i in range(3):
        omega_meas = omega_meas_logs[:,i][:,np.newaxis]
        omega_des = omega_des_logs[:,i][:,np.newaxis]
        omega_dot = omega_dot_logs[:,i][:,np.newaxis]
        C, R2 = lstsq_fit_one(omega_meas, omega_des, omega_dot)
        C_list.append(C)
        R2_list.append(R2)
    
    IPython.embed()

    return C_list, R2_list

# def affine_fit_xyz(omega_meas_logs, omega_des_logs, omega_dot_logs):
#     """
#     Fit a line to the data for x,y,z separately.
#     """

#     def affine_fit_one(omega_meas, omega_des, omega_dot):
#         gain = np.std(omega_dot) / np.std(omega_des - omega_meas)
#         offset = np.mean((omega_des - omega_meas) - (omega_dot / gain))
#         C = [gain, offset]
#         Y_pred = ((omega_des - omega_meas) - offset) * gain
#         Y_mean = np.mean(omega_dot)
#         SS_tot = np.sum((omega_dot - Y_mean) ** 2)
#         SS_res = np.sum((omega_dot - Y_pred) ** 2)
#         R2 = 1 - (SS_res / SS_tot)
#         return C, R2

#     C_list = []
#     R2_list = []
#     for i in range(3):
#         omega_meas = omega_meas_logs[:,i][:,np.newaxis]
#         omega_des = omega_des_logs[:,i][:,np.newaxis]
#         omega_dot = omega_dot_logs[:,i][:,np.newaxis]
#         C, R2 = affine_fit_one(omega_meas, omega_des, omega_dot)
#         C_list.append(C)
#         R2_list.append(R2)
    
#     IPython.embed()

#     return C_list, R2_list

def const_fit(omega_meas_logs, omega_des_logs, omega_dot_logs, const=100):
    """
    All the input arrays are of shape (X,3).
    Fit a line to the data.
    """
    N = omega_meas_logs.shape[0]
    X = omega_des_logs - omega_meas_logs    # (N,3)
    Y = omega_dot_logs                      # (N,3)

    A = np.zeros((N*3, 9))
    for i in range(N):
        A[3*i:3*(i+1),:] = np.kron(np.eye(3), X[i,:])

    Y_flat = Y.flatten()

    C = const * np.eye(3)
    C_flat = C.reshape((-1,1)).squeeze()

    Y_pred_flat = A @ C_flat
    Y_pred = Y_pred_flat.reshape(N, 3)
    
    # Step 4: Compute R^2 value
    Y_mean = np.mean(Y_flat)
    SS_tot = np.sum((Y_flat - Y_mean) ** 2)
    SS_res = np.sum((Y_flat - Y_pred_flat) ** 2)
    R2 = 1 - (SS_res / SS_tot)[:,np.newaxis]
    
    return C, R2

def poly_fit_xyz(omega_meas_logs, omega_des_logs, omega_dot_logs, deg=2):
    """
    With degree 2, we get:
    In [3]: p_list
    Out[3]: 
    [array([-0.07193628, 11.91001465,  0.04944787]),
    array([ 0.03688765, 14.28639733,  0.10458604]),
    array([-4.52841938e-03,  7.28431729e+00,  9.17177026e-03])]

    In [1]: residuals_list
    Out[1]: [array([8.42152336e+08]), array([1.03179062e+09]), array([27732998.32005037])]
    """

    def poly_fit_one(omega_meas, omega_des, omega_dot, deg=2):
        p, residuals, rank, singular_values, rcond = np.polyfit(omega_des - omega_meas, omega_dot, deg=deg, full=True)

        return p, residuals

    p_list = []
    residuals_list = []
    for i in range(3):
        omega_meas = omega_meas_logs[:,i]
        omega_des = omega_des_logs[:,i]
        omega_dot = omega_dot_logs[:,i]
        p, residuals = poly_fit_one(omega_meas, omega_des, omega_dot, deg=deg)
        p_list.append(p)
        residuals_list.append(residuals)
    
    IPython.embed()

    return p_list, residuals_list

def plot_data(omega_meas_logs, omega_des_logs, omega_dot_logs):
    """
    Plot the data.
    """
    fig, axs = plt.subplots(3, 3, figsize=(15,15))
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data files.')
    parser.add_argument('--logs_dir', type=str, default='/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/data_collection',
                        help='Path to the directory containing data collection log folders')
    parser.add_argument('--drop_idx_from_end', type=int, default=None)
    args = parser.parse_args()

    logs_dir = args.logs_dir
    # drop_idx_from_end = args.drop_idx_from_end
    drop_idx_from_end = 1000

    omega_meas_logs, omega_des_logs, omega_dot_logs = compile_all_data(logs_dir, drop_idx_from_end)

    # omega_meas_logs, omega_des_logs, omega_dot_logs = get_big_ang_values(omega_meas_logs, omega_des_logs, omega_dot_logs, thres_mag=1)

    # C_lst, R2_lst = lstsq_fit_xyz(omega_meas_logs, omega_des_logs, omega_dot_logs)
    # C_lst, R2_lst = affine_fit_xyz(omega_meas_logs, omega_des_logs, omega_dot_logs)
    # C_lst, R2_lst = lstsq_fit(omega_meas_logs[:,:-1], omega_des_logs[:,:-1], omega_dot_logs[:,:-1])
    # C_lst, R2_lst = lstsq_fit(omega_meas_logs, omega_des_logs, omega_dot_logs)
    # C_const, R2_const = const_fit(omega_meas_logs, omega_des_logs, omega_dot_logs, const=50)
    C_poly, R2_poly = poly_fit_xyz(omega_meas_logs, omega_des_logs, omega_dot_logs, deg=3)
    # plot_data(omega_meas_logs, omega_des_logs, omega_dot_logs)

    IPython.embed()