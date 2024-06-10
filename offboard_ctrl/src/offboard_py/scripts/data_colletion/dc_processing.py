import IPython
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def find_all_folders(logs_dir):
    folders = os.listdir(logs_dir)
    folders = [folder for folder in folders if os.path.isdir(os.path.join(logs_dir, folder))]
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
        state_log = state_log[:, 1000:-drop_idx_from_end]
        input_log = input_log[:, 1000:-drop_idx_from_end]
        error_log = error_log[:, 1000:-drop_idx_from_end]
        omega_log = omega_log[:, 1000:-drop_idx_from_end]
        omega_timestamp_log = omega_timestamp_log[:, 1000:-drop_idx_from_end]
        
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
    # dc_log_folders = find_all_folders(logs_dir)
    dc_log_folders = ["/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/data_collection/0610_133851-DC-Sim-Good"]

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

def lstsq_fit(omega_meas_logs, omega_des_logs, omega_dot_logs):
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

    C_flat, residuals, _, _ = np.linalg.lstsq(A, Y_flat, rcond=None)
    C = C_flat.reshape((3,3))

    Y_pred_flat = A @ C_flat
    Y_pred = Y_pred_flat.reshape(N, 3)
    
    # Step 4: Compute R^2 value
    Y_mean = np.mean(Y_flat)
    SS_tot = np.sum((Y_flat - Y_mean) ** 2)
    SS_res = np.sum((Y_flat - Y_pred_flat) ** 2)
    R2 = 1 - (SS_res / SS_tot)

    # IPython.embed()
    
    return C, R2

def lstsq_fit_one(omega_meas_logs, omega_des_logs, omega_dot_logs):
    """
    All the input arrays are of shape (N,1).
    Fit a line to the data.
    """
    N = omega_meas_logs.shape[0]
    X = omega_des_logs - omega_meas_logs    # (N,1)
    Y = omega_dot_logs                      # (N,1)

    C_flat, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    # C = C_flat.reshape((3,3))

    Y_pred_flat = X @ C_flat
    
    # Step 4: Compute R^2 value
    Y_mean = np.mean(Y)
    SS_tot = np.sum((Y - Y_mean) ** 2)
    SS_res = np.sum((Y - Y_pred_flat) ** 2)
    R2 = 1 - (SS_res / SS_tot)

    # IPython.embed()
    
    return C_flat, R2

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

def poly_fit(omega_meas_logs, omega_des_logs, omega_dot_logs, deg=2):
    N = omega_meas_logs.shape[0]
    X = omega_des_logs - omega_meas_logs    # (N,3)
    Y = omega_dot_logs                      # (N,3)

    poly = PolynomialFeatures(degree=deg)
    X_poly = poly.fit_transform(X)

    C_matrices = []
    R2_values = []

    for i in range(3):
        Y_i = Y[:,i]
        reg = LinearRegression().fit(X_poly, Y_i)
        C = reg.coef_
        R2 = reg.score(X_poly, Y_i)
        C_matrices.append(C)
        R2_values.append(R2)
    
    C = np.array(C_matrices)
    R2 = np.array(R2_values)

    return C, R2


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

    C_lst, R2_lst = lstsq_fit_one(omega_meas_logs[:,0][:,np.newaxis], omega_des_logs[:,0][:,np.newaxis], omega_dot_logs[:,0][:,np.newaxis])
    # C_lst, R2_lst = lstsq_fit(omega_meas_logs, omega_des_logs, omega_dot_logs)
    # C_const, R2_const = const_fit(omega_meas_logs, omega_des_logs, omega_dot_logs, const=50)

    IPython.embed()