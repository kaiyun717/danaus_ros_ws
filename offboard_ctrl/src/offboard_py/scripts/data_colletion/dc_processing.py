import IPython
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


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
        state_log = state_log[:, :-drop_idx_from_end]
        input_log = input_log[:, :-drop_idx_from_end]
        error_log = error_log[:, :-drop_idx_from_end]
        omega_log = omega_log[:, :-drop_idx_from_end]
        omega_timestamp_log = omega_timestamp_log[:, :-drop_idx_from_end]
        
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
    omega_delta = omega_meas[:,1:] - omega_meas[:,:-1]
    delta_time = omega_timestamp_log[:,1:] - omega_timestamp_log[:,:-1]
    ### Find the nonzero-time indices ###
    nonzero_idx = np.nonzero(delta_time.squeeze())  # This is from the 0 index of `delta_time`, which corresponds to the 0 index of other arrays.
    
    ### Remove the zero-time indices ###
    omega_meas = omega_meas[:,nonzero_idx].squeeze()
    omega_des = omega_des[:,nonzero_idx].squeeze()
    omega_delta = omega_delta[:,nonzero_idx].squeeze()
    delta_time = delta_time[:,nonzero_idx].squeeze()

    omega_dot = omega_delta / delta_time

    assert (omega_des.shape == omega_dot.shape)
    assert (omega_meas.shape == omega_dot.shape)

    return omega_meas, omega_des, omega_dot

def compile_all_data(logs_dir, drop_idx_from_end):
    """
    Compiling all the data from dc_log_folders.
    """
    dc_log_folders = find_all_folders(logs_dir)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data files.')
    parser.add_argument('--logs_dir', type=str, default='/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/data_collection',
                        help='Path to the directory containing data collection log folders')
    parser.add_argument('--drop_idx_from_end', type=int, default=None)
    args = parser.parse_args()

    logs_dir = args.logs_dir
    drop_idx_from_end = args.drop_idx_from_end

    omega_meas_logs, omega_des_logs, omega_dot_logs = compile_all_data(logs_dir, drop_idx_from_end)

    IPython.embed()