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

def load_datas(folder_paths, drop_idx_from_end=None):
    state_logs = []
    input_logs = []
    error_logs = []
    omega_logs = []
    omega_timestamp_logs = []
    params = []
    for folder_path in folder_paths:
        state_log, input_log, error_log, omega_log, omega_timestamp_log, param = load_data(folder_path)

        if drop_idx_from_end is not None:
            state_log = state_log[:, :-drop_idx_from_end]
            input_log = input_log[:, :-drop_idx_from_end]
            error_log = error_log[:, :-drop_idx_from_end]
            omega_log = omega_log[:, :-drop_idx_from_end]
            omega_timestamp_log = omega_timestamp_log[:, :-drop_idx_from_end]
        
        state_logs.append(state_log.T)
        input_logs.append(input_log.T)
        error_logs.append(error_log.T)
        omega_logs.append(omega_log.T)
        omega_timestamp_logs.append(omega_timestamp_log.T)

    state_logs = np.concatenate(state_logs, axis=0)
    input_logs = np.concatenate(input_logs, axis=0)
    error_logs = np.concatenate(error_logs, axis=0)
    omega_logs = np.concatenate(omega_logs, axis=0)
    omega_timestamp_logs = np.concatenate(omega_timestamp_logs, axis=0)

    return state_logs, input_logs, error_logs, omega_logs, omega_timestamp_logs, params

def get_ang_values(state_log, input_log, omega_log, omega_timestamp_log):
    """
    Preprocess the data for angular values.
    """
    omega_meas = omega_log       # gamma, beta, alpha
    omega_des = input_log[:,1:]  # wx, wy, wz
    
    omega_delta = omega_meas[1:,:] - omega_meas[:-1,:]
    delta_time = omega_timestamp_log[1:] - omega_timestamp_log[:-1]
    omega_dot = omega_delta / delta_time

    assert (omega_des[:-1,:].shape == omega_dot.shape)
    assert (omega_meas[:-1,:].shape == omega_dot.shape)

    return omega_meas[:-1,:], omega_des[:-1,:], omega_dot




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data files.')
    parser.add_argument('--logs_dir', type=str, default='/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/data_collection',
                        help='Path to the directory containing data collection log folders')
    parser.add_argument('--drop_idx_from_end', type=int, default=None)
    args = parser.parse_args()

    logs_dir = args.logs_dir
    drop_idx_from_end = args.drop_idx_from_end

    dc_log_folders = find_all_folders(logs_dir)
    state_logs, input_logs, error_logs, omega_logs, omega_timestamp_logs, params = load_datas(dc_log_folders, drop_idx_from_end)
    IPython.embed()