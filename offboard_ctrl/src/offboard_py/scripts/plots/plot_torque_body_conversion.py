import IPython
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data files.')
    parser.add_argument('--logs_dir', type=str, default='/home/dvij/ncbf-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/eth_lqr',
                        help='Path to the directory containing log folders')
    parser.add_argument('--folder_name', type=str, default=None,
                        help='Name of the folder in which data is stored. If not provided, the script will use the latest folder.')
    args = parser.parse_args()

    save_folder = '/home/dvij/ncbf-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/psr_final'
    logs_dir = '/home/dvij/ncbf-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs'
    
    folder_names = ["0501_162918-9_to_1_Conversion-Sim", "0501_163158-5_to_5_Conversion-Sim", "0501_163333-1_to_9_Conversion-Sim"]

    folder_paths = []
    for folder_name in folder_names:
        folder_path = os.path.join(logs_dir, folder_name)
        folder_paths.append(folder_path)

    state_paths = [os.path.join(folder_path, 'state.npy') for folder_path in folder_paths]
    input_paths = [os.path.join(folder_path, 'input.npy') for folder_path in folder_paths]
    error_paths = [os.path.join(folder_path, 'error.npy') for folder_path in folder_paths]
    body_rate_paths = [os.path.join(folder_path, 'body_rates.npy') for folder_path in folder_paths]

    state_logs = [np.load(state_path) for state_path in state_paths]
    input_logs = [np.load(input_path) for input_path in input_paths]
    error_logs = [np.load(error_path) for error_path in error_paths]
    body_rate_logs = [np.load(body_rate_path) for body_rate_path in body_rate_paths]

    # Time
    time = np.arange(state_logs[0].shape[1])  # Assuming time steps are along columns
    time = time/90

    # MU values
    mus = [0.9, 0.5, 0.1]
    
    ################# PLOT STATES #################
    # Task 1: Plot the XYZ errors for all three logs

    avg_errors = []

    plt.figure(figsize=(10, 5))
    for i, error_log in enumerate(error_logs):
        xyz_errors = error_log[6:9, :]
        avg_error_xyz = np.mean(np.square(xyz_errors), axis=0)
        avg_errors.append(np.mean(avg_error_xyz))
        plt.plot(time, avg_error_xyz.T, label=f'$\mu$ = {mus[i]}')

    plt.xlabel('Time (sec)', fontsize=24)
    plt.ylabel('MSE XYZ Error (m)', fontsize=24)
    plt.title('MSE. XYZ Errors for Different $\mu$ Values', fontsize=24)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_folder, 'torque_body_conversion-mse_XYZ_errors')
    plt.savefig(save_path)
    print(avg_errors)

    ################# PLOT INPUTS #################
    # Initialize lists to store lags for each log
    lags_list = []
    for i, (body_rate_log, input_log) in enumerate(zip(body_rate_logs, input_logs)):
        # Extract body rates and torques
        body_rates = body_rate_log[0:3, :]
        torques = input_log[1:, :]
        
        # Initialize list to store lags
        lags = []
        
        # Iterate over each axis (wx, wy, wz)
        for j in range(3):
            # Compute cross-correlation
            cross_corr = np.correlate(body_rates[j, :], torques[j, :], mode='full')
            # Find lag (delay)
            delay_index = np.argmax(cross_corr) - len(body_rates[j, :]) + 1
            # Append lag to list
            lags.append(delay_index)
        
        # Append list of lags to main list
        lags_list.append(lags)

    # Convert to NumPy array for easier manipulation
    lags_list = np.array(lags_list)/90  # Convert to seconds

    ######### PLOT TORQUES AND BODY RATES #########
    # Calculate midpoint index
    midpoint_index = state_logs[0].shape[1] // 20

    # Extract data around the midpoint (1/10 timesteps)
    window_size = state_logs[0].shape[1] // 40
    start_index = midpoint_index - window_size // 2
    end_index = midpoint_index + window_size // 2

    # Plot
    plt.figure(figsize=(10, 5))

    colors = ['r', 'g', 'b']
    # Plot torques
    for i, input_log in enumerate(input_logs):
        torques = input_log[1, start_index:end_index]
        plt.plot(time[start_index:end_index], torques.T, label=f'$\\tau_x$ for $\mu$ = {mus[i]}', color=colors[i], linewidth=2.5)

    # Plot body rates
    for i, body_rate_log in enumerate(body_rate_logs):
        body_rates = body_rate_log[0, start_index:end_index]
        plt.plot(time[start_index:end_index], body_rates.T, label=f'$\omega_x$ for $\mu$ = {mus[i]}', linestyle='--', color=colors[i], linewidth=2.5)

    plt.xlabel('Time (sec)', fontsize=24)
    plt.ylabel('Value', fontsize=24)
    plt.title('Overlay of Torques and Body Rates', fontsize=24)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_folder, 'torque_body_conversion-overlay')
    plt.savefig(save_path)
    print(lags_list)
