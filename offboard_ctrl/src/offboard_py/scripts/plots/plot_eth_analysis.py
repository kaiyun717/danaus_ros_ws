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
    logs_dir = '/home/dvij/ncbf-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/eth_lqr'
    
    folder_name = "0411_234800-Real"
    folder_path = os.path.join(logs_dir, folder_name)

    state_log = np.load(os.path.join(folder_path, 'state.npy'))
    input_log = np.load(os.path.join(folder_path, 'input.npy'))
    error_log = np.load(os.path.join(folder_path, 'error.npy'))

    # Time
    drop_idx = 15*90
    time = np.arange(state_log.shape[1])  # Assuming time steps are along columns
    time = time/90
    
    # IPython.embed()

    ################# PLOT STATES #################
    # Task 1: Plot the XYZ errors for all three logs

    # XYZ error, pend RS error, pend RS dot error

    xyz_errors = error_log[0:3,:drop_idx]
    avg_error_xyz = np.mean(np.square(xyz_errors), axis=0)
    avg_error_xyz = np.mean(avg_error_xyz)

    pend_rs_errors = error_log[9:11,:drop_idx]
    avg_error_pend_rs = np.mean(np.square(pend_rs_errors), axis=0)
    avg_error_pend_rs = np.mean(avg_error_pend_rs)

    pend_rs_dot_errors = error_log[11:13,:drop_idx]
    avg_error_pend_rs_dot = np.mean(np.square(pend_rs_dot_errors), axis=0)
    avg_error_pend_rs_dot = np.mean(avg_error_pend_rs_dot)

    # IPython.embed()

    plt.figure(figsize=(10, 5))
    plt.plot(time[:drop_idx], pend_rs_dot_errors[0,:], label="$\dot{r}_{e}$")
    plt.plot(time[:drop_idx], pend_rs_dot_errors[1,:], label="$\dot{s}_{e}$")
    plt.xlabel('Time (sec)', fontsize=24)
    plt.ylabel('Error (m)', fontsize=24)
    plt.title('Errors in Pendulum Velocities', fontsize=24)
    plt.legend(fontsize=20, framealpha=1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_folder, 'eth_lqr-rsDotErrors')
    plt.savefig(save_path)
    print(avg_error_xyz, avg_error_pend_rs, avg_error_pend_rs_dot)
