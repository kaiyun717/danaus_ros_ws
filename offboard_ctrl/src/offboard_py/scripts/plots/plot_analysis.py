import IPython
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def find_latest_folder(logs_dir):
    folders = os.listdir(logs_dir)
    folders = [folder for folder in folders if os.path.isdir(os.path.join(logs_dir, folder))]
    folders.sort(reverse=True)
    if folders:
        return folders[0]
    else:
        return None
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data files.')
    parser.add_argument('--logs_dir', type=str, default='/home/dvij/ncbf-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/eth_lqr',
                        help='Path to the directory containing log folders')
    parser.add_argument('--folder_name', type=str, default=None,
                        help='Name of the folder in which data is stored. If not provided, the script will use the latest folder.')
    args = parser.parse_args()

    logs_dir = args.logs_dir

    if args.folder_name is None:
        folder_name = find_latest_folder(logs_dir)
        if folder_name is None:
            print("No data folders found.")
    else:
        folder_name = args.folder_name

    # IPython.embed()

    folder_path = os.path.join(logs_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_name}' does not exist.")

    state_path = os.path.join(folder_path, 'state.npy')
    input_path = os.path.join(folder_path, 'input.npy')
    error_path = os.path.join(folder_path, 'error.npy')

    state_log = np.load(state_path)
    input_log = np.load(input_path)
    error_log = np.load(error_path)

    # Time
    time = np.arange(state_log.shape[1])  # Assuming time steps are along columns
    time = time/90  # Assuming 50 Hz sampling rate

    e_rs_norm = np.linalg.norm(error_log[9:11, :], axis=0)
    # print(e_rs_norm) 
    
    # drop_idx = np.where(e_rs_norm > 0.3)
    drop_idx = 15*90
    
    # Extract variables from state_log
    xyz = state_log[0:3, :drop_idx]
    xyz_dot = state_log[3:6, :drop_idx]
    abc = state_log[6:9, :drop_idx]
    rs = state_log[9:11, :drop_idx]
    rs_dot = state_log[11:13, :drop_idx]

    # Extract variables from error_log
    e_xyz = error_log[0:3, :drop_idx]
    e_xyz_dot = error_log[3:6, :drop_idx]
    e_abc = error_log[6:9, :drop_idx]
    e_rs = error_log[9:11, :drop_idx]
    e_rs_dot = error_log[11:13, :drop_idx]

    # Extract variables from input_log
    angular_velocities = input_log[0:3, :drop_idx]
    acceleration = input_log[3, :drop_idx]

    # IPython.embed()

    pitch_roll = abc[1:,:]
    pitch = pitch_roll[0,:]
    roll = pitch_roll[1,:]

    xi = np.sqrt(0.69**2 - rs[0,:]**2 - rs[1,:]**2)
    cos_cos = xi/0.69
    delta_p = np.arccos(cos_cos)

    sqrt_sum_sqrd = np.sqrt(pitch**2 + roll**2 + delta_p**2)

    # Plot and save figures
    plt.figure(figsize=(8, 6))
    plt.plot(time[: drop_idx], np.rad2deg(pitch), label='pitch', linestyle='--')
    plt.plot(time[: drop_idx], np.rad2deg(roll), label='roll', linestyle='--')
    plt.plot(time[: drop_idx], np.rad2deg(delta_p), label='pend angle')
    plt.plot(time[: drop_idx], np.rad2deg(sqrt_sum_sqrd), label='sqrt_sum_sqrd')
    
    max_idx = np.argmax(sqrt_sum_sqrd)
    plt.axvline(x=max_idx/90, color='k', linestyle='--', label='max')
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Angle (deg)', fontsize=14)
    plt.title('Angles vs. Time', fontsize=14)
    plt.legend(fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(folder_path, 'angles')
    plt.savefig(save_path)

    # Plot and save figures
    plt.figure(figsize=(8, 6))
    abc_dot = (abc[1:, 1:] - abc[1:, :-1]) / (1/90)
    pend_pitch = np.arctan2(xi, rs[0,:])
    pend_roll = np.arctan2(xi, rs[1,:])
    pend_pitch_dot = rs_dot[0,:] / xi
    pend_roll_dot = rs_dot[1,:] / xi

    plt.plot(time[930:960], np.rad2deg(abc_dot[0,930:960]), label='pitchDot', linestyle='--')
    plt.plot(time[930:960], np.rad2deg(abc_dot[1,930:960]), label='rollDot', linestyle='--')
    # plt.plot(time[930:960], np.rad2deg(delta_p[930:960]), label='pend angle', linewidth=2)
    plt.plot(time[930:960], np.rad2deg(pend_pitch_dot[930:960]), label='pend pitchDot')
    plt.plot(time[930:960], np.rad2deg(pend_roll_dot[930:960]), label='pend rollDot')
    
    max_idx = np.argmax(abc_dot)
    plt.axvline(x=max_idx/90, color='k', linestyle='--', label='max')
    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Angular Velocity (deg/s)', fontsize=14)
    plt.title('Anglular Velocity vs. Time', fontsize=14)
    plt.legend(fontsize=14)

    plt.tight_layout()
    save_path = os.path.join(folder_path, 'angular_velocities')
    plt.savefig(save_path)
