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
    parser.add_argument('--logs_dir', type=str, default='/home/oem/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/low_sim',
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

    folder_path = os.path.join(logs_dir, folder_name)
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_name}' does not exist.")

    state_path = os.path.join(folder_path, 'state.npy')
    u_nom_path = os.path.join(folder_path, 'u_nom.npy')
    u_safe_path = os.path.join(folder_path, 'u_safe.npy')
    error_path = os.path.join(folder_path, 'error.npy')

    state_log = np.load(state_path)
    u_nom_log = np.load(u_nom_path)
    u_safe_log = np.load(u_safe_path)
    error_log = np.load(error_path)

    # Time
    state_length = state_log.shape[1]
    time = np.arange(state_length)  # Assuming time steps are along columns
    time = time/90  # Assuming 50 Hz sampling rate

    # Extract variables from state_log
    xyz = state_log[0:3, :]
    xyz_dot = state_log[3:6, :]
    abc = state_log[6:9, :]
    rs = state_log[9:11, :]
    rs_dot = state_log[11:13, :]

    # Extract variables from error_log
    e_xyz = error_log[0:3, :]
    e_xyz_dot = error_log[3:6, :]
    e_abc = error_log[6:9, :]
    e_rs = error_log[9:11, :]
    e_rs_dot = error_log[11:13, :]

    # Extract variables from input_log
    u_nom_angular_velocities = u_nom_log[0:3, :]
    u_nom_acceleration = u_nom_log[3, :]
    u_safe_angular_velocities = u_safe_log[0:3, :]
    u_safe_acceleration = u_safe_log[3, :]

    #####################################
    ############ STATE PLOTS ############
    #####################################

    # Plot and save figures
    plt.figure(figsize=(16, 12))

    # Plot [x, y, z] vs time
    plt.subplot(3, 2, 1)
    plt.plot(time[: ], xyz[0, :], label='x')
    plt.plot(time[: ], xyz[1, :], label='y')
    plt.plot(time[: ], xyz[2, :], label='z')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Quad Pos. vs Time')
    plt.legend()

    # Plot [x_dot, y_dot, z_dot] vs time
    plt.subplot(3, 2, 2)
    plt.plot(time[:], xyz_dot[0, :], label='x_dot')
    plt.plot(time[:], xyz_dot[1, :], label='y_dot')
    plt.plot(time[:], xyz_dot[2, :], label='z_dot')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Quad Vel. vs Time')
    plt.legend()

    # Plot [α, β, γ] vs time
    plt.subplot(3, 2, 3)
    plt.plot(time[:], abc[0, :], label='alpha')
    plt.plot(time[:], abc[1, :], label='beta')
    plt.plot(time[:], abc[2, :], label='gamma')
    plt.xlabel('Time')
    plt.ylabel('Orientation')
    plt.title('Quad Orient. vs Time')
    plt.legend()

    # Plot [r, s] vs time
    plt.subplot(3, 2, 4)
    plt.plot(time[:], rs[0, :], label='r')
    plt.plot(time[:], rs[1, :], label='s')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Pole Pos. vs Time')
    plt.legend()

    # Plot [r_dot, s_dot] vs time
    plt.subplot(3, 2, 5)
    plt.plot(time[:], rs_dot[0, :], label='r_dot')
    plt.plot(time[:], rs_dot[1, :], label='s_dot')
    plt.xlabel('Time')
    plt.ylabel('Velocities')
    plt.title('Pole Vel. vs Time')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save plots
    save_path = os.path.join(folder_path, 'states')
    plt.savefig(save_path)

    #####################################
    ############ ERROR PLOTS ############
    #####################################

    # Plot and save figures
    plt.figure(figsize=(16, 12))

    # Plot ERROR [x, y, z] vs time
    plt.subplot(3, 2, 1)
    plt.plot(time[:], e_xyz[0, :], label='x_e')
    plt.plot(time[:], e_xyz[1, :], label='y_e')
    plt.plot(time[:], e_xyz[2, :], label='z_e')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Error Quad Pos. vs Time')
    plt.legend()

    # Plot ERROR [x_dot, y_dot, z_dot] vs time
    plt.subplot(3, 2, 2)
    plt.plot(time[:], e_xyz_dot[0, :], label='x_dot_e')
    plt.plot(time[:], e_xyz_dot[1, :], label='y_dot_e')
    plt.plot(time[:], e_xyz_dot[2, :], label='z_dot_e')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Error Quad Vel. vs Time')
    plt.legend()

    # Plot ERROR [α, β, γ] vs time
    plt.subplot(3, 2, 3)
    plt.plot(time[:], e_abc[0, :], label='alpha_e')
    plt.plot(time[:], e_abc[1, :], label='beta_e')
    plt.plot(time[:], e_abc[2, :], label='gamma_e')
    plt.xlabel('Time')
    plt.ylabel('Orientation')
    plt.title('Error Quad Orient. vs Time')
    plt.legend()

    # Plot ERROR [r, s] vs time
    plt.subplot(3, 2, 4)
    plt.plot(time[:], e_rs[0, :], label='r_e')
    plt.plot(time[:], e_rs[1, :], label='s_e')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Error Pole Pos. vs Time')
    plt.legend()

    # Plot ERROR [r_dot, s_dot] vs time
    plt.subplot(3, 2, 5)
    plt.plot(time[:], e_rs_dot[0, :], label='r_dot_e')
    plt.plot(time[:], e_rs_dot[1, :], label='s_dot_e')
    plt.xlabel('Time')
    plt.ylabel('Velocities')
    plt.title('Error Pole Vel. vs Time')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save plots
    save_path = os.path.join(folder_path, 'errors')
    plt.savefig(save_path)

    #####################################
    ############ INPUT PLOTS ############
    #####################################
    # u_length = state_length - 1
    # time = np.arange(u_length)  # Assuming time steps are along columns
    # time = time/90  # Assuming 50 Hz sampling rate

    # Plot and save figures
    plt.figure(figsize=(16, 12))

    # Plot [wx, wy, wz] vs time
    plt.subplot(2, 1, 1)
    plt.plot(time[:], u_nom_angular_velocities[0, :], linestyle='--', color="blue", label='nom: wx')
    plt.plot(time[:], u_nom_angular_velocities[1, :], linestyle='--', color="red", label='nom: wy')
    plt.plot(time[:], u_nom_angular_velocities[2, :], linestyle='--', color="green", label='nom: wz')
    plt.plot(time[:], u_safe_angular_velocities[0, :], linestyle='-', color="blue", label='safe: wx')
    plt.plot(time[:], u_safe_angular_velocities[1, :], linestyle='-', color="red", label='safe: wy')
    plt.plot(time[:], u_safe_angular_velocities[2, :], linestyle='-', color="green", label='safe: wz')
    plt.xlabel('Time')
    plt.ylabel('Angular Rate')
    plt.title('Input Angular Rate vs Time')
    plt.legend()

    # Plot [a] vs time
    plt.subplot(2, 1, 2)
    plt.plot(time[:], u_nom_acceleration, linestyle='--', label='nom: a')
    plt.plot(time[:], u_safe_acceleration[:], linestyle='-', label='safe: a')
    plt.xlabel('Time')
    plt.ylabel('Thrust')
    plt.title('Input Thrust vs Time')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save plots
    save_path = os.path.join(folder_path, 'inputs')
    plt.savefig(save_path)

    #####################################
    ############ PLOT & SAVE ############
    #####################################

    # # Adjust layout
    # plt.tight_layout()

    # # Save plots
    # save_path = os.path.join(folder_path, 'plots')
    # plt.savefig(save_path)

    # # Show plots
    # plt.show()
