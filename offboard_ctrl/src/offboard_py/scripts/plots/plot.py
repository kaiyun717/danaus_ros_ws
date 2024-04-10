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
    parser.add_argument('--logs_dir', type=str, default='/home/oem/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs',
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
    input_path = os.path.join(folder_path, 'input.npy')
    error_path = os.path.join(folder_path, 'error.npy')

    state_log = np.load(state_path)
    input_log = np.load(input_path)
    error_log = np.load(error_path)

    # Time
    time = np.arange(state_log.shape[1])  # Assuming time steps are along columns
    time = time/90  # Assuming 50 Hz sampling rate
    
    # Extract variables from state_log
    xyz = state_log[0:3, :500]
    xyz_dot = state_log[3:6, :500]
    abc = state_log[6:9, :500]
    rs = state_log[9:11, :500]
    rs_dot = state_log[11:13, :500]

    # Extract variables from error_log
    e_xyz = error_log[0:3, :500]
    e_xyz_dot = error_log[3:6, :500]
    e_abc = error_log[6:9, :500]
    e_rs = error_log[9:11, :500]
    e_rs_dot = error_log[11:13, :500]

    # Extract variables from input_log
    angular_velocities = input_log[0:3, :500]
    acceleration = input_log[3, :500]

    #####################################
    ############ STATE PLOTS ############
    #####################################

    # Plot and save figures
    plt.figure(figsize=(16, 12))

    # Plot [x, y, z] vs time
    plt.subplot(3, 2, 1)
    plt.plot(time[: 500], xyz[0, :], label='x')
    plt.plot(time[: 500], xyz[1, :], label='y')
    plt.plot(time[: 500], xyz[2, :], label='z')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Quad Pos. vs Time')
    plt.legend()

    # Plot [x_dot, y_dot, z_dot] vs time
    plt.subplot(3, 2, 2)
    plt.plot(time[:500], xyz_dot[0, :], label='x_dot')
    plt.plot(time[:500], xyz_dot[1, :], label='y_dot')
    plt.plot(time[:500], xyz_dot[2, :], label='z_dot')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Quad Vel. vs Time')
    plt.legend()

    # Plot [α, β, γ] vs time
    plt.subplot(3, 2, 3)
    plt.plot(time[:500], abc[0, :], label='alpha')
    plt.plot(time[:500], abc[1, :], label='beta')
    plt.plot(time[:500], abc[2, :], label='gamma')
    plt.xlabel('Time')
    plt.ylabel('Orientation')
    plt.title('Quad Orient. vs Time')
    plt.legend()

    # Plot [r, s] vs time
    plt.subplot(3, 2, 4)
    plt.plot(time[:500], rs[0, :], label='r')
    plt.plot(time[:500], rs[1, :], label='s')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Pole Pos. vs Time')
    plt.legend()

    # Plot [r_dot, s_dot] vs time
    plt.subplot(3, 2, 5)
    plt.plot(time[:500], rs_dot[0, :], label='r_dot')
    plt.plot(time[:500], rs_dot[1, :], label='s_dot')
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
    plt.plot(time[:500], e_xyz[0, :], label='x_e')
    plt.plot(time[:500], e_xyz[1, :], label='y_e')
    plt.plot(time[:500], e_xyz[2, :], label='z_e')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Error Quad Pos. vs Time')
    plt.legend()

    # Plot ERROR [x_dot, y_dot, z_dot] vs time
    plt.subplot(3, 2, 2)
    plt.plot(time[:500], e_xyz_dot[0, :], label='x_dot_e')
    plt.plot(time[:500], e_xyz_dot[1, :], label='y_dot_e')
    plt.plot(time[:500], e_xyz_dot[2, :], label='z_dot_e')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Error Quad Vel. vs Time')
    plt.legend()

    # Plot ERROR [α, β, γ] vs time
    plt.subplot(3, 2, 3)
    plt.plot(time[:500], e_abc[0, :], label='alpha_e')
    plt.plot(time[:500], e_abc[1, :], label='beta_e')
    plt.plot(time[:500], e_abc[2, :], label='gamma_e')
    plt.xlabel('Time')
    plt.ylabel('Orientation')
    plt.title('Error Quad Orient. vs Time')
    plt.legend()

    # Plot ERROR [r, s] vs time
    plt.subplot(3, 2, 4)
    plt.plot(time[:500], e_rs[0, :], label='r_e')
    plt.plot(time[:500], e_rs[1, :], label='s_e')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Error Pole Pos. vs Time')
    plt.legend()

    # Plot ERROR [r_dot, s_dot] vs time
    plt.subplot(3, 2, 5)
    plt.plot(time[:500], e_rs_dot[0, :], label='r_dot_e')
    plt.plot(time[:500], e_rs_dot[1, :], label='s_dot_e')
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

    # Plot and save figures
    plt.figure(figsize=(16, 12))

    # Plot [wx, wy, wz] vs time
    plt.subplot(2, 1, 1)
    plt.plot(time[:500], angular_velocities[0, :], label='wx')
    plt.plot(time[:500], angular_velocities[1, :], label='wy')
    plt.plot(time[:500], angular_velocities[2, :], label='wz')
    plt.xlabel('Time')
    plt.ylabel('Angular Rate')
    plt.title('Input Angular Rate vs Time')
    plt.legend()

    # Plot [a] vs time
    plt.subplot(2, 1, 2)
    plt.plot(time[:500], acceleration, label='a')
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
