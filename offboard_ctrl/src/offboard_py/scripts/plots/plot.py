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
    parser.add_argument('--logs_dir', type=str, default='/home/cdundun/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs',
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
    time = time/50  # Assuming 50 Hz sampling rate

    e_rs_norm = np.linalg.norm(error_log[9:11, :], axis=0)
    # print(e_rs_norm) 
    
    # drop_idx = np.where(e_rs_norm > 0.3)
    drop_idx = 5*50
    
    # Extract variables from state_log
    xyz = state_log[10:13, :drop_idx]
    xyz_dot = state_log[13:16, :drop_idx]
    euler_ang = state_log[0:3, :drop_idx]
    euler_ang_vel = state_log[3:6, :drop_idx]
    pend_rp = state_log[6:8, :drop_idx]
    pend_rp_vel = state_log[8:10, :drop_idx]

    # Extract variables from error_log
    e_xyz = error_log[10:13, :drop_idx]
    e_xyz_dot = error_log[13:16, :drop_idx]
    e_euler_ang = error_log[0:3, :drop_idx]
    e_euler_ang_vel = error_log[3:6, :drop_idx]
    e_pend_rp = error_log[6:8, :drop_idx]
    e_pend_rp_vel = error_log[8:10, :drop_idx]

    # Extract variables from input_log
    angular_velocities = input_log[1:4, :drop_idx]
    acceleration = input_log[0, :drop_idx]

    #####################################
    ############ STATE PLOTS ############
    #####################################

    # Plot and save figures
    plt.figure(figsize=(16, 12))

    # Plot [x, y, z] vs time
    plt.subplot(3, 2, 1)
    plt.plot(time[: drop_idx], xyz[0, :], label='$x$')
    plt.plot(time[: drop_idx], xyz[1, :], label='$y$')
    plt.plot(time[: drop_idx], xyz[2, :], label='$z$')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Quad Pos. vs Time')
    plt.legend()

    # Plot [x_dot, y_dot, z_dot] vs time
    plt.subplot(3, 2, 2)
    plt.plot(time[:drop_idx], xyz_dot[0, :], label='$\dot x$')
    plt.plot(time[:drop_idx], xyz_dot[1, :], label='$\dot y$')
    plt.plot(time[:drop_idx], xyz_dot[2, :], label='$\dot z$')
    plt.xlabel('Time')
    plt.ylabel('Velocity')
    plt.title('Quad Vel. vs Time')
    plt.legend()

    # Plot [α, β, γ] vs time
    plt.subplot(3, 2, 3)
    plt.plot(time[:drop_idx], euler_ang[2, :], label='$\\alpha$')
    plt.plot(time[:drop_idx], euler_ang[1, :], label='$\\beta$')
    plt.plot(time[:drop_idx], euler_ang[0, :], label='$\gamma$')
    plt.xlabel('Time')
    plt.ylabel('Orientation')
    plt.title('Quad Orient. vs Time')
    plt.legend()

    # Plot [α_dot, β_dot, γ_dot] vs time
    plt.subplot(3, 2, 4)
    plt.plot(time[:drop_idx], euler_ang_vel[2, :], label='$\dot \\alpha$')
    plt.plot(time[:drop_idx], euler_ang_vel[1, :], label='$\dot \\beta$')
    plt.plot(time[:drop_idx], euler_ang_vel[0, :], label='$\dot \gamma$')
    plt.xlabel('Time')
    plt.ylabel('Orientation Rate')
    plt.title('Quad Orient. Rate vs Time')
    plt.legend()

    # Plot [r, s] vs time
    plt.subplot(3, 2, 5)
    plt.plot(time[:drop_idx], pend_rp[0, :], label='$\phi$')
    plt.plot(time[:drop_idx], pend_rp[1, :], label='$\\theta$')
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Pole Pos. vs Time')
    plt.legend()

    # Plot [r_dot, s_dot] vs time
    plt.subplot(3, 2, 6)
    plt.plot(time[:drop_idx], pend_rp_vel[0, :], label='$\dot \phi$')
    plt.plot(time[:drop_idx], pend_rp_vel[1, :], label='$\dot \\theta$')
    plt.xlabel('Time')
    plt.ylabel('Velocities')
    plt.title('Pole Vel. vs Time')
    plt.legend()

    # Adjust layout
    plt.tight_layout()

    # Save plots
    save_path = os.path.join(folder_path, 'states')
    plt.savefig(save_path)

    # #####################################
    # ############ ERROR PLOTS ############
    # #####################################

    # # Plot and save figures
    # plt.figure(figsize=(16, 12))

    # # Plot ERROR [x, y, z] vs time
    # plt.subplot(3, 2, 1)
    # plt.plot(time[:drop_idx], e_xyz[0, :], label='x_e')
    # plt.plot(time[:drop_idx], e_xyz[1, :], label='y_e')
    # plt.plot(time[:drop_idx], e_xyz[2, :], label='z_e')
    # plt.xlabel('Time')
    # plt.ylabel('Position')
    # plt.title('Error Quad Pos. vs Time')
    # plt.legend()

    # # Plot ERROR [x_dot, y_dot, z_dot] vs time
    # plt.subplot(3, 2, 2)
    # plt.plot(time[:drop_idx], e_xyz_dot[0, :], label='x_dot_e')
    # plt.plot(time[:drop_idx], e_xyz_dot[1, :], label='y_dot_e')
    # plt.plot(time[:drop_idx], e_xyz_dot[2, :], label='z_dot_e')
    # plt.xlabel('Time')
    # plt.ylabel('Velocity')
    # plt.title('Error Quad Vel. vs Time')
    # plt.legend()

    # # Plot ERROR [α, β, γ] vs time
    # plt.subplot(3, 2, 3)
    # plt.plot(time[:drop_idx], e_abc[0, :], label='alpha_e')
    # plt.plot(time[:drop_idx], e_abc[1, :], label='beta_e')
    # plt.plot(time[:drop_idx], e_abc[2, :], label='gamma_e')
    # plt.xlabel('Time')
    # plt.ylabel('Orientation')
    # plt.title('Error Quad Orient. vs Time')
    # plt.legend()

    # # Plot ERROR [r, s] vs time
    # plt.subplot(3, 2, 4)
    # plt.plot(time[:drop_idx], e_rs[0, :], label='r_e')
    # plt.plot(time[:drop_idx], e_rs[1, :], label='s_e')
    # plt.xlabel('Time')
    # plt.ylabel('Position')
    # plt.title('Error Pole Pos. vs Time')
    # plt.legend()

    # # Plot ERROR [r_dot, s_dot] vs time
    # plt.subplot(3, 2, 5)
    # plt.plot(time[:drop_idx], e_rs_dot[0, :], label='r_dot_e')
    # plt.plot(time[:drop_idx], e_rs_dot[1, :], label='s_dot_e')
    # plt.xlabel('Time')
    # plt.ylabel('Velocities')
    # plt.title('Error Pole Vel. vs Time')
    # plt.legend()

    # # Adjust layout
    # plt.tight_layout()

    # # Save plots
    # save_path = os.path.join(folder_path, 'errors')
    # plt.savefig(save_path)

    #####################################
    ############ INPUT PLOTS ############
    #####################################

    # Plot and save figures
    plt.figure(figsize=(16, 12))

    # Plot [wx, wy, wz] vs time
    plt.subplot(2, 1, 1)
    plt.plot(time[:drop_idx], angular_velocities[0, :], label='$\omega_x$')
    plt.plot(time[:drop_idx], angular_velocities[1, :], label='$\omega_y$')
    plt.plot(time[:drop_idx], angular_velocities[2, :], label='$\omega_z$')
    plt.xlabel('Time')
    plt.ylabel('Angular Rate')
    plt.title('Input Angular Rate vs Time')
    plt.legend()

    # Plot [a] vs time
    plt.subplot(2, 1, 2)
    plt.plot(time[:drop_idx], acceleration, label='a')
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
