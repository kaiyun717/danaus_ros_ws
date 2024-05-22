import IPython
import os
import argparse
from matplotlib.ticker import AutoLocator
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
    logs_dir = '/home/dvij/ncbf-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/ncbf'
    
    # folder_name = "0502_124637-nCBF-sim"
    folder_name = "0502_131700-nCBF-sim"
    folder_path = os.path.join(logs_dir, folder_name)

    state_log = np.load(os.path.join(folder_path, 'state.npy'))
    nom_input_log = np.load(os.path.join(folder_path, 'nom_input.npy'))
    safe_input_log = np.load(os.path.join(folder_path, 'safe_input.npy'))
    error_log = np.load(os.path.join(folder_path, 'error.npy'))
    status_log = np.load(os.path.join(folder_path, 'status_log.npy'))
    phi_log = np.load(os.path.join(folder_path, 'phi_val_log.npy'))


    # Time
    drop_idx = 2*20
    time = np.arange(state_log.shape[1])  # Assuming time steps are along columns
    time = time/20
    
    # IPython.embed()

    ################# PLOT STATES #################
    # Task 1: Plot the XYZ errors for all three logs

    # XYZ error, pend RS error, pend RS dot error

    xyz_errors = error_log[10:13,:drop_idx]
    avg_error_xyz = np.mean(np.square(xyz_errors), axis=0)

    ang_states = error_log[0:3,:drop_idx]
    avg_state_ang = np.mean(np.square(ang_states), axis=0)

    pend_states = state_log[6:8,:drop_idx]
    avg_state_pend = np.mean(np.square(pend_states), axis=0)

    roll = ang_states[0,:]
    pitch = ang_states[1,:]
    psi = pend_states[0,:]
    theta = pend_states[1,:]
    delta = np.arccos(np.cos(psi)*np.cos(theta))

    rho = roll**2 + pitch**2 + delta**2

    status = status_log[:, :drop_idx]
    status = np.where(status == 2, 0, np.where(status == 1, 1, 2))

    nom_inputs = nom_input_log[1:,:drop_idx]
    safe_inputs = safe_input_log[1:,:drop_idx]
    # IPython.embed()

    # plt.figure(figsize=(10, 5))
    # plt.plot(time[:drop_idx], avg_error_xyz[:], label="MSE XYZ")
    # plt.plot(time[:drop_idx], status[0,:], label="Status", linewidth=2)
    # plt.xlabel('Time (sec)', fontsize=24)
    # plt.ylabel('Error ($m^2$)', fontsize=24)
    # plt.title('Quadrotor Position Error and Safety Status', fontsize=24)
    # plt.legend(fontsize=20, framealpha=1)
    # plt.xticks(fontsize=20)
    # plt.yticks(fontsize=20)
    # plt.grid(True)
    # plt.tight_layout()
    # save_path = os.path.join(save_folder, 'single_point-ncbf-sim-xyz')
    # plt.savefig(save_path)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.plot(time[12:drop_idx], rho[12:], linewidth=2)
    ax1.set_xlabel('Time (sec)', fontsize=22)
    ax1.set_ylabel('$\gamma^2 + \\beta^2 + \delta_p^2$', fontsize=22)
    ax1.axhline(y=(np.pi/4)**2, color='k', linestyle='--')


    ax2 = ax1.twinx()
    ax2.plot(time[12:drop_idx], status[0,12:], label="Status", linewidth=2, color='g', linestyle='--')
    ax2.set_ylabel('Safety Status', fontsize=18)

    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    
    # ax.legend(lines + lines2, labels + labels2, fontsize=18, framealpha=1)
    ax1.tick_params(axis='both', which='major', labelsize=18)
    # ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_yticks([0, 1, 2])
    ax2.set_yticklabels([0, 1, 2], fontsize=18)

    plt.title('Summed Squared Angles and Safety Status', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_folder, 'multi_point-ncbf-sim-pendAng')
    plt.savefig(save_path)

    
    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Time (sec)', fontsize=22)
    ax1.set_ylabel('$\\tau_{x,nom}$', color=color, fontsize=22)
    ax1.plot(time[12:drop_idx], nom_inputs[0,12:], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(-4.5, 4.5)


    # Create a second y-axis on the right side
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('$\\tau_{x,safe}$', color=color, fontsize=22)
    ax2.plot(time[12:drop_idx], safe_inputs[0,12:], color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(-4.5, 4.5)

    # Create a third y-axis on the right side
    ax3 = ax1.twinx()
    color = 'tab:green'
    ax3.spines['right'].set_position(('outward', 60))
    ax3.set_ylabel('Status', color=color, fontsize=22)
    ax3.plot(time[12:drop_idx], status[0,12:], color=color, linestyle='--', linewidth=2)
    ax3.tick_params(axis='y', labelcolor=color)

    # ax4 = ax1.twinx()
    # color = 'tab:purple'
    # ax4.plot(time[10:drop_idx], roll[10:], label="$\gamma^2 + \\beta^2 + \delta_p^2$", color=color)
    # ax4.tick_params(axis='y', left=False)

    ax1.tick_params(axis='both', which='major', labelsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels([0, 1, 2], fontsize=18)

    plt.title('Control Inputs and Safety Status', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(save_folder, 'multi_point-ncbf-sim-inputs')
    plt.savefig(save_path)










    # ax.plot(time[:drop_idx], rho[:], label="$\gamma^2 + \\beta^2 + \delta_p^2$", linewidth=2)
    # ax.set_xlabel('Time (sec)', fontsize=18)
    # ax.set_ylabel('Summed Squared Angles ($rad^2$)', fontsize=18)

    # ax2 = ax.twinx()
    # ax2.plot(time[:drop_idx], status[0,:], label="Status", linewidth=1, color='r', linestyle='--')
    # ax2.set_ylabel('Safety Status', fontsize=18)

    # plt.axhline(y=(np.pi/4)**2, color='k', linestyle='--')

    # lines, labels = ax.get_legend_handles_labels()
    # lines2, labels2 = ax2.get_legend_handles_labels()
    
    # ax.legend(lines + lines2, labels + labels2, fontsize=18, framealpha=1)
    # plt.xticks(fontsize=18)
    # plt.yticks(fontsize=18)
    # plt.grid(True)
    # plt.tight_layout()
    # save_path = os.path.join(save_folder, 'single_point-ncbf-sim-pendAng')
    # plt.savefig(save_path)