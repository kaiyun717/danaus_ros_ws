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
    
def load_data(folder_path):
    log = np.load(os.path.join(folder_path, 'log.npy'), allow_pickle=True).item()
    state_log = log['state']
    nom_input_log = log['nom_input']
    safe_input_log = log['safe_input']
    status_log = log['status_log']
    phi_val_log = log['phi_val_log']
    next_phi_val_log = log['next_phi_val_log']
    slack_log = log['slack_log']
    params = np.load(os.path.join(folder_path, 'params.npy'), allow_pickle=True).item()
    
    return state_log, nom_input_log, safe_input_log, status_log, phi_val_log, next_phi_val_log, slack_log, params

def plot_angles(state_log, status_log, params, save_path, drop_idx):

    xyz_ang = state_log[0:3, :drop_idx]
    pend_ang = state_log[6:8, :drop_idx]

    status = status_log[:, :drop_idx]
    status = np.where(status == 2, 0, np.where(status == 1, 1, 2))

    time = np.arange(xyz_ang.shape[1])
    time = time / params["hz"]

    roll = xyz_ang[0, :]
    pitch = xyz_ang[1, :]
    psi = pend_ang[0, :]
    theta = pend_ang[1, :]
    delta = np.arccos(np.cos(theta) * np.cos(psi))

    rho = np.sqrt(roll**2 + pitch**2 + delta**2)

    fig, ax1 = plt.subplots(figsize=(10, 5))
    line1, = ax1.plot(time[:], xyz_ang[0, :], label='$\gamma$')
    line2, = ax1.plot(time[:], xyz_ang[1, :], label='$\\beta$')
    line3, = ax1.plot(time[:], pend_ang[0, :], label='$\\theta$')
    line4, = ax1.plot(time[:], pend_ang[1, :], label='$\psi$')
    line5, = ax1.plot(time[:], rho[:], label="$\sqrt{\gamma^2 + \\beta^2 + \delta^2}$", linewidth=2)
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Angles (rad)")
    ax1.axhline(y=(np.pi / 4), color='k', linestyle='--')

    y_min, y_max = ax1.get_ylim()
    normalized_status = (status[0, :] / 2) * (y_max - y_min) + y_min

    ax2 = ax1.twinx()
    line6, = ax2.plot(time[:], normalized_status[:], label='Status', linewidth=2, color="g", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Safety Status")
    ax2.set_yticks([y_min, (y_max + y_min) / 2, y_max])
    ax2.set_yticklabels(['IN', 'ON', 'OUT'])

    on_indices = np.where(status[0, :] == 1)
    if len(on_indices[0]) > 0:  # Check if there are any ON indices
        ax2.scatter(time[on_indices], normalized_status[on_indices], color='red', s=50, label='Safety ON')

    # Combine legends from both axes
    lines = [line1, line2, line3, line4, line5, line6]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.title("Angles & Safety Status")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

def plot_states(state_log, params, save_path, drop_idx):
    xyz_ang = state_log[0:3, :drop_idx]
    xyz_ang_dot = state_log[3:6, :drop_idx]
    pend_ang = state_log[6:8, :drop_idx]
    pend_ang_dot = state_log[8:10, :drop_idx]
    xyz_pos = state_log[10:13, :drop_idx]
    xyz_pos_dot = state_log[13:16, :drop_idx]

    time = np.arange(xyz_ang.shape[1])
    time = time / params["hz"]

    plt.figure(figsize=(16, 12))

    plt.subplot(3, 2, 1)
    plt.plot(time[:], xyz_pos[0, :], label='$x$')
    plt.plot(time[:], xyz_pos[1, :], label='$y$')
    plt.plot(time[:], xyz_pos[2, :], label='$z$')
    plt.xlabel('Time (sec)')
    plt.ylabel('Position (m)')
    plt.title('Quad Pos. vs Time')
    plt.legend()

    plt.subplot(3, 2, 2)
    plt.plot(time[:], xyz_pos_dot[0, :], label='$\dot x$')
    plt.plot(time[:], xyz_pos_dot[1, :], label='$\dot y$')
    plt.plot(time[:], xyz_pos_dot[2, :], label='$\dot z$')
    plt.xlabel('Time (sec)')
    plt.ylabel('Velocity (m/s)')
    plt.title('Quad Vel. vs Time')
    plt.legend()

    plt.subplot(3, 2, 3)
    plt.plot(time[:], xyz_ang[0, :], label='$\gamma$')
    plt.plot(time[:], xyz_ang[1, :], label='$\\beta$')
    plt.plot(time[:], xyz_ang[2, :], label='$\\alpha$')
    plt.xlabel('Time (sec)')
    plt.ylabel('Angle (rad)')
    plt.title('Quad Angle vs Time')
    plt.legend()

    plt.subplot(3, 2, 4)
    plt.plot(time[:], xyz_ang_dot[0, :], label='$\dot \gamma$')
    plt.plot(time[:], xyz_ang_dot[1, :], label='$\dot \\beta$')
    plt.plot(time[:], xyz_ang_dot[2, :], label='$\dot \\alpha$')
    plt.xlabel('Time (sec)')
    plt.ylabel('Ang. Vel. (rad/s)')
    plt.title('Quad Ang. Vel. vs Time')
    plt.legend()

    plt.subplot(3, 2, 5)
    plt.plot(time[:], pend_ang[0, :], label='$\\theta$')
    plt.plot(time[:], pend_ang[1, :], label='$\psi$')
    plt.xlabel('Time (sec)')
    plt.ylabel('Angle (rad)')
    plt.title('Pendulum Angle vs Time')
    plt.legend()

    plt.subplot(3, 2, 6)
    plt.plot(time[:], pend_ang_dot[0, :], label='$\dot \\theta$')
    plt.plot(time[:], pend_ang_dot[1, :], label='$\dot \psi$')
    plt.xlabel('Time (sec)')
    plt.ylabel('Ang. Vel. (rad/s)')
    plt.title('Pendulum Ang. Vel. vs Time')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)

def plot_inputs(nom_input_log, safe_input_log, status_log, params, save_path, drop_idx):
     
    nom_inputs = nom_input_log[:, :drop_idx]
    safe_inputs = safe_input_log[:, :drop_idx]
    status = status_log[:, :drop_idx]
    status = np.where(status == 2, 0, np.where(status == 1, 1, 2))

    time = np.arange(nom_inputs.shape[1])
    time = time / params["hz"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    lines = []
    idx_labels = ["$a$", "$\omega_x$", "$\omega_y$", "$\omega_z$"]
    y_labels = ["Acceleration (m/s^2)", "Body Rates (rad/s)", "Body Rates (rad/s)", "Body Rates (rad/s)"]

    axes = [ax1, ax2, ax3, ax4]
    ### BODY RATES ###
    for i in range(4):
        line1, = axes[i].plot(time[:], nom_inputs[i, :], label="NOM: "+idx_labels[i])
        line2, = axes[i].plot(time[:], safe_inputs[i, :], label="SAFE: "+idx_labels[i])
        axes[i].set_xlabel("Time (sec)")
        axes[i].set_ylabel(y_labels[i])
        axes[i].set_title(idx_labels[i])
        y_min, y_max = axes[i].get_ylim()
        normalized_status = (status[0, :] / 2) * (y_max - y_min) + y_min

        ax_1 = axes[i].twinx()
        line3, = ax_1.plot(time[:], normalized_status, label='Status', linewidth=2, color="g", linestyle="--", alpha=0.3)
        ax_1.set_ylabel("Safety Status")
        ax_1.set_yticks([y_min, (y_max + y_min) / 2, y_max])
        ax_1.set_yticklabels(['IN', 'ON', 'OUT'])

        on_indices = np.where(status[0, :] == 1)
        if len(on_indices[0]) > 0:  # Check if there are any ON indices
            ax_1.scatter(time[on_indices], normalized_status[on_indices], color='red', s=50, label='Safety ON')

        lines.extend([line1, line2, line3])

    labels = [line.get_label() for line in lines]
    axes[0].legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.suptitle("Inputs & Safety Status")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to accommodate the legend
    plt.savefig(save_path, bbox_inches='tight')
    
def plot_phi_and_slack_vals(phi_val_log, next_phi_val_log, status_log, slack_log, params, save_path, drop_idx):

    phi_vals = phi_val_log[:, :drop_idx]
    next_phi_vals = next_phi_val_log[:, :drop_idx]
    slacks = slack_log[:, :drop_idx]
    status = status_log[:, :drop_idx]
    status = np.where(status == 2, 0, np.where(status == 1, 1, 2))

    time = np.arange(phi_vals.shape[1])
    time = time / params["hz"]

    #################
    ### Plot Phis ###
    #################
    fig, ax1 = plt.subplots(figsize=(10, 5))
    line1, = ax1.plot(time[:], phi_vals[0, :], label='$\phi$')
    line2, = ax1.plot(time[:], next_phi_vals[0, :], label='$\phi_{next}$')
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Phi Values")
    ax1.axhline(y=1e-2, color='k', linestyle='--')
    ax1.axhline(y=0, color='k', linestyle='--')

    y_min, y_max = ax1.get_ylim()
    normalized_status = (status[0, :] / 2) * (y_max - y_min) + y_min

    ax2 = ax1.twinx()
    line3, = ax2.plot(time[:], normalized_status[:], label='Status', linewidth=2, color="g", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Safety Status")
    ax2.set_yticks([y_min, (y_max + y_min) / 2, y_max])
    ax2.set_yticklabels(['IN', 'ON', 'OUT'])

    on_indices = np.where(status[0, :] == 1)
    if len(on_indices[0]) > 0:  # Check if there are any ON indices
        ax2.scatter(time[on_indices], normalized_status[on_indices], color='red', s=50, label='Safety ON')

    lines = [line1, line2, line3]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.title("Phi Values & Safety Status")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')

    ###################
    ### Plot Slacks ###
    ###################
    fig, ax1 = plt.subplots(figsize=(10, 5))
    line1, = ax1.plot(time[:], slacks[0, :], label='Slack')
    ax1.set_xlabel("Time (sec)")
    ax1.set_ylabel("Slack Values")

    y_min, y_max = ax1.get_ylim()
    normalized_status = (status[0, :] / 2) * (y_max - y_min) + y_min

    ax2 = ax1.twinx()
    line2, = ax2.plot(time[:], normalized_status[:], label='Status', linewidth=2, color="g", linestyle="--", alpha=0.3)
    ax2.set_ylabel("Safety Status")
    ax2.set_yticks([y_min, (y_max + y_min) / 2, y_max])
    ax2.set_yticklabels(['IN', 'ON', 'OUT'])

    on_indices = np.where(status[0, :] == 1)
    if len(on_indices[0]) > 0:  # Check if there are any ON indices
        ax2.scatter(time[on_indices], normalized_status[on_indices], color='red', s=50, label='Safety ON')

    lines = [line1, line2]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.title("Slack Values & Safety Status")
    plt.tight_layout()
    plt.savefig(save_path.replace("phi_vals", "slacks"), bbox_inches='tight')




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data files.')
    parser.add_argument('--logs_dir', type=str, default='/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/ncbf',
                        help='Path to the directory containing log folders')
    parser.add_argument('--folder_name', type=str, default=None,
                        help='Name of the folder in which data is stored. If not provided, the script will use the latest folder.')
    parser.add_argument('--drop_time', type=float, default=None,
                        help='Time to drop from the start of the log in seconds')
    args = parser.parse_args()

    save_folder = '/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/ncbf'
    logs_dir = args.logs_dir
    folder_name = args.folder_name
    drop_time = args.drop_time

    state_log, nom_input_log, safe_input_log, status_log, phi_val_log, next_phi_val_log, slack_log, params = load_data(os.path.join(logs_dir, folder_name))

    if drop_time is None:
        drop_time = "all"
        drop_idx = int(params["cont_duration"] * params["hz"])
    else:
        drop_idx = int(drop_time * params["hz"])
    
    save_path = os.path.join(save_folder, folder_name, f"angles_{drop_time}.png")
    plot_angles(state_log, status_log, params, save_path, drop_idx)

    save_path = os.path.join(save_folder, folder_name, f"states_{drop_time}.png")
    plot_states(state_log, params, save_path, drop_idx)

    save_path = os.path.join(save_folder, folder_name, f"inputs_{drop_time}.png")
    plot_inputs(nom_input_log, safe_input_log, status_log, params, save_path, drop_idx)

    save_path = os.path.join(save_folder, folder_name, f"phi_vals_{drop_time}.png")
    plot_phi_and_slack_vals(phi_val_log, next_phi_val_log, status_log, slack_log, params, save_path, drop_idx)