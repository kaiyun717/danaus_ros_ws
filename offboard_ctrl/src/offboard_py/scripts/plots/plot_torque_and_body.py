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
    state_log = log['state_log']
    eth_input_log = log['eth_input_log']
    bodyrate_input_log = log['bodyrate_input_log']
    torque_input_log = log['torque_input_log']
    error_log = log['error_log']
    params = np.load(os.path.join(folder_path, 'params.npy'), allow_pickle=True).item()
    
    return state_log, eth_input_log, bodyrate_input_log, torque_input_log, error_log, params

def plot_states(state_log, params, save_path, drop_idx):
    xyz_ang = state_log[0:3, :drop_idx]
    xyz_ang_dot = state_log[3:6, :drop_idx]
    xyz_pos = state_log[6:9, :drop_idx]
    xyz_pos_dot = state_log[9:12, :drop_idx]
    pend_ang = state_log[12:14, :drop_idx]
    pend_ang_dot = state_log[14:16, :drop_idx]
    
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

def plot_inputs(eth_input_log, bodyrate_input_log, torque_input_log, params, save_path, drop_idx):
     
    eth_inputs = eth_input_log[:, :drop_idx]
    bodyrate_inputs = bodyrate_input_log[:, :drop_idx]
    torque_inputs = torque_input_log[:, :drop_idx]

    time = np.arange(eth_inputs.shape[1])
    time = time / params["hz"]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    lines = []
    idx_labels = ["$a$", "$\omega_x$", "$\omega_y$", "$\omega_z$"]
    y_labels = ["Acceleration ($m/s^2$)", "Body Rates (rad/s)", "Body Rates (rad/s)", "Body Rates (rad/s)"]
    torque_labels = ["Acceleration ($m/s^2$)", "$\\tau_x$", "$\\tau_y$", "$\\tau_z$"]

    axes = [ax1, ax2, ax3, ax4]
    ### BODY RATES ###
    for i in range(4):
        line1, = axes[i].plot(time[:], eth_inputs[i, :], label="ETH: "+idx_labels[i])
        line2, = axes[i].plot(time[:], bodyrate_inputs[i, :], label="OUR B.R.: "+idx_labels[i])
        axes[i].set_xlabel("Time (sec)")
        axes[i].set_ylabel(y_labels[i])
        axes[i].set_title(idx_labels[i])

        ax_1 = axes[i].twinx()
        line3, = ax_1.plot(time[:], torque_inputs[i, :], label='Our T.: '+torque_labels[i], color="g", linestyle="--", alpha=0.5)
        ax_1.set_ylabel("Torque (N*m)")
        # ax_1.set_yticks([y_min, (y_max + y_min) / 2, y_max])
        # ax_1.set_yticklabels(['IN', 'ON', 'OUT'])

        lines.extend([line1, line2, line3])

    labels = [line.get_label() for line in lines]
    axes[0].legend(lines, labels, loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    plt.suptitle("Inputs")
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust the layout to accommodate the legend
    plt.savefig(save_path, bbox_inches='tight')
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process data files.')
    parser.add_argument('--logs_dir', type=str, default='/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/torque',
                        help='Path to the directory containing log folders')
    parser.add_argument('--folder_name', type=str, default=None,
                        help='Name of the folder in which data is stored. If not provided, the script will use the latest folder.')
    parser.add_argument('--drop_time', type=float, default=None,
                        help='Time to drop from the start of the log in seconds')
    args = parser.parse_args()

    save_folder = '/home/kai/nCBF-drone/danaus_ros_ws/offboard_ctrl/src/offboard_py/logs/torque'
    logs_dir = args.logs_dir
    folder_name = args.folder_name
    drop_time = args.drop_time

    state_log, eth_input_log, bodyrate_input_log, torque_input_log, error_log, params = load_data(os.path.join(logs_dir, folder_name))

    if drop_time is None:
        drop_time = "all"
        drop_idx = int(params["cont_duration"] * params["hz"])
    else:
        drop_idx = int(drop_time * params["hz"])
    
    save_path = os.path.join(save_folder, folder_name, f"states_{drop_time}.png")
    plot_states(state_log, params, save_path, drop_idx)

    save_path = os.path.join(save_folder, folder_name, f"inputs_{drop_time}.png")
    plot_inputs(eth_input_log, bodyrate_input_log, torque_input_log, params, save_path, drop_idx)
