import numpy as np
import matplotlib.pyplot as plt

# Load data from the .npy files
input_log = np.load("input_log_1.npy")
state_log = np.load("state_log_1.npy")

# Extract variables from state_log
time = np.arange(state_log.shape[1])  # Assuming time steps are along columns
xyz = state_log[0:3, :]
xyz_dot = state_log[3:6, :]
abc = state_log[6:9, :]
rs = state_log[9:11, :]
rs_dot = state_log[11:13, :]

# Extract variables from input_log
angular_velocities = input_log[0:3, :]
acceleration = input_log[3, :]

# Plot and save figures
plt.figure(figsize=(12, 8))

# Plot [x, y, z] vs time
plt.subplot(4, 2, 1)
plt.plot(time, xyz[0, :], label='x')
plt.plot(time, xyz[1, :], label='y')
plt.plot(time, xyz[2, :], label='z')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Position vs Time')
plt.legend()

# Plot [x_dot, y_dot, z_dot] vs time
plt.subplot(4, 2, 2)
plt.plot(time, xyz_dot[0, :], label='x_dot')
plt.plot(time, xyz_dot[1, :], label='y_dot')
plt.plot(time, xyz_dot[2, :], label='z_dot')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Velocity vs Time')
plt.legend()

# Plot [α, β, γ] vs time
plt.subplot(4, 2, 3)
plt.plot(time, abc[0, :], label='alpha')
plt.plot(time, abc[1, :], label='beta')
plt.plot(time, abc[2, :], label='gamma')
plt.xlabel('Time')
plt.ylabel('Orientation')
plt.title('Orientation vs Time')
plt.legend()

# Plot [r, s] vs time
plt.subplot(4, 2, 4)
plt.plot(time, rs[0, :], label='r')
plt.plot(time, rs[1, :], label='s')
plt.xlabel('Time')
plt.ylabel('Coordinates')
plt.title('Coordinates vs Time')
plt.legend()

# Plot [r_dot, s_dot] vs time
plt.subplot(4, 2, 5)
plt.plot(time, rs_dot[0, :], label='r_dot')
plt.plot(time, rs_dot[1, :], label='s_dot')
plt.xlabel('Time')
plt.ylabel('Velocities')
plt.title('Velocities vs Time')
plt.legend()

# Plot [wx, wy, wz] vs time
plt.subplot(4, 2, 6)
plt.plot(time, angular_velocities[0, :], label='wx')
plt.plot(time, angular_velocities[1, :], label='wy')
plt.plot(time, angular_velocities[2, :], label='wz')
plt.xlabel('Time')
plt.ylabel('Angular Velocities')
plt.title('Angular Velocities vs Time')
plt.legend()

# Plot [a] vs time
plt.subplot(4, 2, 7)
plt.plot(time, acceleration, label='a')
plt.xlabel('Time')
plt.ylabel('Acceleration')
plt.title('Acceleration vs Time')
plt.legend()

# Adjust layout
plt.tight_layout()

# Save plots
plt.savefig("plots_1.png")

# Show plots
plt.show()
