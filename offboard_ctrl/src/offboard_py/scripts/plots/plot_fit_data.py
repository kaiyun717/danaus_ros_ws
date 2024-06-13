## For x, y, z##

## PLOT omega_dot.npy vs omega_dot_predicted (C*(omega_des-omega_meas)) ##

import os
import numpy as np
import matplotlib.pyplot as plt

# Import npy files from another directory

# Specify the directory path
directory = 'offboard_ctrl/src/offboard_py/scripts/data_collection/dt_4e4_5datasets'

# Load data from npy files
omega_dot = np.load(os.path.join(directory, 'omega_dot.npy'))
omega_des = np.load(os.path.join(directory, 'omega_des.npy'))
omega_meas = np.load(os.path.join(directory, 'omega_meas.npy'))
C = np.load(os.path.join(directory, 'C.npy'))
R2 = np.load(os.path.join(directory, 'R2_values.npy'))

omega_dot_predicted = C*(omega_des-omega_meas)

# Create regression line to plot

# Plot for x
plt.figure()
plt.scatter(list(range(omega_dot.shape[0])), omega_dot[:,0], label='(X) Measured $\dot{\omega}$', s=10)
plt.scatter(list(range(omega_dot_predicted.shape[0])), omega_dot_predicted[:,0], label='(X) Predicted $\dot{\omega}$', s=10)
plt.plot(omega_dot[:,0], 'k:', color='red', linewidth=0.1)
plt.plot(omega_dot_predicted[:,0], 'k--', color='blue', linewidth=0.1)
plt.title('X $\dot{\omega}$ Linear Model Prediction')
plt.xlabel('Time (20ms intervals)')
plt.ylabel('Angular Acceleration ($rad/s^2$)')
plt.legend()
plt.show()

# Plot for y
plt.figure()
plt.scatter(list(range(omega_dot.shape[0])), omega_dot[:,1], label='(Y) Measured $\dot{\omega}$', s=10)
plt.scatter(list(range(omega_dot_predicted.shape[0])), omega_dot_predicted[:,1], label='(Y) Predicted $\dot{\omega}$', s=10)
plt.plot(omega_dot[:,1], 'k:', color='red', linewidth=0.1)
plt.plot(omega_dot_predicted[:,1], 'k--', color='blue', linewidth=0.1)
plt.title('Y $\dot{\omega}$ Linear Model Prediction')
plt.xlabel('Time (20ms intervals)')
plt.ylabel('Angular Acceleration ($rad/s^2$)')
plt.legend()
plt.show()

# # Plot for z
plt.figure()
plt.scatter(list(range(omega_dot.shape[0])), omega_dot[:,2], label='(Z) Measured $\dot{\omega}$', s=10)
plt.scatter(list(range(omega_dot_predicted.shape[0])), omega_dot_predicted[:,2], label='(Z) Predicted $\dot{\omega}$', s=10)
plt.plot(omega_dot[:,2], 'k:', color='red', linewidth=0.1)
plt.plot(omega_dot_predicted[:,2], color='blue', linewidth=0.1)
plt.title('Z $\dot{\omega}$ Linear Model Prediction')
plt.xlabel('Time (20ms intervals)')
plt.ylabel('Angular Acceleration ($rad/s^2$)')
plt.legend()
plt.show()

x = omega_des[:,0] - omega_meas[:,0]
y = omega_des[:,1] - omega_meas[:,1]
z = omega_des[:,2] - omega_meas[:,2]

# Plot for x
plt.figure()
plt.scatter(x, omega_dot[:,0], label='X Sampled', color="red", s=1)
plt.plot(x, omega_dot_predicted[:,0], label='X Predicted', color="blue")
plt.title('X Body Rate Angular Acceleration Prediction \n $R^2$: ' + str(R2), fontsize=25)
plt.xlabel('$\omega_{des} - \omega_{meas}$ ($rad/s$)', fontsize=25)
plt.ylabel('$\dot{\omega}$ ($rad/s^2$)', fontsize=25)
plt.legend(fontsize=25)
plt.show()

# Plot for y
plt.figure()
plt.scatter(y, omega_dot[:,1], label='Y Sampled', color="red", s=1)
plt.plot(y, omega_dot_predicted[:,1], label='Y Predicted', color="blue")
plt.title('Y Body Rate Angular Acceleration Prediction \n $R^2$: ' + str(R2), fontsize=25)
plt.xlabel('$\omega_{des} - \omega_{meas}$ ($rad/s$)', fontsize=25)
plt.ylabel('$\dot{\omega}$ ($rad/s^2$)', fontsize=25)
plt.legend(fontsize=25)
plt.show()

# Plot for z
plt.figure()
plt.scatter(z, omega_dot[:,2], label='Z Sampled', color="red", s=1)
plt.plot(z, omega_dot_predicted[:,2], label='Z Predicted', color="blue")
plt.title('Z Body Rate Angular Acceleration Prediction \n $R^2$: ' + str(R2), fontsize=25)
plt.xlabel('$\omega_{des} - \omega_{meas}$ ($rad/s$)', fontsize=25)
plt.ylabel('$\dot{\omega}$ ($rad/s^2$)', fontsize=25)
plt.legend(fontsize=25)
plt.show()