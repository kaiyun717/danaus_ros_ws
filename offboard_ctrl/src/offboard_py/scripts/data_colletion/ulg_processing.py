import numpy as np
from numpy import genfromtxt
import scipy
import IPython

def delete_setpoints_before_takeoff(rate_set_np):
    indices = np.where(np.abs(rate_set_np[:,6]) > 0.3)[0]
    if indices.size == 0:
        raise ValueError("No takeoff detected in the rate setpoint data")
    # return rate_set_np[indices[0]:, :]    # THIS IS NOT CORRECT SINCE YOU MAY HAVE ZERO SETPOINTS WHEN LANDING
    return rate_set_np[indices, :]

def get_common_timestamps(ang_vel_np, rate_set_np):
    ang_vel_timestamps = ang_vel_np[:, 0]
    rate_set_timestamps = rate_set_np[:, 0]

    common_timestamps = np.intersect1d(ang_vel_timestamps, rate_set_timestamps)

    return common_timestamps

def get_timestamps_with_same_dt(common_timestamps, dt=20_000):
    common_diff = np.diff(common_timestamps)
    diff_dt = np.where(common_diff == dt)[0]

    print("Number of common timestamps with dt = {} is: {}".format(dt, diff_dt.shape[0]))

    curr = common_timestamps[diff_dt]
    next = common_timestamps[diff_dt + 1]

    assert np.mean(next - curr) == dt, "The difference between the timestamps is not equal to the specified dt"

    return curr, next

def get_data_for_same_dt(curr, next, ang_vel_np, rate_set_np):

    # IPython.embed()

    ### There are repeated timestamps in the data. We need to remove them.
    unique_ang_vel, unique_ang_vel_indices = np.unique(ang_vel_np[:, 0], return_index=True)
    ang_vel_np = ang_vel_np[unique_ang_vel_indices, :]
    unique_rate_set, unique_rate_set_indices = np.unique(rate_set_np[:, 0], return_index=True)
    rate_set_np = rate_set_np[unique_rate_set_indices, :]
    
    ### Get the indices of the current and next timestamps in the data
    omega_des_indices = np.where(np.isin(rate_set_np[:, 0], curr))
    omega_meas_indices = np.where(np.isin(ang_vel_np[:, 0], curr))
    omega_dot_indices = np.where(np.isin(ang_vel_np[:, 0], next))

    omega_des = rate_set_np[omega_des_indices, 1:4].squeeze()
    omega_meas = ang_vel_np[omega_meas_indices, 2:5].squeeze()
    omega_dot = ang_vel_np[omega_dot_indices, 5:8].squeeze()

    return omega_dot, omega_meas, omega_des


def get_all_data(ang_vel_ulg_list, rate_set_ulg_list, dt=20_000):

    omega_dot_list = []
    omega_meas_list = []
    omega_des_list = []

    for i in range(len(ang_vel_ulg_list)):
        ang_vel_np = ang_vel_ulg_list[i]
        rate_set_np = rate_set_ulg_list[i]
        
        common_timestamps = get_common_timestamps(ang_vel_np, rate_set_np)
        curr_timestamps, next_timestamps = get_timestamps_with_same_dt(common_timestamps, dt)
        omega_dot, omega_meas, omega_des = get_data_for_same_dt(curr_timestamps, next_timestamps, ang_vel_np, rate_set_np)

        omega_dot_list.append(omega_dot)
        omega_meas_list.append(omega_meas)
        omega_des_list.append(omega_des)

    omega_dot = np.vstack(omega_dot_list)
    omega_meas = np.vstack(omega_meas_list)
    omega_des = np.vstack(omega_des_list)

    return omega_dot, omega_meas, omega_des

def delete_small_setpoints(omega_dot, omega_meas, omega_des, mag_tol=0.5):
    """
    This is not the best?
    """
    mask = np.all(np.abs(omega_des) >= mag_tol, axis=1)
    indices = np.where(mask)[0]

    omega_dot = omega_dot[indices, :]
    omega_meas = omega_meas[indices, :]
    omega_des = omega_des[indices, :]
    return omega_dot, omega_meas, omega_des

def delete_big_setpoints(omega_dot, omega_meas, omega_des, mag_tol=15):
    """
    This is not the best?
    """
    mask = np.all(np.abs(omega_des) <= mag_tol, axis=1)
    indices = np.where(mask)[0]

    omega_dot = omega_dot[indices, :]
    omega_meas = omega_meas[indices, :]
    omega_des = omega_des[indices, :]
    return omega_dot, omega_meas, omega_des


def lstsq_fit(omega_dot, omega_meas, omega_des):
    """
    All the input arrays are of shape (X,3).
    Fit a line to the data.
    """
    N = omega_meas.shape[0]
    X = omega_des - omega_meas    # (N,3)
    Y = omega_dot                      # (N,3)

    C, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    R2_values = np.zeros([1,3])

    # Find the individual R2 values for Cx Cy Cz
    for i in range(3):
        SSR = residuals[i]
        SST = np.sum((Y[:, i] - np.mean(Y[:, i]))**2)
        R2 = 1 - SSR/SST
        print("R2 for C{}: {}".format(i, R2)) 
        R2_values[0,i] = R2
    return C, R2_values

def lstsq_fit_all(omega_dot, omega_meas, omega_des):
    omega_dot_all = np.reshape(omega_dot, (-1, 1))
    omega_meas_all = np.reshape(omega_meas, (-1, 1))
    omega_des_all = np.reshape(omega_des, (-1, 1))

    X = omega_des_all - omega_meas_all
    Y = omega_dot_all

    C, residuals, _, _ = np.linalg.lstsq(X, Y, rcond=None)

    SSR = residuals[0]
    SST = np.sum((Y - np.mean(Y))**2)
    R2 = 1 - SSR/SST

    print("R2: ", R2)
    return C, R2


if __name__ == "__main__":

    ########################################################################################################
    ########### Load the data ##############################################################################
    ########################################################################################################
    ### = ang_vel_ulg =                                                                                  ###
    ### (timestamp, timestamp_sample, xyz[0], xyz[1], xyz[2], xyz_deriv[0], xyz_deriv[1], xyz_deriv[2])  ###
    ### = rate_set_ulg =                                                                                 ###
    ### (timestamp, roll, pitch, yaw, thrust_body[0], thrust_body[1], thrust_body[2], reset_integral)    ###
    ########################################################################################################

    ######## OKAY DATA ########
    log_dir_ulg_0610_133851 = "/home/kai/nCBF-drone/flight_logs/0610_133851-17_37_46"
    log_dir_ulg_0610_173119 = "/home/kai/nCBF-drone/flight_logs/0610_173119-21_30_44"
    log_dir_ulg_0610_204621 = "/home/kai/nCBF-drone/flight_logs/0610_204621-00_45_44"

    ang_vel_ulg_0610_133851 = genfromtxt(log_dir_ulg_0610_133851 + '/17_37_46_vehicle_angular_velocity_0.csv', delimiter=',')
    rate_set_ulg_0610_133851 = genfromtxt(log_dir_ulg_0610_133851 + '/17_37_46_vehicle_rates_setpoint_0.csv', delimiter=',')
    rate_set_ulg_0610_133851 = delete_setpoints_before_takeoff(rate_set_ulg_0610_133851)

    ang_vel_ulg_0610_173119 = genfromtxt(log_dir_ulg_0610_173119 + '/21_30_44_vehicle_angular_velocity_0.csv', delimiter=',')
    rate_set_ulg_0610_173119 = genfromtxt(log_dir_ulg_0610_173119 + '/21_30_44_vehicle_rates_setpoint_0.csv', delimiter=',')
    rate_set_ulg_0610_173119 = delete_setpoints_before_takeoff(rate_set_ulg_0610_173119)

    ang_vel_ulg_0610_204621 = genfromtxt(log_dir_ulg_0610_204621 + '/00_45_44_vehicle_angular_velocity_0.csv', delimiter=',')
    rate_set_ulg_0610_204621 = genfromtxt(log_dir_ulg_0610_204621 + '/00_45_44_vehicle_rates_setpoint_0.csv', delimiter=',')
    rate_set_ulg_0610_204621 = delete_setpoints_before_takeoff(rate_set_ulg_0610_204621)

    # ang_vel_ulg_list = [ang_vel_ulg_0610_133851, ang_vel_ulg_0610_173119, ang_vel_ulg_0610_204621]
    # rate_set_ulg_list = [rate_set_ulg_0610_133851, rate_set_ulg_0610_173119, rate_set_ulg_0610_204621]

    ##############
    ## RESULTS ###
    ##############
    ## If small_setpoint to 0.1:
    ## >>> In [1]: R2_values
    ## >>> Out[1]: array([[0.3828725 , 0.46113398, 0.24129362]])

    ## Without any cutoff:
    ## >>> In [2]: R2_values
    ## >>> Out[2]: array([[0.3895024 , 0.46733396, 0.24012244]])

    ######## OKAY DATA ########

    # ######## BAD DATA ########
    # ang_vel_ulg_0611_183152 = genfromtxt("/home/kai/nCBF-drone/flight_logs/0611_183152-22_31_02/22_31_02_vehicle_angular_velocity_0.csv", delimiter=',')
    # rate_set_ulg_0611_183152 = genfromtxt("/home/kai/nCBF-drone/flight_logs/0611_183152-22_31_02/22_31_02_vehicle_rates_setpoint_0.csv", delimiter=',')
    # rate_set_ulg_0611_183152 = delete_setpoints_before_takeoff(rate_set_ulg_0611_183152)

    # ang_vel_ulg_list = [ang_vel_ulg_0611_183152]
    # rate_set_ulg_list = [rate_set_ulg_0611_183152]
    # ######## BAD DATA ########

    ######## 50 Hz ctrl ########
    ang_vel_ulg_0611_190110 = genfromtxt("/home/kai/nCBF-drone/flight_logs/0611_190110-23_00_27/23_00_27_vehicle_angular_velocity_0.csv", delimiter=',')
    rate_set_ulg_0611_190110 = genfromtxt("/home/kai/nCBF-drone/flight_logs/0611_190110-23_00_27/23_00_27_vehicle_rates_setpoint_0.csv", delimiter=',')
    rate_set_ulg_0611_190110 = delete_setpoints_before_takeoff(rate_set_ulg_0611_190110)

    ang_vel_ulg_0611_194231 = genfromtxt("/home/kai/nCBF-drone/flight_logs/0611_194231-23_42_03/23_42_03_vehicle_angular_velocity_0.csv", delimiter=',')
    rate_set_ulg_0611_194231 = genfromtxt("/home/kai/nCBF-drone/flight_logs/0611_194231-23_42_03/23_42_03_vehicle_rates_setpoint_0.csv", delimiter=',')
    rate_set_ulg_0611_194231 = delete_setpoints_before_takeoff(rate_set_ulg_0611_194231)
    ######## 50 Hz ctrl ########

    ang_vel_ulg_list = [ang_vel_ulg_0611_190110, ang_vel_ulg_0611_194231, ang_vel_ulg_0610_133851, ang_vel_ulg_0610_173119, ang_vel_ulg_0610_204621]
    rate_set_ulg_list = [rate_set_ulg_0611_190110, rate_set_ulg_0611_194231, rate_set_ulg_0610_133851, rate_set_ulg_0610_173119, rate_set_ulg_0610_204621]

    omega_dot, omega_meas, omega_des = get_all_data(ang_vel_ulg_list, rate_set_ulg_list, dt=40_000)
    # omega_dot, omega_meas, omega_des = delete_small_setpoints(omega_dot, omega_meas, omega_des, mag_tol=0.1)
    # omega_dot, omega_meas, omega_des = delete_big_setpoints(omega_dot, omega_meas, omega_des, mag_tol=15)
    
    # C, R2_values = lstsq_fit(omega_dot, omega_meas, omega_des)
    C, R2_values = lstsq_fit_all(omega_dot, omega_meas, omega_des)
    IPython.embed()
    print("C: ", C)
    print("R2 values: ", R2_values)
