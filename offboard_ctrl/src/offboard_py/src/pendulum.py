import numpy as np

from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose

def pendulum_angles(drone_pose, pend_pose, L=0.5, com=0.5, length=1):
    """
    Function to calculate the roll and pitch of the pendulum from the vertical axis
    :param drone_pose: Pose of the drone
    :param pend_pose: Pose of the pendulum
    :param L: Length from the base of the pendulum to com (m)
    :param com: Length from the top of the pendulum to the center of mass (m)
    :param length: Length of the pendulum (m)
    :return
    """
    # Extracting the orientation of the drone
    drone_quaternion = (drone_pose.orientation.x, 
                        drone_pose.orientation.y, 
                        drone_pose.orientation.z, 
                        drone_pose.orientation.w)
    drone_euler = euler_from_quaternion(drone_pose.orientation)
    drone_roll = drone_euler[0]
    drone_pitch = drone_euler[1]
    drone_yaw = drone_euler[2]

    # Extracting the orientation of the pendulum
    pendulum_quaternion = (pend_pose.orientation.x, 
                           pend_pose.orientation.y, 
                           pend_pose.orientation.z, 
                           pend_pose.orientation.w)
    pendulum_euler = euler_from_quaternion(pendulum_quaternion)
    pendulum_roll = pendulum_euler[0]
    pendulum_pitch = pendulum_euler[1]
    pendulum_yaw = pendulum_euler[2]

    # position of the pendulum com relative to its base along the x-axis in global frame
    r = ...
    # position of the pendulum com relative to its base along the y-axis in global frame
    s = ...
