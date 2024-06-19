import IPython
import rospy
import math
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from gazebo_msgs.srv import GetLinkState
from mavros_msgs.msg import State
from mavros_msgs.srv import *
import collections


class PendulumCB:
    def __init__(self, mode, vehicle, L_p=0.69) -> None:
        """
        Callback class for the pendulum state.

        mode: 'sim' or 'real'
        vehicle: 'danaus12_old' or 'danaus12_newold'
        L_p: CoM of the pendulum wrt the pivot point (base)
        """
        ##### Parameters #####
        self.mode = mode
        self.L_p = L_p
        self.vehicle = vehicle

        ##### History Info #####
        # The previous pose/ang of the pendulum are used to calculate the velocities.
        self.pose = PoseStamped()
        self.prev_pose = PoseStamped()
        self.prev_pend_ang = np.array([0, 0])
        
        self.avg_vel = np.array([0, 0])
        self.avg_pend_vel = np.array([0, 0])
        # Moving average weight
        self.w_avg = 0.5

        if mode == 'sim':
            self.pose_sub = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        elif mode == 'real':
            self.pose_sub = rospy.Subscriber('vicon/pose/pend/pend', PoseStamped, self.pose_cb)
        else:
            raise ValueError('Invalid mode')

        self.dq_len = 10
        self.pose_deque = collections.deque(maxlen=self.dq_len)

    def pose_cb(self, msg):
        # self.pose_deque.append(msg)
        self.prev_pose = self.pose
        self.pose = msg

    def get_rs_pose(self, vehicle_pose=None):
        """
        Get the position of the pendulum wrt the pivot point (base).
        The pivot point is given as the `vehicle_pose`.
        """
        if self.mode == 'sim':
            return self._get_rs_pose_sim(vehicle_pose)
        elif self.mode == 'real':
            return self._get_rs_pose_real(vehicle_pose)
        else:
            raise ValueError('Invalid mode')
    
    def _get_rs_pose_real(self, vehicle_pose):
        """
        RS pose for real-world experiments.
        """
        r = self.pose.pose.position.x - vehicle_pose[0]
        s = self.pose.pose.position.y - vehicle_pose[1]
        if rospy.Time.now() - self.pose.header.stamp > rospy.Duration(0.2):
            rospy.logwarn_throttle(1, "Pendulum pose is too old!")
            return None
        return np.array([r, s])
    
    def _get_rs_pose_sim(self, vehicle_pose):
        """
        RS pose for simulation.
        """
        response = self.pose_sub(link_name=self.vehicle+'::pendulum', reference_frame='')
        r = response.link_state.pose.position.x - vehicle_pose[0] #(vehicle_pose[0] -0.001995)
        s = response.link_state.pose.position.y - vehicle_pose[1] #(vehicle_pose[1] + 0.000135)
        return np.array([r, s])
    
    def get_rs_ang(self, vehicle_pose=None):
        """
        RS angles for the pendulum wrt the pivot point (base).
        `roll` is about the y-axis and `pitch` is about the x-axis.
        `pitch` corresponds to `r`. If `r` is positive, then `pitch` is positive.
        `roll` corresponds to `s`. If `s` is positive, then `roll` is negative.
        """
        rs_pose = self.get_rs_pose(vehicle_pose)
        xi = np.sqrt(self.L_p**2 - rs_pose[0]**2 - rs_pose[1]**2)
        pitch = np.arctan2(rs_pose[0], xi)      # Pitch is about the y-axis. Thus, use x and z.
        roll = np.arctan2(-rs_pose[1], xi)       # Roll is about the x-axis. Thus, use y and z.
        return np.array([roll, pitch])

    def get_rs_ang_vel(self, vehicle_pose=None, vehicle_vel=None):
        """
        RS angular velocities for the pendulum wrt the pivot point (base).
        (Same logic as angles also applies for the angular velocities.) 
        """
        rs_vel = self.get_rs_vel(vehicle_vel)
        pitch_dot = rs_vel[0]/self.L_p
        roll_dot = -rs_vel[1]/self.L_p
        return np.array([roll_dot, pitch_dot])
    
    def get_rs_vel(self, vehicle_vel=None, dt=None):
        """
        RS velocities for the pendulum wrt the pivot point (base).
        """
        if self.mode == 'sim':
            return self._get_rs_vel_sim(vehicle_vel)
        elif self.mode == 'real':
            return self._get_rs_vel_real(vehicle_vel)
        else:   
            raise ValueError('Invalid mode')

    def _get_rs_vel_sim(self, vehicle_vel):
        """
        RS velocities for the pendulum in simulation.
        """
        # response = self.pose_sub(link_name='danaus12_old::pendulum', reference_frame='base_link')    # Position changes with base_link frame
        response = self.pose_sub(link_name=self.vehicle+'::pendulum', reference_frame='')
        r = response.link_state.twist.linear.x - vehicle_vel[0]
        s = response.link_state.twist.linear.y - vehicle_vel[1]
        return np.array([r, s])

    def _get_rs_vel_real(self, vehicle_vel):
        """
        RS velocities for the pendulum in real-world experiments.
        We use an moving average to smoothen out the velocity calculations.
        """
        # if len(self.pose_deque) == self.dq_len:
        #     self.prev_pose = self.pose_deque.popleft()
        # else:
        #     return np.array([0, 0])
        current_time = self.pose.header.stamp.to_sec()
        prev_time = self.prev_pose.header.stamp.to_sec()
        dt = current_time - prev_time
        if dt < 1e-5:
            print("Time difference is too small!")
            return np.array([0, 0])
        
        r = (self.pose.pose.position.x - self.prev_pose.pose.position.x) / dt 
        s = (self.pose.pose.position.y - self.prev_pose.pose.position.y) / dt 
        self.avg_vel = (1 - self.w_avg)*np.array([r, s]) + self.w_avg*self.avg_vel
        
        # rs_vel_msg = TwistStamped()
        # rs_vel_msg.twist.linear.x = r
        # rs_vel_msg.twist.linear.y = s
        # rs_vel_msg.header.stamp = rospy.Time.now()
        # self.rs_vel_pub.publish(rs_vel_msg)

        # rs_vel_avg_msg = TwistStamped()
        # rs_vel_avg_msg.twist.linear.x = self.avg_vel[0]
        # rs_vel_avg_msg.twist.linear.y = self.avg_vel[1]
        # rs_vel_avg_msg.header.stamp = rospy.Time.now()
        # self.rs_vel_avg_pub.publish(rs_vel_avg_msg)

        # rospy.loginfo(f"Velocity: r: {r}, s: {s}")
        # rospy.loginfo(f"Avg.Vel.: r: {self.avg_vel[0]}, s: {self.avg_vel[1]}")
        return self.avg_vel - vehicle_vel[:2]
    

if __name__ == "__main__":
    from src.callbacks.fcu_state_callbacks import VehicleStateCB

    rospy.init_node('pendulum_state_cb', anonymous=True)
    
    pend_cb = PendulumCB("sim")
    quad_cb = VehicleStateCB("sim")
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        quad_pose = quad_cb.get_xyz_pose()
        quad_vel = quad_cb.get_xyz_velocity()
        pend_pose = pend_cb.get_rs_pose(quad_pose)
        pend_vel = pend_cb.get_rs_vel(quad_vel)
        print(f"{pend_pose=}")
        
        rate.sleep()