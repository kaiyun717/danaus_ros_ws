import rospy
import math
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from gazebo_msgs.srv import GetLinkState
from mavros_msgs.msg import State
from mavros_msgs.srv import *

class PendulumCB:
    def __init__(self, mode) -> None:
        self.pose = PoseStamped()
        self.prev_pose = PoseStamped()

        if mode == 'sim':
            # self.pose_sub = rospy.Subscriber('pendulum/pose', PoseStamped, self.pose_cb)
            self.pose_sub = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        elif mode == 'real':
            self.pose_sub = rospy.Subscriber('vicon/pose/pendulum/pendulum', PoseStamped, self.pose_cb)
        else:
            raise ValueError('Invalid mode')

        self.mode = mode

    def pose_cb(self, msg):
        self.pose = msg

    def get_local_pose(self):
        return self.pose
    
    def get_rel_pose(self, vehicle_pose: PoseStamped):
        r = self.pose.pose.position.x - vehicle_pose.pose.position.x
        s = self.pose.pose.position.y - vehicle_pose.pose.position.y
        dz = self.pose.pose.position.z - vehicle_pose.pose.position.z

        pitch_rad = math.atan2(r, dz)
        roll_rad = math.atan2(s, dz)

        if abs(roll_rad) > 20 / 180 * math.pi:
            roll_rad = 20 / 180 * math.pi * roll_rad / abs(roll_rad)

        if abs(pitch_rad) > 20 / 180 * math.pi:
            pitch_rad = 20 / 180 * math.pi * pitch_rad / abs(pitch_rad)

        return r, s, dz, pitch_rad, roll_rad

    def get_rz_pose(self, vehicle_pose=None):
        if self.mode == 'sim':
            return self._get_rz_pose_sim()
        elif self.mode == 'real':
            return self._get_rz_pose_real(vehicle_pose)
        else:
            raise ValueError('Invalid mode')
    
    def _get_rz_pose_real(self, vehicle_pose):
        # r = self.pose.pose.position.x - vehicle_pose.pose.position.x
        # s = self.pose.pose.position.y - vehicle_pose.pose.position.y
        r = self.pose.pose.position.x - vehicle_pose[0]
        s = self.pose.pose.position.y - vehicle_pose[1]
        return np.array([r, s])
    
    def _get_rz_pose_sim(self):
        response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='base_link')
        r = response.link_state.pose.position.x
        s = response.link_state.pose.position.y
        return np.array([r, s])
    
    def get_rz_vel(self, dt=None):
        if self.mode == 'sim':
            return self._get_rz_vel_sim()
        elif self.mode == 'real':
            return self._get_rz_vel_real(dt)
        else:   
            raise ValueError('Invalid mode')

    def _get_rz_vel_sim(self):
        response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='base_link')
        r = response.link_state.twist.linear.x
        s = response.link_state.twist.linear.y
        return np.array([r, s])

    def _get_rz_vel_real(self, dt):
        r = (self.pose.pose.position.x - self.prev_pose.pose.position.x) / dt
        s = (self.pose.pose.position.y - self.prev_pose.pose.position.y) / dt
        self.prev_pose = self.pose
        return np.array([r, s])