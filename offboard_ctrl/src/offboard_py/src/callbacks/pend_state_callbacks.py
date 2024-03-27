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
            return self._get_rz_pose_sim(vehicle_pose)
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
    
    def _get_rz_pose_sim(self, vehicle_pose):
        # response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='base_link')    # Position changes with base_link frame
        response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='')
        r = response.link_state.pose.position.x - vehicle_pose[0] #(vehicle_pose[0] -0.001995)
        s = response.link_state.pose.position.y - vehicle_pose[1] #(vehicle_pose[1] + 0.000135)
        # r = vehicle_pose[0] - response.link_state.pose.position.x
        # s = vehicle_pose[1] - response.link_state.pose.position.y
        return np.array([r, s])
    
    def get_rz_vel(self, dt=None):
        if self.mode == 'sim':
            return self._get_rz_vel_sim()
        elif self.mode == 'real':
            return self._get_rz_vel_real(dt)
        else:   
            raise ValueError('Invalid mode')

    def _get_rz_vel_sim(self):
        # response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='base_link')    # Position changes with base_link frame
        response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='')
        r = response.link_state.twist.linear.x
        s = response.link_state.twist.linear.y
        return np.array([r, s])

    def _get_rz_vel_real(self, dt):
        r = (self.pose.pose.position.x - self.prev_pose.pose.position.x) / dt
        s = (self.pose.pose.position.y - self.prev_pose.pose.position.y) / dt
        self.prev_pose = self.pose
        return np.array([r, s])
    

if __name__ == "__main__":
    from src.callbacks.fcu_state_callbacks import VehicleStateCB

    rospy.init_node('pendulum_state_cb', anonymous=True)
    
    pend_cb = PendulumCB("sim")
    quad_cb = VehicleStateCB("sim")
    rate = rospy.Rate(2)

    while not rospy.is_shutdown():
        quad_pose = quad_cb.get_xyz_pose()
        print("RZ Pose: ", pend_cb.get_rz_pose(vehicle_pose=quad_pose))
        print("RZ Vel: ", pend_cb.get_rz_vel())
        
        rate.sleep()