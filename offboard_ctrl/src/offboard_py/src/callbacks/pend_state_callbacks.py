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
    def __init__(self, mode, L_p=0.69) -> None:
        self.pose = PoseStamped()
        self.prev_pose = PoseStamped()
        self.prev_pend_ang = np.array([0, 0])
        
        self.avg_vel = np.array([0, 0])
        self.avg_pend_vel = np.array([0, 0])
        
        self.w_avg = 0.5

        if mode == 'sim':
            # self.pose_sub = rospy.Subscriber('pendulum/pose', PoseStamped, self.pose_cb)
            self.pose_sub = rospy.ServiceProxy('/gazebo/get_link_state', GetLinkState)
        elif mode == 'real':
            self.pose_sub = rospy.Subscriber('vicon/pose/pend/pend', PoseStamped, self.pose_cb)
        else:
            raise ValueError('Invalid mode')

        # self.rs_vel_pub = rospy.Publisher('pendulum/rs_vel', TwistStamped, queue_size=10)
        # self.rs_vel_avg_pub = rospy.Publisher('pendulum/rs_vel_avg', TwistStamped, queue_size=10)

        self.L_p = L_p

        self.mode = mode
        self.dq_len = 10
        self.pose_deque = collections.deque(maxlen=self.dq_len)

    def pose_cb(self, msg):
        # self.pose_deque.append(msg)
        self.prev_pose = self.pose
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

    def get_rs_pose(self, vehicle_pose=None):
        if self.mode == 'sim':
            return self._get_rs_pose_sim(vehicle_pose)
        elif self.mode == 'real':
            return self._get_rs_pose_real(vehicle_pose)
        else:
            raise ValueError('Invalid mode')
    
    def _get_rs_pose_real(self, vehicle_pose):
        # r = self.pose.pose.position.x - vehicle_pose.pose.position.x
        # s = self.pose.pose.position.y - vehicle_pose.pose.position.y
        r = self.pose.pose.position.x - vehicle_pose[0]
        s = self.pose.pose.position.y - vehicle_pose[1]
        if rospy.Time.now() - self.pose.header.stamp > rospy.Duration(0.2):
            rospy.logwarn_throttle(1, "Pendulum pose is too old!")
            return None
        # rospy.loginfo(f"Position: r: {r}, s: {s}")
        return np.array([r, s])
    
    def _get_rs_pose_sim(self, vehicle_pose):
        # response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='base_link')    # Position changes with base_link frame
        response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='')
        r = response.link_state.pose.position.x - vehicle_pose[0] #(vehicle_pose[0] -0.001995)
        s = response.link_state.pose.position.y - vehicle_pose[1] #(vehicle_pose[1] + 0.000135)
        # r = vehicle_pose[0] - response.link_state.pose.position.x
        # s = vehicle_pose[1] - response.link_state.pose.position.y
        return np.array([r, s])
    
    def get_rs_ang(self, vehicle_pose=None):
        rs_pose = self.get_rs_pose(vehicle_pose)
        # xi = np.sqrt(self.L_p**2 - rs_pose[0]**2 - rs_pose[1]**2)
        # roll = np.arctan2(rs_pose[1], xi)       # Roll is about the x-axis. Thus, use y and z.
        # pitch = np.arctan2(rs_pose[0], xi)      # Pitch is about the y-axis. Thus, use x and z.
        roll = rs_pose[1] / self.L_p 
        pitch = rs_pose[0] / self.L_p
        return np.array([roll, pitch])

    def get_rs_ang_vel(self, vehicle_pose=None, vehicle_vel=None):
        rs_vel = self.get_rs_vel(vehicle_vel)
        
        # current_time = self.pose.header.stamp.to_sec()
        # prev_time = self.prev_pose.header.stamp.to_sec()
        # dt = current_time - prev_time
        # if dt < 1e-5:
        #     print("Time difference is too small!")
        #     return np.array([0, 0])
        rs_ang_vel = rs_vel.flatten()/self.L_p
        return np.array([rs_ang_vel[1], rs_ang_vel[0]])
    
    def get_rs_vel(self, vehicle_vel=None, dt=None):
        if self.mode == 'sim':
            return self._get_rs_vel_sim(vehicle_vel)
        elif self.mode == 'real':
            return self._get_rs_vel_real(vehicle_vel)
        else:   
            raise ValueError('Invalid mode')

    def _get_rs_vel_sim(self, vehicle_vel):
        # response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='base_link')    # Position changes with base_link frame
        response = self.pose_sub(link_name='danaus12_pend::pendulum', reference_frame='')
        r = response.link_state.twist.linear.x - vehicle_vel[0]
        s = response.link_state.twist.linear.y - vehicle_vel[1]
        return np.array([r, s])

    def _get_rs_vel_real(self, vehicle_vel):
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
    
    pend_cb = PendulumCB("real")
    quad_cb = VehicleStateCB("real")
    rate = rospy.Rate(100)

    while not rospy.is_shutdown():
        quad_pose = quad_cb.get_xyz_pose()
        quad_vel = quad_cb.get_xyz_velocity()
        pend_pose = pend_cb.get_rs_pose(quad_pose)
        pend_vel = pend_cb.get_rs_vel(quad_vel)
        
        rate.sleep()