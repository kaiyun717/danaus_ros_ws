import rospy
import math
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import *

class PendulumCB:
    def __init__(self) -> None:
        self.pose = PoseStamped()
        self.prev_pose = PoseStamped()
        self.pose_sub = rospy.Subscriber('vicon/pose/pendulum/pendulum', PoseStamped, self.pose_cb)

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

    def get_rz_pose(self, vehicle_pose: PoseStamped):
        r = self.pose.pose.position.x - vehicle_pose.pose.position.x
        s = self.pose.pose.position.y - vehicle_pose.pose.position.y
        return np.array([r, s])
    
    def get_rz_vel(self, dt):
        r = (self.pose.pose.position.x - self.prev_pose.pose.position.x) / dt
        s = (self.pose.pose.position.y - self.prev_pose.pose.position.y) / dt
        self.prev_pose = self.pose
        return np.array([r, s])