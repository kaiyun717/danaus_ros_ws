import rospy
import numpy as np

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import *


class VehicleStateCB:
    def __init__(self) -> None:
        self.state = State()
        self.state_sub = rospy.Subscriber('mavros/state', State, self.state_cb)
        
        self.pose = PoseStamped()
        self.pose_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.pose_cb)

        self.velocity = TwistStamped()
        self.velocity_sub = rospy.Subscriber('mavros/local_position/velocity', TwistStamped, self.velocity_cb)

    def state_cb(self, msg):
        self.state = msg
    
    def pose_cb(self, msg):
        self.pose = msg

    def velocity_cb(self, msg):
        self.velocity = msg

    def get_state(self):
        return self.state
    
    def get_pose(self):
        return self.pose
    
    def get_xyz_pose(self):
        return np.array([self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z])
    
    def get_velocity(self):
        return self.velocity
    
    def get_xyz_velocity(self):
        return np.array([self.velocity.twist.linear.x, self.velocity.twist.linear.y, self.velocity.twist.linear.z])
    
    def get_zyx_angular_velocity(self):
                        # Yaw, Pitch, Roll 
        return np.array([self.velocity.twist.angular.z, self.velocity.twist.angular.y, self.velocity.twist.angular.x])
    