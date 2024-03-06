import rospy

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import *


class StateCB:
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
    
    def get_velocity(self):
        return self.velocity