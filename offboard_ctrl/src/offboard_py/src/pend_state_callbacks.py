import rospy

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import *

class PendulumCB:
    def __init__(self) -> None:
        self.pose = PoseStamped()
        self.pose_sub = rospy.Subscriber('vicon/pose/pendulum/pendulum', PoseStamped, self.pose_cb)

    def pose_cb(self, msg):
        self.pose = msg

    def get_pose(self):
        return self.pose