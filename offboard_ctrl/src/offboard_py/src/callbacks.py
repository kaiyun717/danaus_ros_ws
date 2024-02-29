import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

current_state = State()
current_vision_pose = PoseStamped()


def state_callback(msg):
    global current_state
    current_state = msg

def vision_pose_callback(msg):
    global current_vision_pose
    current_vision_pose = msg