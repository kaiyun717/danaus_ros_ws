import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest

# from src.callbacks import state_callback, vision_pose_callback
from src.arm import arm_copter

current_state = State()
current_vision_pose = PoseStamped()


def state_callback(msg):
    global current_state
    current_state = msg

def vision_pose_callback(msg):
    global current_vision_pose
    current_vision_pose = msg

def main(node_name):
    rospy.init_node(node_name)
    
    state_sub = rospy.Subscriber("mavros/state", State, callback=state_callback)
    vision_pose_sub = rospy.Subscriber("mavros/vision_pose/pose", PoseStamped, callback=vision_pose_callback)

    arm_copter()
    
    # Setpoint publishing must be faster than 2Hz
    rate = rospy.Rate(400)  # Setpoint rate is set to be 40Hz

    


if __name__ == "__main__":
    node_name = "mission_node"


