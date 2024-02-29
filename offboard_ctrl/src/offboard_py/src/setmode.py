import rospy
from mavros_msgs.srv import CommandBool, SetMode

def setmode(current_mode, next_mode, delay):
    """
    http://docs.ros.org/jade/api/mavros_msgs/html/srv/SetMode.html
    """
    print("\n######### Set Mode #########")
    rospy.wait_for_service("/mavros/set_mode")
    modes = rospy.ServiceProxy("/mavros/set_mode", SetMode)
    resp = modes(current_mode, next_mode)
    rospy.sleep(delay)
    return


