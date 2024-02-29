import rospy
from mavros_msgs.srv import CommandBool


def arm_copter():
    print("\n############ Arming ############")
    rospy.wait_for_service("mavros/cmd/arming")
    arming_client = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
    response = arming_client(1)
    rospy.sleep(0.005)  # 5ms

