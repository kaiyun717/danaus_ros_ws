import rospy

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, CommandBoolRequest, CommandTOL, SetMode, SetModeRequest


class FcuModes:
    def __init__(self) -> None:
        self.arm_cmd = CommandBoolRequest()
        self.offb_mode_cmd = SetModeRequest()
        self.offb_mode_cmd.custom_mode = 'OFFBOARD'

    def set_arm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            arm_service = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
            self.arm_cmd.value = True
            arm_srv_msg = arm_service.call(self.arm_cmd)
            return arm_srv_msg
        except rospy.ServiceException as e:
            rospy.loginfo("Service arming call failed: %s" % e)

    def set_disarm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            arm_service = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
            self.arm_cmd.value = False
            disarm_srv_msg = arm_service.call(self.arm_cmd)
            return disarm_srv_msg
        except rospy.ServiceException as e:
            rospy.loginfo("Service disarming call failed: %s" % e)

    def set_offboard_mode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flight_mode_service = rospy.ServiceProxy('mavros/set_mode', SetMode)
            offb_mode_srv_msg = flight_mode_service.call(self.offb_mode_cmd)
            return offb_mode_srv_msg
        except rospy.ServiceException as e:
            rospy.loginfo("Service offboard call failed: %s" % e)