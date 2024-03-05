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
            rospy.loginfo(arm_srv_msg)
        except rospy.ServiceException as e:
            print("Service arming call failed: %s" % e)

    def set_disarm(self):
        rospy.wait_for_service('mavros/cmd/arming')
        try:
            arm_service = rospy.ServiceProxy("mavros/cmd/arming", CommandBool)
            self.arm_cmd.value = False
            disarm_srv_msg = arm_service.call(self.arm_cmd)
            rospy.loginfo(disarm_srv_msg)
        except rospy.ServiceException as e:
            print("Service disarming call failed: %s" % e)

    def set_takeoff(self, height=1):
        rospy.wait_for_service('mavros/cmd/takeoff')
        try:
            takeoff_service = rospy.ServiceProxy('mavros/cmd/takeoff', CommandTOL)
            takeoff_srv_msg = takeoff_service(altitude=height)
            rospy.loginfo(takeoff_srv_msg)
        except rospy.ServiceException as e:
            print("Service takeoff call failed: %s" % e)

    def set_land(self):
        rospy.wait_for_service('mavros/cmd/land')
        try:
            land_service = rospy.ServiceProxy('mavros/cmd/land', CommandTOL)
            land_srv_msg = land_service()
            rospy.loginfo(land_srv_msg)
        except rospy.ServiceException as e:
            print("Service land call failed: %s" % e)

    def set_offboard_mode(self):
        rospy.wait_for_service('mavros/set_mode')
        try:
            flight_mode_service = rospy.ServiceProxy('mavros/set_mode', SetMode)
            offb_mode_srv_msg = flight_mode_service.call(self.offb_mode_cmd)
            return offb_mode_srv_msg
        except rospy.ServiceException as e:
            print("Service offboard call failed: %s" % e)