import rospy

from geometry_msgs.msg import Point, PoseStamped
from mavros_msgs.msg import *
from mavros_msgs.srv import *

from src.fcu_modes import FcuModes
from src.fcu_state_callbacks import StateCB
from src.pend_state_callbacks import PendStateCB
from src.utils import setpoint_reached

if __name__ == "__main__":
    rospy.init_node("pend_ang_ctrl_node")
    
    # Setpoint publisher
    local_pos_pub = rospy.Publisher("mavros/setpoint_position/local", PoseStamped, queue_size=10)

    # Flight modes class
    # Flight modes are activated using ROS services
    fcu_mode = FcuModes()
    fcu_state_cb = StateCB()
    pend_state_cb = PendStateCB()

    # This is 50 Hz
    rate = rospy.Rate(20)

    while not rospy.is_shutdown() and not fcu_state_cb.get_state().connected:
        rate.sleep()
    
    takeoff_pose = PoseStamped()
    takeoff_pose.pose.position.x = 0
    takeoff_pose.pose.position.y = 0
    takeoff_pose.pose.position.z = 1

    for i in range(100):
        if rospy.is_shutdown():
            break
        local_pos_pub.publish(takeoff_pose)
        rate.sleep()
    
    last_request = rospy.Time.now()

    # OFFBOARD mode and arming
    # Takeoff to 1m
    while not rospy.is_shutdown():
        if fcu_state_cb.get_state().mode == "OFFBOARD" and fcu_state_cb.get_state().armed:
            local_pos_pub.publish(takeoff_pose)
            rospy.loginfo("Takeoff to 1m complete.")
            break
        elif fcu_state_cb.get_state().mode != "OFFBOARD" and (rospy.Time.now() - last_request > rospy.Duration(5.0)):
            fcu_mode.set_offboard_mode()
            last_request = rospy.Time.now()
        else:
            if not fcu_state_cb.get_state().armed and (rospy.Time.now() - last_request > rospy.Duration(5.0)):
                fcu_mode.set_arm()
                last_request = rospy.Time.now()
        
        rate.sleep()

    # Pendulum control
    while not rospy.is_shutdown():



    