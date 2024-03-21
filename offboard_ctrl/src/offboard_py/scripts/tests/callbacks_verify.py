import rospy
from src.callbacks.fcu_state_callbacks import VehicleStateCB

if __name__ == "__main__":
    
    rospy.init_node("velocity_verify_node")

    quad_cb = VehicleStateCB()
    
    while (not rospy.is_shutdown()):
        print("XYZ Velocity: ", quad_cb.get_xyz_velocity())
        print("ZXY Angular Velocity: ", quad_cb.get_zyx_angular_velocity())
        rospy.sleep(0.05)