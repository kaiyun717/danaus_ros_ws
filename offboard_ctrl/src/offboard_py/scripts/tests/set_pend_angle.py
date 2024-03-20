# #!/usr/bin/env python

import rospy
import numpy as np
from gazebo_msgs.srv import SetLinkState, GetLinkProperties, SetLinkProperties, SetLinkPropertiesRequest
from gazebo_msgs.msg import LinkState
from geometry_msgs.msg import Pose, Twist, Vector3, Quaternion
from gazebo_msgs.srv import GetLinkState, ApplyJointEffort
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

from src.callbacks.pend_state_callbacks import PendulumCB


class PendulumPIDController:
    def __init__(self):
        rospy.init_node('pendulum_state_controller')

        self.pend_cb = PendulumCB(mode="sim")

        self.get_link_state_service = rospy.ServiceProxy('/gazebo/set_link_state', GetLinkState)
        self.get_link_properties_service = rospy.ServiceProxy('/gazebo/get_link_properties', GetLinkProperties)
        self.set_link_state_service = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        self.set_link_properties_service = rospy.ServiceProxy('/gazebo/set_link_properties', SetLinkProperties)

        self.control_rate = rospy.Rate(50)  # Control rate: 50 Hz

    def control_pendulum(self):
        tol = 0.05
        req_time = rospy.Duration(0.5)

        consecutive_time = rospy.Duration(0.0)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            link_state = LinkState()
            link_state.link_name = 'danaus12_pend::pendulum'
            link_state.reference_frame = 'base_link'

            response = self.set_link_state_service(link_state)

            pendulum_position = self.pend_cb.get_rz_pose()

            # Calculate the norm of the position
            position_norm = np.linalg.norm(pendulum_position)

            # Check if the norm is less than 0.05m
            if position_norm < tol:
                consecutive_time += rospy.Time.now() - start_time
                if consecutive_time >= req_time:
                    rospy.loginfo("Pendulum position has been less than 0.05m for 0.5 seconds straight.")
                    self.reset_gravity(True)
                    return True
            else:
                consecutive_time = rospy.Duration(0.0)
                start_time = rospy.Time.now()

            self.control_rate.sleep()

    def reset_gravity(self, grav=True):
        curr_pend_properties = self.get_link_properties_service(link_name='danaus12_pend::pendulum')
        new_pend_properties = SetLinkPropertiesRequest(
            link_name='danaus12_pend::pendulum',
            gravity_mode=grav,
            com=curr_pend_properties.com,
            mass=curr_pend_properties.mass,
            ixx=curr_pend_properties.ixx,
            ixy=curr_pend_properties.ixy,
            ixz=curr_pend_properties.ixz,
            iyy=curr_pend_properties.iyy,
            iyz=curr_pend_properties.iyz,
            izz=curr_pend_properties.izz
        )
        # new_pend_properties.link_name = 'danaus12_pend::pendulum'
        # new_pend_properties.gravity_mode = grav
        # new_pend_properties.com = curr_pend_properties.com
        # new_pend_properties.mass = curr_pend_properties.mass
        # new_pend_properties.ixx = curr_pend_properties.ixx
        # new_pend_properties.ixy = curr_pend_properties.ixy
        # new_pend_properties.ixz = curr_pend_properties.ixz
        # new_pend_properties.iyy = curr_pend_properties.iyy
        # new_pend_properties.iyz = curr_pend_properties.iyz
        # new_pend_properties.izz = curr_pend_properties.izz
        self.set_link_properties_service(new_pend_properties)




if __name__ == '__main__':
    try:
        controller = PendulumPIDController()
        controller.control_pendulum()
    except rospy.ROSInterruptException:
        pass
