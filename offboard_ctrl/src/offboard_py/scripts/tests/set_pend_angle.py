# #!/usr/bin/env python

import rospy
import argparse
import numpy as np
from gazebo_msgs.srv import SetLinkState, GetLinkProperties, SetLinkProperties, SetLinkPropertiesRequest
from gazebo_msgs.msg import LinkState
from geometry_msgs.msg import Pose, Twist, Vector3, Quaternion
from gazebo_msgs.srv import GetLinkState, ApplyJointEffort
from std_msgs.msg import Float64
from geometry_msgs.msg import Vector3

from src.callbacks.pend_state_callbacks import PendulumCB
from src.callbacks.fcu_state_callbacks import VehicleStateCB


class PendulumPIDController:
    def __init__(self, vehicle):
        rospy.init_node('pendulum_state_controller')

        self.quad_cb = VehicleStateCB(mode="sim")
        self.pend_cb = PendulumCB(mode="sim", vehicle=vehicle)

        self.get_link_state_service = rospy.ServiceProxy('/gazebo/set_link_state', GetLinkState)
        self.get_link_properties_service = rospy.ServiceProxy('/gazebo/get_link_properties', GetLinkProperties)
        self.set_link_state_service = rospy.ServiceProxy('/gazebo/set_link_state', SetLinkState)
        self.set_link_properties_service = rospy.ServiceProxy('/gazebo/set_link_properties', SetLinkProperties)

        self.control_rate = rospy.Rate(50)  # Control rate: 50 Hz

        self.vehicle = vehicle
        if self.vehicle == "danaus12_old":
            self.pend_z_pose = 0.531659
            self.mass = 0.67634104 + 0.03133884
            self.L = 0.5
        elif self.vehicle == "danaus12_newold":
            self.pend_z_pose = 0.721659
            self.pend_z_pose = 0.69
            self.L = 0.69
            self.mass = 0.70034104 + 0.046

    def control_pendulum(self):
        tol = 0.05
        req_time = rospy.Duration(10e4)

        consecutive_time = rospy.Duration(0.0)
        start_time = rospy.Time.now()

        while not rospy.is_shutdown():
            link_state = LinkState()
            # link_state.pose.position.x = -0.001995
            # link_state.pose.position.y = 0.000135
            # link_state.pose.position.x = 0.0
            # link_state.pose.position.y = 0.0
            link_state.pose.position.x = 0.001995
            link_state.pose.position.y = -0.000135
            link_state.pose.position.z = self.pend_z_pose
            link_state.link_name = self.vehicle+'::pendulum'
            link_state.reference_frame = 'base_link'

            _ = self.set_link_state_service(link_state)

            quad_xyz = self.quad_cb.get_xyz_pose()
            pendulum_position = self.pend_cb.get_rs_pose(vehicle_pose=quad_xyz)

            # Calculate the norm of the position
            position_norm = np.linalg.norm(pendulum_position)
            print(f"{quad_xyz=}")
            print(f"{pendulum_position=}")
            print(f"{position_norm=}")

            # Check if the norm is less than 0.05m
            if position_norm < tol:
                consecutive_time += rospy.Time.now() - start_time
                if consecutive_time >= req_time:
                    rospy.loginfo("Pendulum position has been less than 0.05m for 0.5 seconds straight.")
                    # self.reset_gravity(True)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--vehicle", type=str)
    args = parser.parse_args()
    vehicle = args.vehicle

    try:
        controller = PendulumPIDController(vehicle)
        controller.control_pendulum()
    except rospy.ROSInterruptException:
        pass
