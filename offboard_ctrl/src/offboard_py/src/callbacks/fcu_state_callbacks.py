import rospy
import numpy as np
import tf.transformations as tf


from geometry_msgs.msg import PoseStamped, TwistStamped, TransformStamped
from mavros_msgs.msg import State, AttitudeTarget
from mavros_msgs.srv import *
from gazebo_msgs.msg import ModelStates


class VehicleStateCB:
    def __init__(self, mode) -> None:
        self.state = State()
        self.state_sub = rospy.Subscriber('mavros/state', State, self.state_cb)
        self.mode = mode
        self.pose = PoseStamped()
        self.velocity = TwistStamped()
        self.accel = TwistStamped()
        if mode == "real":
            # self.pose_sub = rospy.Subscriber('mavros/local_position/pose', PoseStamped, self.pose_cb)
            self.vicon_sub = rospy.Subscriber('vicon/danaus12/danaus12', TransformStamped, self.pose_cb)
        elif mode == "sim":
            self.pose_sub = rospy.Subscriber('gazebo/model_states', ModelStates, self.pose_cb)
        
        if mode == "real":
            self.velocity_sub = rospy.Subscriber('mavros/local_position/velocity_local', TwistStamped, self.velocity_cb)
            self.velocity_body_sub = rospy.Subscriber('mavros/local_position/velocity_body', TwistStamped, self.velocity_body_cb)
        elif mode == "sim":
            self.velocity_sub = rospy.Subscriber('mavros/local_position/velocity_local', TwistStamped, self.velocity_cb)
            self.velocity_body_sub = rospy.Subscriber('mavros/local_position/velocity_body', TwistStamped, self.velocity_body_cb)
            # self.accel_body_sub = rospy.Subscriber('mavros/imu/data_raw', TwistStamped, self.velocity_body_cb)

        self.targ_att_sub = rospy.Subscriber("/mavros/setpoint_raw/target_attitude", AttitudeTarget, self.targ_att_cb)

    def targ_att_cb(self, data):
        # Assuming the thrust is in the z component of the thrust vector
        self.thrust = data.thrust

    def state_cb(self, msg):
        self.state = msg
    
    def pose_cb(self, msg):
        if self.mode == "sim":
            # self.pose = PoseStamped()
            self.pose.pose = msg.pose[2]
            self.velocity.twist = msg.twist[2]
            # print(type(msg.pose[2]))
        elif self.mode == "real":
            # self.pose = msg
            self.pose.pose.position.x = msg.transform.translation.x
            self.pose.pose.position.y = msg.transform.translation.y
            self.pose.pose.position.z = msg.transform.translation.z
            self.pose.pose.orientation = msg.transform.rotation

    def velocity_cb(self, msg):
        if self.mode == "sim":
            # pass
            self.velocity = msg
            # self.velocity.twist = msg.twist[2]
        elif self.mode == "real":
            self.velocity = msg

    def velocity_body_cb(self, msg):
        if self.mode == "sim":
            # pass
            self.velocity_body = msg
        elif self.mode == "real":
            self.velocity_body = msg

    def get_state(self):
        return self.state
    
    def get_pose(self):
        return self.pose
    
    def get_xyz_pose(self):
        return np.array([self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z])
    
    def get_velocity(self):
        return self.velocity
    
    def get_xyz_velocity(self):
        return np.array([self.velocity.twist.linear.x, self.velocity.twist.linear.y, self.velocity.twist.linear.z])
        # return np.array([self.pose.twist.linear.x, self.pose.twist.linear.y, self.pose.twist.linear.z])
    
    def get_zyx_angles(self):
        roll, pitch, yaw = tf.euler_from_quaternion([self.pose.pose.orientation.x, self.pose.pose.orientation.y, self.pose.pose.orientation.z, self.pose.pose.orientation.w])
        return np.array([yaw, pitch, roll])

    def get_zyx_angular_velocity(self):
                        # Yaw, Pitch, Roll 
        return np.array([self.velocity.twist.angular.z, self.velocity.twist.angular.y, self.velocity.twist.angular.x])
    
    def get_zyx_angular_velocity_body(self):
        # Yaw, Pitch, Roll 
        return np.array([self.velocity_body.twist.angular.z, self.velocity_body.twist.angular.y, self.velocity_body.twist.angular.x])
    
    def get_xyz_angles(self):
        roll, pitch, yaw = tf.euler_from_quaternion([self.pose.pose.orientation.x, self.pose.pose.orientation.y, self.pose.pose.orientation.z, self.pose.pose.orientation.w])
        return np.array([roll, pitch, yaw])
    
    def get_xyz_angular_velocity(self):
        return np.array([self.velocity.twist.angular.x, self.velocity.twist.angular.y, self.velocity.twist.angular.z])

    def get_xyz_angular_velocity_body(self):
        # Yaw, Pitch, Roll 
        return np.array([self.velocity_body.twist.angular.x, self.velocity_body.twist.angular.y, self.velocity_body.twist.angular.z])
        
if __name__ == "__main__":
    rospy.init_node('vehicle_state_cb', anonymous=True)
    
    state_cb = VehicleStateCB("sim")
    rate = rospy.Rate(2)

    while not rospy.is_shutdown():
        print("XYZ Pose: ", state_cb.get_xyz_pose())
        # print("Yaw, Pitch, Roll: ", state_cb.get_zyx_angles()*180/np.pi)
        