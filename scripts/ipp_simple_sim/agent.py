import math
import rospy
import numpy as np
from ipp_simple_sim.msg import gimbalbounds
class Agent:
    '''
    Agent frame is ENU
    '''
    def __init__(self, agent_num,init_x, init_y, init_z, agent_l, hvel, vvel, init_roll=0, init_pitch=0, init_yaw=0, init_gimbal_yaw=0,init_gimbal_yaw_min = 0, init_gimbal_yaw_max = 0):
        self.agent_num = agent_num
        self.x = init_x
        self.y = init_y
        self.z = init_z
        self.roll = init_roll  # roll angle
        self.pitch = init_pitch  # pitch angle
        self.yaw = init_yaw  # yaw angle
        self.gimbal_yaw = init_gimbal_yaw  # gimbal yaw angle
        self.vel = hvel
        self.vvel = vvel
        self.gimbal_yaw_min = init_gimbal_yaw_min
        self.gimbal_yaw_max = init_gimbal_yaw_max
        self.prev_gimbal_direction = 1

        self.gimbal_bounds_pub = rospy.Publisher('uav1/gimbal_bounds', gimbalbounds, queue_size=10)

    def go_to_goal(self, max_omega, max_zvel, next_waypoint, K_p, K_p_z):
        '''
        Returns angular velocity and velocity in z-axis towards desired direction
        '''
        e = next_waypoint - [self.x, self.y, self.z]  # dist to desired position
        yaw_d = math.atan2(e[1], e[0])  # desired yaw
        z_error = e[2]
        
        omega = K_p*math.atan2(math.sin(yaw_d - self.yaw), math.cos(yaw_d - self.yaw))  # omega is desired heading
        z_d = K_p_z*z_error

        if omega > max_omega:
            omega = max_omega  # setting max angular vel
        if z_d > max_zvel:
            z_d = max_zvel
        return omega, z_d

    def position_uncertainty(self):
        '''
        Returns position uncertainty
        '''

        # hard-coding values for now
        sigma_x = 0.1
        sigma_y = 0.1
        sigma_z = 0.1
        sigma_roll = 0.1
        sigma_pitch = 0.1
        sigma_yaw = 0.1
        return [sigma_x, sigma_y, sigma_z, sigma_roll, sigma_pitch, sigma_yaw]

    def publish_gimbal_bounds(self,gimbal_yaw_min,gimbal_yaw_max):

        gimbal_bounds = gimbalbounds()
        gimbal_bounds.gimbal_yaw_min = gimbal_yaw_min
        gimbal_bounds.gimbal_yaw_max = gimbal_yaw_max
        self.gimbal_bounds_pub.publish(gimbal_bounds)
