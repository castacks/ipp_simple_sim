import math
import numpy as np


class Agent:
    '''
    Agent frame is ENU
    '''
    def __init__(self, agent_num,init_x, init_y, init_z, agent_l, hvel, vvel, init_theta=0, init_psi=0, init_phi=0):
        self.agent_num = agent_num
        self.x = init_x
        self.y = init_y
        self.z = init_z
        self.theta = init_theta  # roll angle
        self.psi = init_psi  # pitch angle
        self.phi = init_phi  # yaw angle
        self.vel = hvel
        self.vvel = vvel
        
        # self.X = np.array([self.x, self.y, self.z])
        self.agent_l = agent_l
        self.pose_tip =  [self.x + self.agent_l/2 * math.cos(self.phi), self.y + self.agent_l/2 * math.sin(self.phi)]
        self.pose_bottom =  [self.x - self.agent_l/2 * math.cos(self.phi), self.y - self.agent_l/2 * math.sin(self.phi)]

    def go_to_goal(self, max_omega, max_zvel, next_waypoint, K_p, K_p_z):
        '''
        Returns angular velocity and velocity in z-axis towards desired direction
        '''
        e = next_waypoint - [self.x, self.y, self.z]  # dist to desired position
        psi_d = math.atan2(e[1], e[0])  # desired phi
        z_error = e[2]
        
        omega = K_p*math.atan2(math.sin(psi_d - self.psi), math.cos(psi_d - self.psi))  # omega is desired heading
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
        sigma_psi = 0.1
        sigma_phi = 0.1
        sigma_theta = 0.1
        return [sigma_x, sigma_y, sigma_z, sigma_psi, sigma_phi, sigma_theta]