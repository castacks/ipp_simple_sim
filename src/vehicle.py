import math
import numpy as np


class Vehicle:
    def __init__(self, init_x, init_y, init_z, init_phi, vehicle_l, vel):
        self.x = init_x
        self.y = init_y
        self.z = init_z
        self.phi = init_phi
        self.vel = vel
        
        self.X = np.array([self.x, self.y, self.z])
        self.vehicle_l = vehicle_l
        self.pose_tip =  [self.x + self.vehicle_l/2 * math.cos(self.phi), self.y + self.vehicle_l/2 * math.sin(self.phi)]
        self.pose_bottom =  [self.x - self.vehicle_l/2 * math.cos(self.phi), self.y - self.vehicle_l/2 * math.sin(self.phi)]

    def go_to_goal(self, max_omega):
        e = self.data["goalX"] - self.X  # dist to desired position
        phi_d = math.atan2(e[1], e[0])  # desired phi
        omega = self.data["K_p"]*math.atan2(math.sin(phi_d - self.phi), math.cos(phi_d - self.phi))  # omega is desired heading
        if omega > max_omega:
            omega = max_omega  # setting max angular vel
        return omega
