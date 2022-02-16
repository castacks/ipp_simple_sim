import math
import numpy as np

class Target:
    def __init__(self, init_x, init_y, vel, phi):
        self.x = init_x
        self.y = init_y
        self.vel = vel
        self.heading = phi
        self.is_detected = False

    def propagate(self, del_t):
        '''
        Moves ships with given constant velocity 
        '''
        self.x += self.vel[0] * del_t
        self.y += self.vel[1] * del_t
        