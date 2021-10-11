import math
import numpy as np

class Target:
    def __init__(self, init_x, init_y, vel, data):
        self.x = init_x
        self.y = init_y
        self.vel = vel

        self.X = np.array([self.x, self.y])

        self.data = data

    def propagate(self, del_t):
        '''
        Moves ships with given constant velocity 
        '''
        self.X[0] += self.vel[0] * del_t
        self.X[1] += self.vel[1] * del_t
        