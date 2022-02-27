import math
import numpy as np

class Target:
    def __init__(self, id, init_x, init_y, heading, linear_speed, angular_speed):
        self.id = id
        self.x = init_x
        self.y = init_y
        self.heading = heading
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.is_detected = False

    def propagate(self, del_t):
        '''
        Moves ships with given constant velocity 
        '''
        vx = math.cos(self.heading) * self.linear_speed
        vy = math.sin(self.heading) * self.linear_speed
        self.x += vx * del_t
        self.y += vy * del_t
        self.heading += self.angular_speed * del_t
        