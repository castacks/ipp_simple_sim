import math
import numpy as np

class Target:
    p_change_per_second = 0.1  # on average change once 10 seconds
    linear_speed_std = 0.01
    angular_speed_std = 0.01

    def __init__(self, id, init_x, init_y, heading, linear_speed, angular_speed):
        self.id = id
        self.x = init_x
        self.y = init_y
        self.heading = heading
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.is_detected = False
        self.time_since_last_change = 0

    def propagate(self, del_t):
        '''
        Moves ships with given constant velocity 
        '''

        if np.random.rand() < self.p_change_per_second * del_t:
            # ship prefers to stay with no angular speed
            new_angular_speed = np.random.normal(0, self.angular_speed_std)
            new_linear_speed = self.linear_speed + np.random.normal(0, self.linear_speed_std)
            # print("Target {} changed speed to {}".format(self.id, new_linear_speed))
            # print("Target {} changed angular speed to {}".format(self.id, new_angular_speed))
            self.angular_speed = new_angular_speed
            self.linear_speed = new_linear_speed
            self.time_since_last_change = 0

        vx = math.cos(self.heading) * self.linear_speed
        vy = math.sin(self.heading) * self.linear_speed
        self.x += vx * del_t
        self.y += vy * del_t
        self.heading += self.angular_speed * del_t
        
        self.time_since_last_change += del_t