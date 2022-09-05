import rospy
import math
import numpy as np

class Target:
    p_change_per_second = 0.1  # on average change once 10 seconds
    linear_speed_std = 0#.01
    angular_speed_std = 0#.01

    def __init__(self, id, init_x, init_y, heading, linear_speed, angular_speed, linear_speed_std, angular_speed_std):
        self.id = id
        self.x = init_x
        self.y = init_y
        self.heading = heading
        self.linear_speed = linear_speed
        self.angular_speed = angular_speed
        self.is_detected = False
        self.time_since_last_change = 0
        self.linear_speed_std = linear_speed_std
        self.angular_speed_std = angular_speed_std

    def propagate(self, del_t):
        '''
        Moves ships with given constant velocity 
        '''

        # Note AJ 2022.09.05: turned off random control cuz we have issues with consistency across different propagation steps
        # if np.random.rand() < self.p_change_per_second * del_t:
        #     # ship prefers to stay with no angular speed
        #     new_angular_speed = np.random.normal(0, self.angular_speed_std)
        #     new_linear_speed = self.linear_speed + np.random.normal(0, self.linear_speed_std)
        #     rospy.logdebug("Target {} changed speed to {}".format(self.id, new_linear_speed))
        #     rospy.logdebug("Target {} changed angular speed to {}".format(self.id, new_angular_speed))
        #     self.angular_speed = new_angular_speed
        #     self.linear_speed = new_linear_speed
        #     self.time_since_last_change = 0

        vx = math.cos(self.heading) * self.linear_speed
        vy = math.sin(self.heading) * self.linear_speed
        self.x += vx * del_t
        self.y += vy * del_t
        self.heading += self.angular_speed * del_t
        
        self.time_since_last_change += del_t