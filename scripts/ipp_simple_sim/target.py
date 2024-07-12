import rospy
import math
import numpy as np

class Target:
    """Class to represent a target in the simulation.

    """

    def __init__(
        self, 
        id: int, 
        init_x: float, 
        init_y: float, 
        heading: float, 
        linear_speed: float, 
        angular_speed: float, 
        linear_speed_std: float, 
        angular_speed_std: float,
        decay_rate: float,
    ) -> None:
        """Initialize a target with the given parameters.

        Args:
            id (int): The unique identifier of the target.
            init_x (float): The initial x-coordinate of the target.
            init_y (float): The initial y-coordinate of the target.
            heading (float): The initial heading of the target.
            linear_speed (float): The initial linear speed of the target.
            angular_speed (float): The initial angular speed of the target.
            linear_speed_std (float): The standard deviation of the linear speed noise.
            angular_speed_std (float): The standard deviation of the angular speed noise.
            decay_rate (float): The rate at which the target's speed decays. Pass in 0 for no decay.
        """
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
        self.decay_rate = decay_rate
        self.prev_time = -1
        if decay_rate == 0:
            self.lifespan = float('inf')
        else:
            self.lifespan = np.random.exponential(1/decay_rate)

    def propagate(self, del_t: float) -> None:
        """Propagate the target's position and speed forward in time.

        Args:
            del_t (float): The time step to propagate the target by.
        """

        curr_time  = rospy.get_time()
        delta_t = del_t
        if self.prev_time == -1:
            self.prev_time = rospy.get_time()
        else:
            delta_t = curr_time - self.prev_time

        vx = math.cos(self.heading) * self.linear_speed + np.random.normal(0, self.linear_speed_std)
        vy = math.sin(self.heading) * self.linear_speed + np.random.normal(0, self.linear_speed_std)
        self.x += vx * delta_t
        self.y += vy * delta_t
        self.heading += self.angular_speed * delta_t + np.random.normal(0, self.angular_speed_std)
        
        self.time_since_last_change += delta_t

        self.prev_time = curr_time
        self.lifespan -= delta_t
        print (f"TARGET: {self.id} LIFESPAN: ", self.lifespan)
    
    def __str__(self) -> str:
        return "True Target {} at ({}, {}) with heading {} and speed {}".format(self.id, self.x, self.y, self.heading, self.linear_speed)