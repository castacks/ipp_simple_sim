import math
import random
import numpy as np
from vehicle import *
from target import *
from geographic_msgs.msg import GeoPose


class Environment:
    def __init__(self, targets=[], max_omega=5, 
                init_x=0, init_y=0, init_z=0, init_phi=0, K_p=0.01,
                vehicle_l=3, vel=5, n_rand_targets=-1, del_t=1, waypt_threshold=5):
        # if initial position not specified, randomly spawn vehicle between (50, 1000)
        if init_x is 0 and init_y is 0 and init_z is 0:
            init_x = random.randrange(50, 1000)
            init_y = random.randrange(50, 1000)
            init_z = random.randrange(20, 120, 20) # discretized by step-size 20
            init_phi = random.uniform(0, np.pi)
        
        # vehicle pose
        self.init_x = init_x
        self.init_y = init_y
        self.init_z = init_z
        self.init_phi = init_phi 

        self.del_t = del_t
        
        self.vehicle_l = vehicle_l  # vehicle length
        self.vel = vel  # velocity
        self.max_omega = max_omega  # max angular velocity

        self.targets = targets

        self.global_waypt_list = []

        if self.targets == []:
            # if targets not specified, randomly generate between 1-10 targets
            if n_rand_targets == -1:
                self.n_rand_targets = random.randrange(1, 10)
            else:
                self.n_rand_targets = n_rand_targets
        
        self.waypt_threshold = waypt_threshold

        self.K_p = K_p  # proportionality constant for PID controller

        self.generate_targets()
        self.vehicle = self.init_vehicle()
    
    def generate_targets(self)->None:
        if self.targets == []:
            for i in range(self.n_rand_targets):
                # creating list of random target objects
                self.targets.append(
                    Target(
                        init_x = random.randrange(50, 1000),
                        init_y = random.randrnge(50, 1000),
                        vel = [random.uniform(0, 3), random.uniform(0, 3)],
                        data = bool(random.getrandbits(1))  # 1=target, 0=non-target
                    )
                )
        else:
            t_tuples = self.targets
            self.targets = []
            for target in t_tuples:
                self.targets.append(
                    Target(
                        init_x=target[0],
                        init_y=target[1],
                        vel=target[3],
                        data=target[4]
                    )
                )
    
    def init_vehicle(self)->Vehicle:
        return Vehicle(self.init_x,
                        self.init_y,
                        self.init_z,
                        self.init_phi,
                        self.vehicle_l,
                        self.vel)
    
    def get_sensor_measurements():
        pass

    def traverse(self, flag):
        have_wypts = flag
        while(have_wypts):
            if len(self.global_waypt_list) == 0:
                have_wypts = False
            else:
                next_position = np.array([self.global_waypt_list[0].position.position.x,
                                            self.global_waypt_list[0].position.position.y,
                                            self.global_waypt_list[0].position.position.z])
                dist_to_waypt = np.linalg.norm(self.vehicle.X, next_position)
                
                # update waypoint list if reached waypoint
                if dist_to_waypt < self.waypt_threshold:
                    print ("Reached waypoint -> ", next_position)
                    self.global_waypt_list.pop(0)
                
                # else keep trying to navigate to next waypoint
                else:
                    omega = self.vehicle.go_to_goal(self.max_omega, next_position, self.K_p)
                    self.vehicle.phi += self.del_t*omega
                    self.vehicle.x += self.del_t*self.vel*math.cos(self.vehicle.phi)
                    self.vehicle.y += self.del_t*self.vel*math.sin(self.vehicle.phi)                                        
                    self.vehicle.z += self.del_t*self.vel                  
    
    def update_waypts(self, new_wpts)->None:
        self.global_waypt_list.append(new_wpts)
        self.traverse(True)
    
    def update_states(self)->None:
        for target in self.targets:
            target.propagate(self.del_t)
        self.get_sensor_measurements()
