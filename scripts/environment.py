import math
import random
from turtle import heading
import numpy as np
from vehicle import *
from target import *
from sensor import *
from geographic_msgs.msg import GeoPose


class Environment:
    def __init__(self, list_of_target_dicts=[], max_omega=5, max_zvel=5,
                 init_x=None, init_y=None, init_z=None, init_psi=None,
                 K_p=0.01, K_p_z=0.01,
                 vehicle_l=3, hvel=5, vvel=2, n_rand_targets=-1, del_t=1,
                 waypt_threshold=5,
                 sensor_focal_length=5, sensor_width=10, sensor_height=10, sensor_a=1,
                 sensor_b=1, sensor_d=1, sensor_g=1, sensor_h=1, sensor_pitch=20):
        '''
        Setup simulation environment
        '''
        # if initial position not specified, randomly spawn vehicle between (50, 1000)
        init_x = random.randrange(50, 1000) if init_x is None else init_x
        init_y = random.randrange(50, 1000) if init_y is None else init_y
        init_z = random.randrange(20, 120,
                                  20) if init_z is None else init_z  # discretized by step-size 20
        init_psi = random.uniform(0, np.pi) if init_psi is None else init_psi

        # drone pose
        self.init_x = init_x
        self.init_y = init_y
        self.init_z = init_z
        self.init_psi = init_psi

        self.del_t = del_t

        self.vehicle_l = vehicle_l  # vehicle length
        self.hvel = hvel  # horizontal velocity
        self.vvel = vvel  # vertical velocity
        self.max_omega = max_omega  # max angular velocity
        self.max_zvel = max_zvel  # max vertical velocity

        self.sensor_focal_length = sensor_focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.sensor_a = sensor_a
        self.sensor_b = sensor_b
        self.sensor_d = sensor_d
        self.sensor_g = sensor_g
        self.sensor_h = sensor_h
        self.sensor_pitch = sensor_pitch

        # if targets not specified, randomly generate between 1-10 targets
        self.n_rand_targets = random.randrange(1, 10) if not list_of_target_dicts and n_rand_targets == -1 else n_rand_targets

        self.targets = self.generate_targets(list_of_target_dicts, self.n_rand_targets)

        self.global_waypt_list = []

        self.waypt_threshold = waypt_threshold

        self.K_p = K_p  # x-y proportionality constant for PID controller
        self.K_p_z = K_p_z  # z-axis proportionality constant for PID controller

        self.vehicle = self.init_vehicle()
        self.sensor = self.init_sensor()

        self.curr_waypt_num = 0

    def generate_targets(self, list_of_target_dicts, n_rand_targets=None):
        '''
        Generates ships with initial positions
        '''

        # when no targets specified
        if not list_of_target_dicts:
            if n_rand_targets is None:
                raise ValueError(
                    "Passed in no targets but didn't pass in n_rand_targets")
            targets = [
                Target(
                    id=idx,
                    init_x=np.random.uniform(-600, 600),
                    init_y=np.random.uniform(-600, 600),
                    heading=np.random.uniform(0, 2 * 3.1416),
                    linear_speed=np.random.normal(6, 2),
                    angular_speed=np.random.normal(0, 0.001),
                    linear_speed_std=0.05,
                    angular_speed_std=0.01
                )
                for idx in range(n_rand_targets)
            ]
        # when targets are specified
        else:
            targets = [
                Target(
                    id=target["id"],
                    init_x=target["x"], init_y=target["y"], heading=target["heading"],
                    linear_speed=target["linear_speed"],
                    angular_speed=target["angular_speed"],
                    linear_speed_std=target["linear_speed_std"],
                    angular_speed_std=target["angular_speed_std"]
                ) for target in list_of_target_dicts
            ]
        return targets

    def init_vehicle(self):
        return Vehicle(init_x=self.init_x,
                       init_y=self.init_y,
                       init_z=self.init_z,
                       init_psi=self.init_psi,
                       vehicle_l=self.vehicle_l,
                       hvel=self.hvel,
                       vvel=self.vvel)

    def init_sensor(self):
        return SensorModel(self.sensor_a,
                           self.sensor_b,
                           self.sensor_d,
                           self.sensor_g,
                           self.sensor_h,
                           self.sensor_width,
                           self.sensor_height,
                           self.sensor_focal_length)

    def get_ground_intersect(self, vehicle_pos, pitch, yaw):
        return self.sensor.project_camera_bounds_to_plane(vehicle_pos,
                                                self.sensor.rotated_camera_fov(
                                                    phi=pitch, psi=yaw))

    def get_sensor_measurements(self):
        '''
        Get sensor measurements from camera sensor
        '''
        camera_projection = self.get_ground_intersect(
            [self.vehicle.x, self.vehicle.y, self.vehicle.z], self.sensor_pitch,
            self.vehicle.psi)
        detected_targets = []
        for target in self.targets:
            if self.sensor.is_point_inside_camera_projection([target.x, target.y],
                                                             camera_projection):
                range_to_target = np.linalg.norm(
                    np.array([target.x, target.y, 0]) - np.array(
                        [self.vehicle.x, self.vehicle.y, self.vehicle.z]))
                # is_detected = self.sensor.get_detection(range_to_target)
                detection_prob = np.random.random()
                sensor_tpr = self.sensor.tpr(range_to_target)
                if detection_prob < sensor_tpr:
                    target.is_detected = True
                    detected_targets.append(target)
        return detected_targets, camera_projection

    def traverse(self):
        '''
        Waypoint manager and vehicle state update- moves vehicle towards waypoints as long as waypoints exist in global_waypt_list
        '''
        if not self.global_waypt_list or len(self.global_waypt_list.plan) == 0:
            return
        else:
            next_position = np.array(
                [self.global_waypt_list.plan[0].position.position.x,
                 self.global_waypt_list.plan[0].position.position.y,
                 self.global_waypt_list.plan[0].position.position.z])
            dist_to_waypt = np.linalg.norm(
                [self.vehicle.x, self.vehicle.y, self.vehicle.z] - next_position)

            # update waypoint list if reached waypoint
            if dist_to_waypt < self.waypt_threshold:
                print("Reached waypoint -> ", next_position)
                self.curr_waypt_num += 1
                self.global_waypt_list.plan.pop(0)

            # else keep trying to navigate to next waypoint
            else:
                omega, z_d = self.vehicle.go_to_goal(self.max_omega, self.max_zvel,
                                                     next_position, self.K_p,
                                                     self.K_p_z)
                self.vehicle.psi += self.del_t * omega
                self.vehicle.x += self.del_t * self.hvel * math.cos(self.vehicle.psi)
                self.vehicle.y += self.del_t * self.hvel * math.sin(self.vehicle.psi)
                self.vehicle.z += self.del_t * z_d

    def update_waypts(self, new_wpts):
        '''
        Receive new waypoints and send them to waypoint manager
        '''
        # self.global_waypt_list.append(new_wpts)
        self.global_waypt_list = new_wpts
        self.curr_waypt_num = 0

    def update_states(self):
        '''
        Updates the environment states
        '''
        self.traverse()

        # update the states for all ships in the environment
        for target in self.targets:
            target.propagate(self.del_t)

    def get_vehicle_uncertainty(self):
        return self.vehicle.position_uncertainty()

    # function that gets target heading and return heading with gaussian noise
    def get_target_heading_noise(self, heading):
        # gaussian noise model for now
        return heading + np.random.normal(0, 0.05)
