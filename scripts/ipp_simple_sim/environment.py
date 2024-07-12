import math
import random
import rospy
from typing import List, Any
import numpy as np
from ipp_simple_sim.agent import *
from ipp_simple_sim.target import *
from ipp_simple_sim.sensor import *
from planner_map_interfaces.msg import Plan
from ipp_planners.srv import SearchMapUpdate

class Environment:
    """Class to represent the simulation environment.
    
    """
    def __init__(
        self, 
        max_omega: float=5, 
        max_zvel: float=5,
        init_x: float=None, 
        init_y: float=None, 
        init_z: float=None, 
        init_yaw: float=None,
        K_p: float=0.01, 
        K_p_z: float=0.01,
        num_agents: int=1,
        agent_l: float=3, 
        hvel: float=5, 
        vvel: float=2, 
        n_rand_targets: int=-1, 
        del_t: float=0.02,
        waypoint_threshold: float=5,
        sensor_focal_length: float=5, 
        sensor_width: float=10, 
        sensor_height: float=10,
        sensor_pitch: float=20, 
        sensor_max_range: float=500
    ):
        """Initialize the environment with the given parameters.

        Args:
            max_omega (float, optional): The maximum angular velocity of the agents. Defaults to 5.
            max_zvel (float, optional): The maximum vertical velocity of the agents. Defaults to 5.
            init_x (float, optional): The initial x-coordinate of the agents. Defaults to None.
            init_y (float, optional): The initial y-coordinate of the agents. Defaults to None.
            init_z (float, optional): The initial z-coordinate of the agents. Defaults to None.
            init_yaw (float, optional): The initial yaw of the agents. Defaults to None.
            K_p (float, optional): The x-y proportionality constant for the PID controller. Defaults to 0.01.
            K_p_z (float, optional): The z-axis proportionality constant for the PID controller. Defaults to 0.01.
            num_agents (int, optional): The number of agents to simulate. Defaults to 1.
            agent_l (float, optional): The length of the agents. Defaults to 3.
            hvel (float, optional): The horizontal velocity of the agents. Defaults to 5.
            vvel (float, optional): The vertical velocity of the agents. Defaults to 2.
            n_rand_targets (int, optional): The number of random targets to generate. Defaults to -1.
            del_t (float, optional): The time step to propagate the environment by. Defaults to 0.02.
            waypoint_threshold (float, optional): The threshold distance to a waypoint before moving to the next one. Defaults to 5.
            sensor_focal_length (float, optional): The focal length of the camera sensor. Defaults to 5.
            sensor_width (float, optional): The width of the camera sensor. Defaults to 10.
            sensor_height (float, optional): The height of the camera sensor. Defaults to 10.
            sensor_pitch (float, optional): The pitch of the camera sensor. Defaults to 20.
            sensor_max_range (float, optional): The maximum range of the camera sensor. Defaults to 500.
        
        """
        # if initial position not specified, randomly spawn agent between (50, 1000)
        init_x = random.randrange(50, 1000) if init_x is None else init_x
        init_y = random.randrange(50, 1000) if init_y is None else init_y
        init_z = random.randrange(20, 120,
                                  20) if init_z is None else init_z  # discretized by step-size 20
        init_yaw = random.uniform(0, np.pi) if init_yaw is None else init_yaw

        # drone pose
        self.init_x = init_x
        self.init_y = init_y
        self.init_z = init_z
        self.init_yaw = init_yaw

        self.del_t = del_t
        self.num_agents = num_agents #number of agents to simulate

        self.agent_l = agent_l  # agent length
        self.hvel = hvel  # horizontal velocity
        self.vvel = vvel  # vertical velocity
        self.max_omega = max_omega  # max angular velocity
        self.max_zvel = max_zvel  # max vertical velocity

        self.sensor_focal_length = sensor_focal_length
        self.sensor_width = sensor_width
        self.sensor_height = sensor_height
        self.sensor_pitch = sensor_pitch
        self.sensor_max_range = sensor_max_range

        # if targets not specified, randomly generate between 1-10 targets
        self.targets = []

        self.waypoint_threshold = waypoint_threshold

        self.K_p = K_p  # x-y proportionality constant for PID controller
        self.K_p_z = K_p_z  # z-axis proportionality constant for PID controller
        self.sensor = self.init_sensor()

        self.global_waypoint_list = [Plan() for i in range(self.num_agents)]
        self.agent = [self.init_agent(i) for i in range(self.num_agents)]
        self.prev_agentxyz = [[0.0, 0.0, 0.0] for i in range(self.num_agents)]
        self.curr_waypoint_num = [0 for i in range(self.num_agents)]
        self.remaining_budget = [0 for i in range(self.num_agents)]
        self.n_rand_targets = n_rand_targets

        # self.prev_time = -1

    def generate_targets(
            self, 

            target_priors: List[Any], 
            set_true_targets_to_target_prior_means,
            sample_true_targets_from_target_priors,
            sample_additional_true_targets_from_search_prior,
            search_map_params: SearchMapUpdate,
            heading: tuple,
            linear_speed: tuple,
            angular_speed: tuple,
            linear_speed_std: float,
            angular_speed_std: float,
            decay_rate: float,
    ) -> None:
        """Generates targets in the environment based on the given target priors.

        Args:
            target_priors (List[Any]): The target priors to generate targets from.
            set_true_targets_to_target_prior_means (bool): Whether to set the true simulated target states at the means of the target prior distributions.
            sample_true_targets_from_target_priors (bool): Whether to sample a number of true simulated target states from the target prior distributions.
            sample_additional_true_targets_from_search_prior (bool): Whether to sample additional true simulated target states from the search map belief.
            search_map_params (SearchMapUpdate): The search map parameters to sample the targets from.
            heading (tuple): The range of headings to sample from.
            linear_speed (tuple): The range of linear speeds to sample from.
            angular_speed (tuple): The range of angular speeds to sample from.
            linear_speed_std (float): The standard deviation of the linear speed noise.
            angular_speed_std (float): The standard deviation of the angular speed noise.
            decay_rate (float): The rate at which the target's speed decays. Pass in 0 for no decay.
        """

        if set_true_targets_to_target_prior_means:
            rospy.loginfo(
                "Set true simulated target states from plan request prior distributions"
            )
            self.targets.extend(self.init_true_targets_from_target_prior_means(target_priors, decay_rate))  
            
        elif sample_true_targets_from_target_priors:
            rospy.loginfo(
                "Sampling true simulated target states from plan request prior distributions"
            )
            self.targets.extend(self.sample_true_targets_from_target_priors(target_priors, decay_rate))

        if sample_additional_true_targets_from_search_prior:
            rospy.loginfo("Sampling true simulated target states from search map prior")
            self.targets.extend(self.sample_additional_true_targets_from_search_prior(
                search_map_params,
                heading,
                linear_speed,
                angular_speed,
                linear_speed_std,
                angular_speed_std,
                decay_rate,
                ))
        
    def init_true_targets_from_target_prior_means(self, target_priors: List[Any], decay_rate: float) -> List[Target]:
        """
        Given the list of target priors, initialize the true target. 
        Targets are initialized deterministically from the mean of the target priors.

        Args:
            target_priors (List[Any]): The target priors to generate targets from.
            decay_rate (float): The rate at which the target's speed decays. Pass in 0 for no decay.
        
        Returns:
            List[Target]: The initialized list of true targets.
        """
        sampled_targets = []
        for prior in target_priors:
            t = prior.target
            if t and not (t.x == 0 and t.y == 0 and t.xdot == 0 and t.ydot == 0):
                prior_heading = np.arctan2(t.ydot, t.xdot)
                prior_speed = np.sqrt(t.xdot**2 + t.ydot**2)
                sim_target = Target(
                    id=t.local_id,
                    init_x=t.x,
                    init_y=t.y,
                    heading=prior_heading,
                    linear_speed=prior_speed,
                    angular_speed=0,
                    linear_speed_std=0.0,
                    angular_speed_std=0.0,
                    decay_rate=decay_rate,
                )
                sampled_targets.append(sim_target)
        return sampled_targets     

    def sample_true_targets_from_target_priors(self, target_priors: List[Any], decay_rate: float) -> List[Target]:
        """
        Sample the true target states from a multivariate normal distribution of the target priors.

        Args:
            target_priors (List[Any]): The target priors to generate targets from.
            decay_rate (float): The rate at which the target's speed decays. Pass in 0 for no decay.
        
        Returns:
            List[Target]: The initialized list of true targets.
        """
        sampled_targets = []
        for prior in target_priors:
            t = prior.target
            if t and not (t.x == 0 and t.y == 0 and t.xdot == 0 and t.ydot == 0):
                for _ in range(self.n_rand_targets):
                    prior_speed = np.sqrt(t.xdot**2 + t.ydot**2)
                    jacobian = np.array(
                        [
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, -t.ydot / (t.xdot**2 + t.ydot**2), t.xdot / (t.xdot**2 + t.ydot**2)],
                            [0, 0, t.xdot / prior_speed, t.ydot / prior_speed],
                            [0, 0, 0, 0],
                        ]
                    )
                    covs = np.array(t.covariance).reshape(5, 5)[0:4, 0:4]
                    covs = jacobian @ covs @ jacobian.T
                    target_state = np.random.multivariate_normal([0, 0, 0, 0, 0], covs, 1)[
                        0
                    ]
                    # target_state *= 3/4  # scale down the variance
                    sim_target = Target(
                        id=t.id,
                        init_x=target_state[0] + t.x + np.random.normal(0, 0.5),
                        init_y=target_state[1] + t.y + np.random.normal(0, 0.5),
                        heading=target_state[2] + t.heading + np.random.normal(0, 0.05),
                        linear_speed=target_state[3] + t.linear_speed + np.random.normal(0, 0.5),
                        angular_speed=target_state[4] + t.angular_speed + np.random.normal(0, 0.05),
                        linear_speed_std=0.00,
                        angular_speed_std=0.000,
                        decay_rate=decay_rate,
                    )
                    sampled_targets.append(sim_target)
        return sampled_targets
    
    def sample_additional_true_targets_from_search_prior(
            self, 
            search_map_params: SearchMapUpdate,
            heading: tuple,
            linear: tuple,
            angular_speed: tuple,
            linear_speed_std: float,
            angular_speed_std: float,
            decay_rate: float,
    ) -> List[Target]:
        """Sample additional true targets from the search map prior.

        Args:
            search_map_params: The search map parameters to sample the targets from.
            heading (tuple): The range of headings to sample from.
            linear (tuple): The range of linear speeds to sample from.
            angular_speed (tuple): The range of angular speeds to sample from.
            linear_speed_std (float): The standard deviation of the linear speed noise.
            angular_speed_std (float): The standard deviation of the angular speed noise.
            decay_rate (float): The rate at which the target's speed decays. Pass in 0 for no decay.
        
        Returns:
            List[Target]: The initialized list of true targets
        """

        sampled_targets = []
        
        # SearchMap.h stores the map as rows (first axis) as x, cols (second axis) as y
        n_rows = int(
            (search_map_params.x_end - search_map_params.x_start)
            / search_map_params.map_resolution
        )
        n_cols = int(
            (search_map_params.y_end - search_map_params.y_start)
            / search_map_params.map_resolution
        )

        search_map = np.array(search_map_params.map_values).reshape(n_rows, n_cols)

        linear_idxs = np.random.choice(
            search_map.size,
            size=self.n_rand_targets,
            p=search_map.ravel() / float(search_map.sum()),
        )
        rows, cols = np.unravel_index(linear_idxs, search_map.shape)

        id_to_start_from = (
            max(target.id for target in self.targets) + 1
            if len(self.targets) > 0
            else 0
        )

        for idx, (r, c) in enumerate(zip(rows, cols)):
            x = search_map_params.x_start + r * search_map_params.map_resolution
            y = search_map_params.y_start + c * search_map_params.map_resolution
            target = Target(
                id=idx + id_to_start_from,
                init_x=x,
                init_y=y,
                heading=np.random.uniform(*heading),
                linear_speed=np.random.uniform(*linear),
                angular_speed=np.random.normal(*angular_speed),
                linear_speed_std=linear_speed_std,
                angular_speed_std=angular_speed_std,
                decay_rate=decay_rate,
            )
            
            rospy.loginfo(f"Sampled target from search prior: {target}")
            sampled_targets.append(target)

        return sampled_targets
        

    def init_agent(self,id_num):
        return Agent(agent_num=id_num,
                        init_x=self.init_x,
                        init_y=self.init_y,
                        init_z=self.init_z,
                        init_yaw=self.init_yaw,
                        agent_l=self.agent_l,
                        hvel=self.hvel,
                        vvel=self.vvel)

    def init_sensor(self):
        return SensorModel(
            self.sensor_focal_length,
            self.sensor_width, self.sensor_height, self.sensor_pitch, self.sensor_max_range
        )

    def get_ground_intersect(self, agent_pos, pitch, yaw):
        return self.sensor.project_camera_bounds_to_plane(agent_pos,
                                                self.sensor.rotated_camera_fov(
                                                    pitch=pitch, yaw=yaw))

    def get_sensor_measurements(self):
        '''
        Get sensor measurements from camera sensor
        '''
        detected_targets_list = []
        camera_projections_list = []
        for i in range(self.num_agents):
            camera_projection = self.get_ground_intersect(
            [self.agent[i].x, self.agent[i].y, self.agent[i].z], self.sensor_pitch,
            self.agent[i].yaw)
            detected_targets = []
            for target in self.targets:
                if self.sensor.is_point_inside_camera_projection([target.x, target.y],
                                                             camera_projection):
                    range_to_target = np.linalg.norm(
                    np.array([target.x, target.y, 0]) - np.array(
                        [self.agent[i].x, self.agent[i].y, self.agent[i].z]))
                    # is_detected = self.sensor.get_detection(range_to_target)
                    detection_prob = np.random.random()
                    sensor_tpr = self.sensor.tpr(range_to_target)
                    if detection_prob < sensor_tpr:
                        target.is_detected = True
                        detected_targets.append(target)
            detected_targets_list.append(detected_targets)
            camera_projections_list.append(camera_projection)
        return detected_targets_list, camera_projections_list

    def traverse(self, delta_t):
        '''
        Waypoint manager and agent state update- moves agent towards waypoints as long as waypoints exist in global_waypoint_list
        '''
        for i in range(self.num_agents):
            agent_waypoint_list = self.global_waypoint_list[i]
            if len(agent_waypoint_list.plan) == 0:
                continue
            else:
                next_position = np.array(
                    [agent_waypoint_list.plan[0].position.position.x,
                    agent_waypoint_list.plan[0].position.position.y,
                    agent_waypoint_list.plan[0].position.position.z])
                dist_to_waypoint = np.linalg.norm(
                    [self.agent[i].x, self.agent[i].y, self.agent[i].z] - next_position)
                # update waypoint list if reached waypoint
                if dist_to_waypoint < self.waypoint_threshold:
                    self.curr_waypoint_num[i] += 1
                    agent_waypoint_list.plan.pop(0)
                omega, z_d = self.agent[i].go_to_goal(self.max_omega, self.max_zvel,
                                                    next_position, self.K_p,
                                                    self.K_p_z)
                self.agent[i].yaw += delta_t * omega
                self.agent[i].x += delta_t * self.hvel * math.cos(self.agent[i].yaw)
                self.agent[i].y += delta_t * self.hvel * math.sin(self.agent[i].yaw)
                self.agent[i].z += delta_t * z_d
                delta_dist = np.linalg.norm(
                        np.array([self.agent[i].x, self.agent[i].y, self.agent[i].z]) - np.array(self.prev_agentxyz[i]))
                if delta_dist > self.agent[i].vel*2:
                    print("Jump for agent ", i, " of ", delta_dist, "m")
                    print("Ignoring jump and not changing remaining budget")
                else:
                    self.remaining_budget[i] -= delta_dist
                self.prev_agentxyz[i] = [self.agent[i].x, self.agent[i].y, self.agent[i].z]
        

    def update_waypoints(self, new_wpts, id_num):
        '''
        Receive new waypoints and send them to waypoint manager
        '''
        # self.global_waypoint_list.append(new_wpts)
        self.global_waypoint_list[id_num] = new_wpts
        self.curr_waypoint_num[id_num] = 0

    def update_states(self):
        '''
        Updates the environment states
        '''
        self.traverse(self.del_t)
        # update the states for all ships in the environment
        for target in self.targets:
            target.propagate(self.del_t)

    def get_agent_uncertainty(self,id_num):
        return self.agent[id_num].position_uncertainty()

    # function that gets target heading and return heading with gaussian noise
    def get_target_heading_noise(self, heading):
        # gaussian noise model for now
        return heading + np.random.normal(0, 0.05)
