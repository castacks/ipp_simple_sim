import math
import numpy as np

class KalmanFilter:
    '''
    Kalman filter for the agent
    '''
    def __init__(self, initial_state, initial_estimate_error, process_noise, measurement_noise):
        self.state_estimate = initial_state
        self.estimate_error = initial_estimate_error
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, control_input):
        # Prediction step
        # Simulate agent dynamics in the prediction step
        self.state_estimate += control_input
        self.estimate_error += self.process_noise

    def update(self, measurement):
        # Update step
        kalman_gain = self.estimate_error / (self.estimate_error + self.measurement_noise)
        self.state_estimate += kalman_gain * (measurement - self.state_estimate)
        self.estimate_error *= (1 - kalman_gain)

    def get_state_estimate(self):
        return self.state_estimate
    

class Agent:
    '''
    Agent frame is ENU
    '''
    def __init__(self, agent_num,init_x, init_y, init_z, agent_l, hvel, vvel, init_roll=0, init_pitch=0, init_yaw=0):
        self.agent_num = agent_num
        self.x = init_x
        self.y = init_y
        self.z = init_z
        self.roll = init_roll  # roll angle
        self.pitch = init_pitch  # pitch angle
        self.yaw = init_yaw  # yaw angle
        self.vel = hvel
        self.vvel = vvel

        # Kalman filter parameters
        initial_state = np.array([init_x, init_y, init_z, init_roll, init_pitch, init_yaw])
        initial_estimate_error = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Initial position uncertainty
        process_noise = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])  # Process noise
        measurement_noise = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Measurement noise

        self.kalman_filter = KalmanFilter(initial_state, initial_estimate_error, process_noise, measurement_noise)

    def go_to_goal(self, max_omega, max_zvel, next_waypoint, K_p, K_p_z):
        '''
        Returns angular velocity and velocity in z-axis towards desired direction
        '''
        e = next_waypoint - [self.x, self.y, self.z]  # dist to desired position
        yaw_d = math.atan2(e[1], e[0])  # desired yaw
        z_error = e[2]
        
        omega = K_p*math.atan2(math.sin(yaw_d - self.yaw), math.cos(yaw_d - self.yaw))  # omega is desired heading
        z_d = K_p_z*z_error

        if omega > max_omega:
            omega = max_omega  # setting max angular vel
        if z_d > max_zvel:
            z_d = max_zvel

        # Update Kalman filter with measurement
        measurement = np.array([self.x, self.y, self.z]) + np.random.normal(0, self.kalman_filter.measurement_noise)
        self.kalman_filter.update(measurement)

        # Predict next state based on control input
        control_input = np.array([omega, self.vel, z_d])
        self.kalman_filter.predict(control_input)

        # Get the estimated state from Kalman filter
        estimated_state = self.kalman_filter.get_state_estimate()
        self.x, self.y, self.z = estimated_state

        return omega, z_d

    def position_uncertainty(self):
        '''
        Returns position uncertainty
        '''

        # hard-coding values for now
        sigma_x = 0.1
        sigma_y = 0.1
        sigma_z = 0.1
        sigma_roll = 0.1
        sigma_pitch = 0.1
        sigma_yaw = 0.1
        naive_estimate = [sigma_x, sigma_y, sigma_z, sigma_roll, sigma_pitch, sigma_yaw]
        return self.kalman_filter.estimate_error.tolist()