import math
import rospy
import random
import numpy as np

class SensorModel:
    '''
    Sensor frame is ENU
    '''
    def __init__(self, focal_length, width, height, pitch, max_range):
        self.focal_length = focal_length
        self.width = width
        self.height = height
        self.pitch = pitch
        self.max_range = max_range

    # for now, FPR is artificially the same as TPR
    def fpr(self, sensed_distance):
        return self.tpr(sensed_distance)

    def tnr(self, sensed_distance):
        return 1.0 - self.fpr(sensed_distance)


    def get_detection(self, range):
        return random.random() < self.tpr(range)
    
    def Rx(self, roll_angle):  # roll
        return np.array([[1, 0, 0],  
                        [0, math.cos(roll_angle), -math.sin(roll_angle)],
                        [0, math.sin(roll_angle), math.cos(roll_angle)]]) 
    
    def Ry(self, pitch_angle):  # pitch
        return  np.array([[math.cos(pitch_angle), 0, math.sin(pitch_angle)],
                        [0, 1, 0],
                        [-math.sin(pitch_angle), 0, math.cos(pitch_angle)]])

    def Rz(self, yaw_angle):  # yaw
        return np.array([[math.cos(yaw_angle), -math.sin(yaw_angle), 0],
                        [math.sin(yaw_angle), math.cos(yaw_angle), 0],
                        [0, 0, 1]])

    def rotated_camera_fov(self, roll=0, pitch=0, yaw=0):
        '''
        Rotating the FOV bounds of the camera with axis of camera as agent position
        '''

        # first find bounds of camera sensor and append focal length for scale in ENU 
        q1 = np.array([[self.focal_length, -self.width/2, -self.height/2]])        
        q2 = np.array([[self.focal_length, -self.width/2, self.height/2]])
        q3 = np.array([[self.focal_length, self.width/2, self.height/2]])
        q4 = np.array([[self.focal_length, self.width/2, -self.height/2]])
        q_body = [q1.T, q2.T, q3.T, q4.T]
        q_rotated = []

        # expensive and inefficient operation, need to speed up
        for q in q_body:
            q_rotated.append(np.matmul(self.Rz(yaw), np.matmul(self.Ry(pitch), np.matmul(self.Rx(roll), q))))

        return q_rotated
    
    def wrap_to_mpi_pi(self, angle_in):
        angle_out = angle_in
        while angle_out <= -np.pi:
            angle_out += 2*np.pi
        while angle_out >= np.pi:
            angle_out -= 2*np.pi
        return angle_out

    def wrap_to_0_2pi(self, angle_in):
        angle = angle_in
        while angle < 0:
            angle += 2*np.pi
        while angle >= 2*np.pi:
            angle -= 2*np.pi
        return angle
    def project_camera_bounds_to_plane(self, agent_pos, q_rotated):
        projected_camera_bounds = []
        for pt in q_rotated:
            '''
            parametric form of line through 2 points in 3D space:
            <x1 + (x1-x2)t, y1 + (y1-y2)t, z1 + (z1-z2)t> = <x,y,z>

            Substituting z-term in eqn of plane z=0,

            z1 + (z1-z2)t = 0
            so, t = -z1 / (z1-z2)

            hence, substituting t in the eqn of the line
            '''

            sensor_cutoff_distance = 1000 # TODO update with Junbin model and pass the cutoff distance in
            # rospy.loginfo("sensor cutoff distance: %f", self.max_range)
            length_q = np.sqrt(pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2])
            intercept_x = agent_pos[0] + (sensor_cutoff_distance / length_q) * pt[0]
            intercept_y = agent_pos[1] + (sensor_cutoff_distance / length_q) * pt[1]
            intercept_z = agent_pos[2] + (sensor_cutoff_distance / length_q) * pt[2]
            ray_portion = abs(agent_pos[2]) / (abs(agent_pos[2]) + abs(intercept_z))
            pos_portion = abs(intercept_z) / (abs(agent_pos[2]) + abs(intercept_z))
            x = ray_portion * intercept_x + pos_portion * agent_pos[0]
            y = ray_portion * intercept_y + pos_portion * agent_pos[1]
            z = ray_portion * intercept_z + pos_portion * agent_pos[2]
            
            projected_camera_bounds.append(np.array([x, y, z])) 
        return projected_camera_bounds
    
    # def project_camera_bounds_to_plane(self, agent_pos, q_rotated,sensor_cutoff_distance):
    #     projected_camera_bounds = []
    #     for pt in q_rotated:
    #         '''
    #         parametric form of line through 2 points in 3D space:
    #         <x1 + (x1-x2)t, y1 + (y1-y2)t, z1 + (z1-z2)t> = <x,y,z>

    #         Substituting z-term in eqn of plane z=0,

    #         z1 + (z1-z2)t = 0
    #         so, t = -z1 / (z1-z2)

    #         hence, substituting t in the eqn of the line
    #         '''

    #         # sensor_cutoff_distance = 1000 # TODO update with Junbin model and pass the cutoff distance in
    #         sensor_cutoff_distance = self.max_range
    #         reach_ground = False

    #         for pt in q_rotated:
    #             length_q = np.sqrt(pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2])
    #             intercept_x = agent_pos[0] + (sensor_cutoff_distance / length_q) * pt[0]
    #             intercept_y = agent_pos[1] + (sensor_cutoff_distance / length_q) * pt[1]
    #             intercept_z = agent_pos[2] + (sensor_cutoff_distance / length_q) * pt[2]


    #             if intercept_z <= 0:
    #                 reach_ground = True
                
    #             point_on_sphere = [intercept_x, intercept_y, intercept_z]
    #             projected_camera_bounds.append(np.array(point_on_sphere))
            
    #         if not reach_ground:
    #             rospy.loginfo("Camera range is so small that nothing on the sea level is considered to be detectable. Check the Z of the agent?")
    #             return projected_camera_bounds
            
    #         first_ray_end = projected_camera_bounds[0]
    #         last_ray_end = projected_camera_bounds[3]

    #         if (first_ray_end[2] * last_ray_end[2] < 0):

    #             theta0 = np.arccos(-agent_pos[2] / sensor_cutoff_distance)
    #             theta1 = np.arccos((last_ray_end[2] - agent_pos[2]) / sensor_cutoff_distance)
    #             theta2 = np.arccos((first_ray_end[2] - agent_pos[2]) / sensor_cutoff_distance)

    #             p1 = abs(theta2 - theta0) / abs(theta2 - theta1)
    #             p2 = abs(theta1 - theta0) / abs(theta2 - theta1)

    #             psi1 = np.arctan((last_ray_end[1] - agent_pos[1]) / (last_ray_end[0] - agent_pos[0]))
    #             psi2 = np.arctan((first_ray_end[1] - agent_pos[1]) / (first_ray_end[0] - agent_pos[0]))
    
    #             # Correction of atan, whose result is in [-pi/2, pi/2]
    #             if last_ray_end[0] - agent_pos[0] < 0:
    #                 psi1 += np.pi
    #             if first_ray_end[0] - agent_pos[0] < 0:
    #                 psi2 += np.pi

    #             # The interpolation of radian may be in 2 directions, the section we care is the 0~180 angle formed by the rays, not the 180~360 part
    #             # If the <180 angle covers the +x axis, wrap to [-pi, pi]
    #             if last_ray_end[0] + first_ray_end[0] >= 0:
    #                 psi1 = self.wrap_to_mpi_pi(psi1)
    #                 psi2 = self.wrap_to_mpi_pi(psi2)
    #             else:
    #                 psi1 = self.wrap_to_0_2pi(psi1)
    #                 psi2 = self.wrap_to_0_2pi(psi2)

    #             psi0 = p1 * psi1 + p2 * psi2
    #             intersect_vec_x = sensor_cutoff_distance * np.sin(theta0) * np.cos(psi0)
    #             intersect_vec_y = sensor_cutoff_distance * np.sin(theta0) * np.sin(psi0)
    #             intersect_vec_z = sensor_cutoff_distance * np.cos(theta0)

    #             z0_intercept = np.array([agent_pos[0] + intersect_vec_x, agent_pos[1] + intersect_vec_y, agent_pos[2] + intersect_vec_z])
    #             projected_camera_bounds.append(z0_intercept)

    #         # Check if the first ray intersects with the ground plane
    #         if first_ray_end[2] < 0:  # Top left corner of the image frame
    #             ray_portion = abs(agent_pos[2]) / (abs(agent_pos[2]) + abs(first_ray_end[2]))
    #             pos_portion = abs(first_ray_end[2]) / (abs(agent_pos[2]) + abs(first_ray_end[2]))
    #             intersect_x = ray_portion * first_ray_end[0] + pos_portion * agent_pos[0]
    #             intersect_y = ray_portion * first_ray_end[1] + pos_portion * agent_pos[1]
    #             intersect_z = ray_portion * first_ray_end[2] + pos_portion * agent_pos[2]
    #             z0_intercept = np.array([intersect_x, intersect_y, intersect_z])  # intersect_z should be 0
    #             projected_camera_bounds.append(z0_intercept)
            
    #     return projected_camera_bounds

        
    
    def substitute_pt_in_line(self, query_pt, pt1, pt2):
        '''
        Eqn of line through 2 given points:

        y2-y1 / x2-x1 = (y-y1) / (x-x1)

        Solving this given pt1 <x1,y1> , pt2 <x2,y2> and a query pt <x,y> if,
        result=0 then query_pt lies on line through pt1 and pt2
        result>0 then query_pt lies to the left of line
        result<0 then query_pt lies to the right of line
        '''
        return ((query_pt[1] - pt1[1]) * (pt2[0] - pt1[0]) - (query_pt[0] - pt1[0]) * (pt2[1] - pt1[1]))
    
    def is_point_inside_camera_projection(self, target_pos, camera_projection):
        # winding number algorithm
        '''
        For a convex polygon, if the sides of the polygon can be considered as a path 
        from the first vertex, then a query point is said to be inside the polygon if it
        lies on the same side of all the line segments making up the path

        :params camera_projection -> projected convex polygon of camera bounds on z=0
                target_pos -> ground truth target position
        '''
        num_polygon_edges = len(camera_projection)
        num_same_sides_l = 0
        num_same_sides_r = 0
        for i in range(num_polygon_edges):
            pt_in_line = self.substitute_pt_in_line(target_pos, camera_projection[i], 
                                                    camera_projection[(i+1)%num_polygon_edges])
            if pt_in_line == 0:
                return 0
            num_same_sides_l += pt_in_line > 0
            num_same_sides_r += pt_in_line < 0
        return abs(num_same_sides_r) == num_polygon_edges or num_same_sides_l == num_polygon_edges
    