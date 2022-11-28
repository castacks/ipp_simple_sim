import math
import random
import numpy as np

class SensorModel:
    '''
    Sensor frame is ENU
    '''
    def __init__(self, focal_length, width, height, pitch, max_range, endurance, hedge):
        self.focal_length = focal_length
        self.width = width
        self.height = height
        self.pitch = pitch
        self.max_range = max_range
        self.endurance = endurance
        self.hedge = hedge


    def fnr(self, sensed_distance):
        if (sensed_distance > self.max_range):
            return 0.5
        else:
            return (0.5 - self.hedge) * (sensed_distance / self.max_range) ** self.endurance + self.hedge

    def tpr(self ,sensed_distance):
        return 1.0 - self.fnr(sensed_distance)

    # for now, FPR is artificially the same as TPR
    def fpr(self, sensed_distance):
        return self.tpr(sensed_distance)

    def tnr(self, sensed_distance):
        return 1.0 - self.fpr(sensed_distance)


    def get_detection(self, range):
        return random.random() < self.tpr(range)
    
    def Rx(self, theta):  # roll
        return np.array([[1, 0, 0],  
                        [0, math.cos(theta), -math.sin(theta)],
                        [0, math.sin(theta), math.cos(theta)]]) 
    
    def Ry(self, phi):  # pitch
        return  np.array([[math.cos(phi), 0, math.sin(phi)],
                        [0, 1, 0],
                        [-math.sin(phi), 0, math.cos(phi)]])

    def Rz(self, psi):  # yaw
        return np.array([[math.cos(psi), -math.sin(psi), 0],
                        [math.sin(psi), math.cos(psi), 0],
                        [0, 0, 1]])

    def rotated_camera_fov(self, theta=0, psi=0, phi=0):
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
            q_rotated.append(np.matmul(self.Rz(psi), np.matmul(self.Ry(phi), np.matmul(self.Rx(theta), q))))

        return q_rotated
    
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
    