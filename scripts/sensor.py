import math
import random
import numpy as np

class SensorModel:
    '''
    Sensor frame is ENU
    '''
    def __init__(self, a, b, d, g, h, width, height, focal_length):
        self.a = a
        self.b = b
        self.d = d        
        self.g = g
        self.h = h

        self.width = width
        self.height = height
        self.focal_l = focal_length

    def tpr(self, range):
        '''
        double tpr_val = .5;
        double inflection_point = 200;
        double flatten_point = 600;
        if (range>=0 && range <=flatten_point)
            return 1/(1.1 + exp(0.01*(range-610)));
        else if (range>flatten_point)
            return 0.5;
        '''
        flatten_point = 600
        if range>=0 and range <=flatten_point:
            return 1/(1.1 + math.exp(0.01*(range-610)))
        else:
            return 0.5
    
    def fpr(self, range):
        '''
        double inflection_point = 200;
        double flatten_point = 600;
        if (range>=0 && range <=flatten_point)
            return 1- 1/(1.1 + exp(0.01*(range-610)));
        else if (range>flatten_point)
            return 0.5;
        '''
        flatten_point = 600
        if range>=0 and range <=flatten_point:
            return 1- 1/(1.1 + math.exp(0.01*(range-610)))
        else:
            return 0.5

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
        Rotating the FOV bounds of the camera with axis of camera as vehicle position
        '''

        # first find bounds of camera sensor and append focal length for scale in ENU 
        q1 = np.array([[self.width/2, self.height/2, -self.focal_l]])        
        q2 = np.array([[-self.width/2, self.height/2, -self.focal_l]])
        q3 = np.array([[-self.width/2, -self.height/2, -self.focal_l]])
        q4 = np.array([[self.width/2, -self.height/2, -self.focal_l]])
        q_body = [q1.T, q2.T, q3.T, q4.T]
        q_rotated = []

        # expensive and inefficient operation, need to speed up
        for q in q_body:
            q_rotated.append(np.matmul(self.Rz(psi), np.matmul(self.Ry(phi), np.matmul(self.Rx(theta), q))))

        return q_rotated
    
    def reqd_plane_intercept(self, vehicle_pos, q_rotated):
        # translating rotated camera bounds to world frame
        reqd_intercept = []
        for pt in q_rotated:
            translated_pt = np.array([vehicle_pos[0] + pt[0], vehicle_pos[1] + pt[1], vehicle_pos[2] + pt[2]])
            
            # finding intersection of line through vehicle_pos and translated_pt on the plane z=0
            '''
            parametric form of line through 2 points in 3D space:
            <x1 + (x1-x2)t, y1 + (y1-y2)t, z1 + (z1-z2)t> = <x,y,z>

            Substituting z-term in eqn of plane z=0,

            z1 + (z1-z2)t = 0
            so, t = -z1 / (z1-z2)

            hence, substituting t in the eqn of the line
            '''
            x = vehicle_pos[0] - (vehicle_pos[0] - translated_pt[0]) * (-vehicle_pos[2] / (vehicle_pos[2]-translated_pt[2]))
            y = vehicle_pos[1] - (vehicle_pos[1] - translated_pt[1]) * (-vehicle_pos[2] / (vehicle_pos[2]-translated_pt[2]))
            z = vehicle_pos[2] + (vehicle_pos[2] - translated_pt[2]) * (-vehicle_pos[2] / (vehicle_pos[2]-translated_pt[2]))
            reqd_intercept.append(np.array([x, y, z])) 
        return reqd_intercept
    
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
    