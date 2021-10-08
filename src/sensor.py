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
        return 1 / (self.a + self.b*(math.pow(math.exp, self.d*(range-self.g)))) - self.h

    def get_detection(self, range)    :
        return random.random() < self.tpr(range)
    
    def Rx(self, theta):  # roll
        return np.matrix([1, 0, 0],  
                        [0, math.cos(theta), -math.sin(theta)],
                        [0, math.sin(theta), math.cos(theta)]) 
    
    def Ry(self, phi):  # pitch
        return  np.matrix([math.cos(phi), 0, math.sin(phi)],
                        [0, 1, 0],
                        [-math.sin(phi), 0, math.cos(phi)])
    
    def Rz(self, psi):  # yaw
        return np.matrix([math.cos(psi), -math.sin(psi), 0],
                        [math.sin(psi), math.cos(psi), 0],
                        [0, 0, 1])
    
    def rotated_camera_fov(self, theta=0, phi=0, psi=0):
        '''
        Rotation from camera to world frame
        '''

        # first find bounds of camera sensor and append focal length for scale
        q1 = np.array([self.width/2, self.height/2, self.focal_l])        
        q2 = np.array([self.width/2, -self.height/2, self.focal_l])
        q3 = np.array([-self.width/2, -self.height/2, self.focal_l])
        q4 = np.array([-self.width/2, self.height/2, self.focal_l])

        q_body = [q1.transpose, q2.transpose, q3.transpose, q4.transpose]
        q_rotated = []

        # expensive and inefficient operation, need to speed up
        for q in q_body:
            q_rotated.append(np.matmul(self.Rz(psi), np.matmul(self.Ry(phi), np.matmul(self.Rx(theta), q))))

        return q_rotated
    
    def fov_world_frame_vertices(self, position, theta, q_rotated):
        p1_4 = position / (math.sin(math.sin(theta)) * self.height/2 + math.cos(theta) * self.focal_l)
        p2_3 = position / (-math.sin(math.sin(theta)) * self.height/2 + math.cos(theta) * self.focal_l)

        q_world = [position + p1_4*q_rotated[0],
                    position + p2_3*q_rotated[1],
                    position + p2_3*q_rotated[2],
                    position + p1_4*q_rotated[3]]
        return q_world
