#!/usr/bin/env python2

import os
import rospy
import numpy as np
from rospkg import RosPack
from planner_map_interfaces.msg import Plan
from environment import *
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt8
from simple_ipp_sim.msg import TargetPoses, TargetPose, Detections
from tf.transformations import quaternion_from_euler

from visualization_msgs.msg import Marker, MarkerArray

package = RosPack()
package_path = package.get_path("simple_ipp_sim")

# https://sashamaps.net/docs/resources/20-colors/
COLORS = [[230, 25, 75],   [60, 180, 75],   [255, 225, 25], [0, 130, 200],
               [245, 130, 48],  [145, 30, 180],  [70, 240, 240], [240, 50, 230],
               [210, 245, 60],  [250, 190, 212], [0, 128, 128],  [220, 190, 255],
               [170, 110, 40],  [255, 250, 200], [128, 0, 0],    [170, 255, 195],
               [128, 128, 0],   [255, 215, 180], [0, 0, 128],    [128, 128, 128],
               [255, 255, 255], [0, 0, 0]]
def get_color(index):
    r, g, b = COLORS[index % len(COLORS)]
    ros_color = ColorRGBA()
    ros_color.r = r / 255.0 
    ros_color.g = g / 255.0 
    ros_color.b = b / 255.0 
    ros_color.a = 1.0
    return  ros_color

class SimManager:
    def __init__(self):

        self.planner_path_topic = rospy.get_param("~planner_path")
        self.sim_env = self.env_setup()
        self.agent_traj_list = []

    def env_setup(self):
        # ships
        targets_list = rospy.get_param("/env_setup/targets", [])

        # drone state
        init_x = rospy.get_param("/env_setup/init_x")
        init_y = rospy.get_param("/env_setup/init_y")
        init_z = rospy.get_param("/env_setup/init_z")
        init_psi = rospy.get_param("/env_setup/init_psi")

        max_omega = rospy.get_param("/env_setup/max_omega")
        max_zvel = rospy.get_param("/env_setup/max_zvel")

        agent_l = rospy.get_param("/env_setup/agent_l")

        hvel = rospy.get_param("/env_setup/hvel")
        vvel = rospy.get_param("/env_setup/vvel")

        n_rand_targets = rospy.get_param("/env_setup/n_rand_targets")

        del_t = rospy.get_param("/env_setup/del_t")

        K_p = rospy.get_param("/env_setup/K_p")
        K_p_z = rospy.get_param("/env_setup/K_p_z")

        waypt_threshold = rospy.get_param("/env_setup/waypt_threshold")

        sensor_focal_length = rospy.get_param("/env_setup/sensor_focal_length")
        sensor_width = rospy.get_param("/env_setup/sensor_width")
        sensor_height = rospy.get_param("/env_setup/sensor_height")
        sensor_a = rospy.get_param("/env_setup/sensor_a")
        sensor_b = rospy.get_param("/env_setup/sensor_b")
        sensor_d = rospy.get_param("/env_setup/sensor_d")
        sensor_g = rospy.get_param("/env_setup/sensor_g")
        sensor_h = rospy.get_param("/env_setup/sensor_h")
        sensor_pitch = rospy.get_param("/env_setup/sensor_pitch")

        return Environment(targets_list, 
                            max_omega, 
                            max_zvel,
                            init_x, 
                            init_y,
                            init_z,
                            init_psi,
                            K_p,
                            K_p_z,
                            agent_l,
                            hvel,
                            vvel,
                            n_rand_targets,
                            del_t,
                            waypt_threshold,
                            sensor_focal_length,
                            sensor_width,
                            sensor_height,
                            sensor_a,
                            sensor_b,
                            sensor_d,
                            sensor_g,
                            sensor_h,
                            sensor_pitch)
    
    def get_agent_position(self, time, frame):
        agent_pose = PoseStamped()
        agent_pose.header.frame_id = frame
        agent_pose.header.stamp = time
        # print self.sim_env.agent.x
        agent_pose.pose.position.x = self.sim_env.agent.x
        agent_pose.pose.position.y = self.sim_env.agent.y
        agent_pose.pose.position.z = self.sim_env.agent.z

        quat = quaternion_from_euler(0, 0, self.sim_env.agent.psi)
        agent_pose.pose.orientation.x = quat[0]
        agent_pose.pose.orientation.y = quat[1]
        agent_pose.pose.orientation.z = quat[2]
        agent_pose.pose.orientation.w = quat[3]
        return agent_pose
    
    def get_target_positions(self, time, frame):
        targets_pose = TargetPoses()
        targets_pose.header.frame_id = frame
        targets_pose.header.stamp = time

        for target in self.sim_env.targets:
            target_pose = TargetPose()

            target_pose.id = target.id
            target_pose.x = target.x
            target_pose.y = target.y
            target_pose.heading = target.heading
            target_pose.linear_speed = target.linear_speed
            target_pose.angular_speed = target.angular_speed
            target_pose.is_detected = target.is_detected

            # print target_pose
            targets_pose.targets.append(target_pose)
        
        return targets_pose
    
    def get_camera_pose(self, time, frame):
        camera_pose = Odometry()
        camera_pose.header.frame_id = frame
        camera_pose.header.stamp = time
        camera_pose.pose.pose.position.x = self.sim_env.agent.x
        camera_pose.pose.pose.position.y = self.sim_env.agent.y
        camera_pose.pose.pose.position.z = self.sim_env.agent.z
        
        quat = quaternion_from_euler(0, self.sim_env.sensor_pitch, self.sim_env.agent.psi)
        camera_pose.pose.pose.orientation.x = quat[0]
        camera_pose.pose.pose.orientation.y = quat[1]
        camera_pose.pose.pose.orientation.z = quat[2]
        camera_pose.pose.pose.orientation.w = quat[3]

        cov_matrix = np.zeros((6,6))
        cov_matrix[0,0] = self.sim_env.get_agent_uncertainty()[0]
        cov_matrix[1,1] = self.sim_env.get_agent_uncertainty()[1]
        cov_matrix[2,2] = self.sim_env.get_agent_uncertainty()[2]
        cov_matrix[3,3] = self.sim_env.get_agent_uncertainty()[3]
        cov_matrix[4,4] = self.sim_env.get_agent_uncertainty()[4]
        cov_matrix[5,5] = self.sim_env.get_agent_uncertainty()[5]

        camera_pose.pose.covariance = cov_matrix.flatten().tolist()

        return camera_pose
    
    def get_waypt_num(self):
        waypt_number = UInt8()
        waypt_number.data = self.sim_env.curr_waypt_num
        return waypt_number
    
    def get_target_detections(self, time, frame):
        detection_msg = Detections()
        detection_msg.header.frame_id = frame
        detection_msg.header.stamp = time

        detected_targets, camera_projection = self.sim_env.get_sensor_measurements()

        for target in detected_targets:
            detection_msg.headings.append(self.sim_env.get_target_heading_noise(target.heading))

            target_camera_unit_vector = Point()

            range_to_target = np.linalg.norm(np.array([target.x, target.y, 0]) - np.array([self.sim_env.agent.x, 
                                                                            self.sim_env.agent.y, self.sim_env.agent.z]))
            
            i_hat = (target.x - self.sim_env.agent.x) / range_to_target
            j_hat = (target.y - self.sim_env.agent.y) / range_to_target
            k_hat = - self.sim_env.agent.z / range_to_target

            # R = np.matmul(self.sim_env.sensor.Rz(self.sim_env.agent.phi), self.sim_env.sensor.Ry(self.sim_env.sensor_pitch))
            # print("current pitch is", self.sim_env.sensor_pitch)
            # print("current psi is", self.sim_env.agent.psi)
            R = np.matmul(self.sim_env.sensor.Rz(self.sim_env.agent.psi),self.sim_env.sensor.Ry(self.sim_env.sensor_pitch))
            R_inv = np.linalg.inv(R)
            camera_frame_pose = np.matmul(R_inv, [i_hat, j_hat, k_hat])

            target_camera_unit_vector.x = camera_frame_pose[0]
            target_camera_unit_vector.y = camera_frame_pose[1]
            target_camera_unit_vector.z = camera_frame_pose[2]
            detection_msg.target_camera_vectors.append(target_camera_unit_vector)

            detection_msg.target_ids.append(target.id)

        return detection_msg, camera_projection
    
    def planner_callback(self, msg):
        self.sim_env.update_waypts(msg)

    def get_ocean_marker(self, time, frame):
        ocean_marker = Marker()
        ocean_marker.header.frame_id = frame
        ocean_marker.ns = "ocean"
        ocean_marker.header.stamp = time
        ocean_marker.id = 0
        ocean_marker.type = Marker.CUBE
        ocean_marker.action = Marker.ADD
        ocean_marker.lifetime = rospy.Duration()
        ocean_marker.color.r = 0
        ocean_marker.color.b = 1.0
        ocean_marker.color.g = 0.8
        ocean_marker.color.a = 1
        ocean_marker.scale.x = 10000
        ocean_marker.scale.y = 10000
        ocean_marker.scale.z = 1
        ocean_marker.pose.position.z = -1
        return ocean_marker



    def get_agent_marker(self, time, frame, agent_pose):
        agent_marker = Marker()
        agent_marker.header.frame_id = frame
        agent_marker.header.stamp = time
        agent_marker.ns = "agent_pose"
        agent_marker.id = 0
        agent_marker.type = Marker.ARROW
        agent_marker.action = Marker.ADD
        agent_marker.lifetime = rospy.Duration()
        # agent_marker.pose = Pose(Point(0, 0, 100), Quaternion(0, 0, 0, 1))
        agent_marker.pose.position = agent_pose.pose.position
        # print (agent_pose.pose.orientation)
        agent_marker.pose.orientation = agent_pose.pose.orientation
        agent_marker.color.r = 0
        agent_marker.color.g = 1
        agent_marker.color.b = 0
        agent_marker.color.a = 1
        agent_marker.scale.x = 1
        agent_marker.scale.y = 1
        agent_marker.scale.z = 1

        return agent_marker

    def get_agent_trajectory_marker(self, time, frame, agent_pose):
        trajectory_marker = Marker()
        trajectory_marker.header.frame_id = frame
        trajectory_marker.header.stamp = time
        trajectory_marker.ns = "agent_trajectory"
        trajectory_marker.id = 0
        trajectory_marker.type = Marker.LINE_STRIP
        trajectory_marker.action = Marker.ADD
        trajectory_marker.lifetime = rospy.Duration()

        self.agent_traj_list.append([agent_pose.pose.position.x, agent_pose.pose.position.y, agent_pose.pose.position.z])
        if len(self.agent_traj_list) > 1000:  # setting traj length to 100
            self.agent_traj_list.pop(0)

        trajectory_marker.pose.position.x = 0
        trajectory_marker.pose.position.y = 0
        trajectory_marker.pose.position.z = 0
        
        for i in range(1, len(self.agent_traj_list)):
            trajectory_marker.points.append(Point(self.agent_traj_list[i][0], 
                                self.agent_traj_list[i][1], self.agent_traj_list[i][2]))
        
        trajectory_marker.color.r = 1
        trajectory_marker.color.g = 69/255
        trajectory_marker.color.b = 0
        trajectory_marker.color.a = 1
        trajectory_marker.scale.x = 1
        trajectory_marker.scale.y = 1
        trajectory_marker.scale.z = 1

        return trajectory_marker

    """
    Four points of the sensor footprint polygon. 
    """
    def get_projection_points_marker(self, time, frame, agent_pose, camera_projection):
        projection_points_marker = Marker()
        projection_points_marker.header.frame_id = frame
        projection_points_marker.header.stamp = time
        projection_points_marker.ns = "projection_marker"
        projection_points_marker.id = 0
        projection_points_marker.type = Marker.POINTS
        projection_points_marker.action = Marker.ADD
        projection_points_marker.color.r = 1
        projection_points_marker.color.g = 1
        projection_points_marker.color.b = 0
        projection_points_marker.color.a = 1
        projection_points_marker.scale.x = 1
        projection_points_marker.scale.y = 1
        projection_points_marker.scale.z = 1

        points = []
        for np_point in camera_projection:
            ros_point = Point()
            ros_point.x = np_point[0]
            ros_point.y = np_point[1]
            ros_point.z = np_point[2]
            points.append(ros_point)
        
        projection_points_marker.points = points
        return projection_points_marker

    def get_projection_marker(self, time, frame, agent_pose, camera_projection):
        projection_marker = Marker()
        projection_marker.header.frame_id = frame
        projection_marker.header.stamp = time
        projection_marker.ns = "projection_marker"
        projection_marker.id = 0
        projection_marker.type = Marker.LINE_STRIP
        projection_marker.action = Marker.ADD
        # projection_marker.lifetime = rospy.Duration()
        projection_marker.color.r = 1
        projection_marker.color.g = 69/255
        projection_marker.color.b = 0
        projection_marker.color.a = 1
        projection_marker.scale.x = 1
        projection_marker.scale.y = 1
        projection_marker.scale.z = 1

        points = []

        agent_point = Point()
        agent_point.x = agent_pose.pose.position.x
        agent_point.y = agent_pose.pose.position.y
        agent_point.z = agent_pose.pose.position.z

        # connect the projected camera bounds
        for edge in range(len(camera_projection)):
            point_a = Point()
            point_a.x = camera_projection[edge][0]
            point_a.y = camera_projection[edge][1]
            point_a.z = camera_projection[edge][2]

            point_b = Point()
            point_b.x = camera_projection[(edge + 1) % len(camera_projection)][0]
            point_b.y = camera_projection[(edge + 1) % len(camera_projection)][1]
            point_b.z = camera_projection[(edge + 1) % len(camera_projection)][2]

            points.append(point_b)
            points.append(point_a)
            points.append(agent_point)
            points.append(point_b)
            points.append(point_a)

        projection_marker.points = points

        return projection_marker
    
    def get_targets_marker(self, time, frame, target_positions):
        targets_marker_array = MarkerArray()

        for idx, target in enumerate(target_positions.targets):
            target_marker = Marker()
            target_marker.header.frame_id = frame
            target_marker.header.stamp = time
            target_marker.ns = "target_pose"
            target_marker.id = idx
            target_marker.type = Marker.MESH_RESOURCE
            target_marker.action = Marker.ADD
            target_marker.mesh_use_embedded_materials = True
            target_marker.mesh_resource = "package://simple_ipp_sim/meshes/boat.dae"
            target_marker.lifetime = rospy.Duration()
            quat = quaternion_from_euler(0, 0, target.heading)
            target_marker.pose = Pose(Point(target.x, 
                                            target.y,
                                            0),  # z offset to make it appear above grid-map
                                            Quaternion(quat[0], quat[1], quat[2], quat[3]))

            target_marker.color = get_color(target.id)
            target_marker.scale.x = 1
            target_marker.scale.y = 1
            target_marker.scale.z = 1
            targets_marker_array.markers.append(target_marker)
        
        return targets_marker_array

    def main(self):
        waypt_num_pub = rospy.Publisher('/ship_simulator/waypoint_num', UInt8, queue_size=10)
        agent_pose_pub = rospy.Publisher('/ship_simulator/agent_pose', PoseStamped, queue_size=10)
        target_pose_pub = rospy.Publisher('/ship_simulator/target_poses', TargetPoses, queue_size=10)
        sensor_detections_pub = rospy.Publisher('/ship_simulator/sensor_measurement', Detections, queue_size=10)
        camera_pose_pub = rospy.Publisher('/ship_simulator/camera_pose', Odometry, queue_size=10)
        
        # Marker Publishers
        ocean_marker_pub = rospy.Publisher('/ship_simulator/markers/ocean_plane', Marker, queue_size=2)
        agent_marker_pub = rospy.Publisher('/ship_simulator/markers/agent_pose', Marker, queue_size=10)
        projection_marker_pub = rospy.Publisher('/ship_simulator/markers/camera_projection', Marker, queue_size=10)
        projection_points_marker_pub = rospy.Publisher('/ship_simulator/markers/camera_projection_points', Marker, queue_size=10)
        targets_marker_pub = rospy.Publisher('/ship_simulator/markers/targets', MarkerArray, queue_size=10)
        agent_trajectory_pub = rospy.Publisher('/ship_simulator/markers/agent_trajectory', Marker, queue_size=10)

        waypt_sub = rospy.Subscriber(self.planner_path_topic, Plan, self.planner_callback)
        rate = rospy.Rate(1/self.sim_env.del_t)  
        counter = 0

        # filename = "./data/" + rospy.get_param('/experiment', 'blank_sim_manager') + "_target_positions.csv"
        # with open(filename, 'w') as f:
        #     f.write("time_stamp,target_id,x,y,heading,linear_speed,angular_speed\n")

        start_time = rospy.Time.now()
        time_since_last_write = start_time

        while not rospy.is_shutdown():
            time = rospy.Time.now()
            frame = "local_enu"
            agent_position = self.get_agent_position(time, frame)
            target_positions = self.get_target_positions(time, frame)

            # if rospy.Time.now() - time_since_last_write > rospy.Duration(10):
            #     with open(filename, 'a') as f:
            #         timestamp = rospy.Time.now()
            #         for t in target_positions.targets:
            #             f.write(str(timestamp.to_sec()) + "," + str(t.id) + "," + str(t.x) + "," + str(t.y) + "," + str(t.heading) + "," + str(t.linear_speed) + "," + str(t.angular_speed) + "\n")
            #         time_since_last_write = timestamp


            target_detections, camera_projection = self.get_target_detections(time, frame)
            # import code; code.interact(local=locals())
            
            camera_pose = self.get_camera_pose(time, frame)
            waypoint_number  = self.get_waypt_num()

            waypt_num_pub.publish(waypoint_number)
            agent_pose_pub.publish(agent_position)
            target_pose_pub.publish(target_positions)
            sensor_detections_pub.publish(target_detections)
            camera_pose_pub.publish(camera_pose)

            ocean_marker_pub.publish(self.get_ocean_marker(time, frame))
            agent_marker_pub.publish(self.get_agent_marker(time, frame, agent_position))
            projection_marker_pub.publish(self.get_projection_marker(time, frame, agent_position, camera_projection))
            projection_points_marker_pub.publish(self.get_projection_points_marker(time, frame, agent_position, camera_projection))
            targets_marker_pub.publish(self.get_targets_marker(time, frame, target_positions))
            if counter % 10 == 0:
                agent_trajectory_pub.publish(self.get_agent_trajectory_marker(time, frame, agent_position))

            counter += 1
            # if counter == 100:
            #     # Currently doing state update every 100 iters
            self.sim_env.update_states()
                # counter = 0
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node("sim_manager_node", anonymous=True)
    obj = SimManager()
    obj.main()
        
            