#!/usr/bin/env python

import os
import rospy
import numpy as np
from rospkg import RosPack
import tf
from planner_map_interfaces.msg import Plan, PlanRequest, GroundTruthTargets, GroundTruthTarget
from simple_ipp_sim.environment import *
from geometry_msgs.msg import PoseStamped, Point, Pose, Quaternion
from std_msgs.msg import ColorRGBA
from nav_msgs.msg import Odometry
from std_msgs.msg import UInt8, UInt32, Float32
from simple_ipp_sim.msg import Detections
from tf.transformations import quaternion_from_euler, euler_from_quaternion

from visualization_msgs.msg import Marker, MarkerArray

package = RosPack()
package_path = package.get_path("simple_ipp_sim")

visualization_scale = 50.0

# /onr_ipp_node/visualization_scale

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
    def __init__(self, agent_names):
        self.agent_names = agent_names
        self.num_agents = len(agent_names)
        self.planner_path_topic = rospy.get_param("~planner_path")
        self.sim_env = self.sim_manager_node()
        self.agent_traj_list = [[] for i in range(self.num_agents)]

        self.pause_while_planning = rospy.get_param("sim_manager_node/pause_while_planning")
        self.waiting_for_plan = False

        rospy.loginfo("simulating " + str(self.num_agents) + " agent(s)")

    def sim_manager_node(self):
        # ships
        targets_list = rospy.get_param("~targets", [])

        # drone state
        init_x = rospy.get_param("~init_x")
        init_y = rospy.get_param("~init_y")
        init_z = rospy.get_param("~init_z")
        init_psi = rospy.get_param("~init_psi")

        max_omega = rospy.get_param("~max_omega")
        max_zvel = rospy.get_param("~max_zvel")

        agent_l = rospy.get_param("~agent_l")

        hvel = rospy.get_param("~hvel")
        vvel = rospy.get_param("~vvel")

        n_rand_targets = rospy.get_param("~n_rand_targets")

        del_t = rospy.get_param("~del_t")

        K_p = rospy.get_param("~K_p")
        K_p_z = rospy.get_param("~K_p_z")

        waypoint_threshold = rospy.get_param("~waypoint_threshold")

        agent_name = self.agent_names[0] #assume we're always simulating atleast one agent. Default is ""
        sensor_focal_length = rospy.get_param(agent_name  + "/sensor/focal_length")
        sensor_width = rospy.get_param(agent_name  + "/sensor/width")
        sensor_height = rospy.get_param(agent_name  + "/sensor/height")
        sensor_pitch = rospy.get_param(agent_name  + "/sensor/pitch")
        sensor_max_range = max(rospy.get_param(agent_name + "/sensor/max_range"))
        sensor_endurance = rospy.get_param(agent_name + "/sensor/endurance")
        sensor_hedge = rospy.get_param(agent_name + "/sensor/hedge")

        return Environment(targets_list, 
                            max_omega, 
                            max_zvel,
                            init_x, 
                            init_y,
                            init_z,
                            init_psi,
                            K_p,
                            K_p_z,
                            self.num_agents,
                            agent_l,
                            hvel,
                            vvel,
                            n_rand_targets,
                            del_t,
                            waypoint_threshold,
                            sensor_focal_length,
                            sensor_width,
                            sensor_height,
                            sensor_pitch,
                            sensor_max_range,
                            sensor_endurance,
                            sensor_hedge
                            )
    
    def get_agent_odometry(self, time, frame):
        agent_odom_list = []
        for id_num in range(self.num_agents):
            agent_odom = Odometry()
            agent_odom.header.frame_id = frame
            agent_odom.header.stamp = time
            agent_odom.pose.pose.position.x = self.sim_env.agent[id_num].x
            agent_odom.pose.pose.position.y = self.sim_env.agent[id_num].y
            agent_odom.pose.pose.position.z = self.sim_env.agent[id_num].z
            quat = quaternion_from_euler(0, 0, self.sim_env.agent[id_num].psi)
            agent_odom.pose.pose.orientation.x = quat[0]
            agent_odom.pose.pose.orientation.y = quat[1]
            agent_odom.pose.pose.orientation.z = quat[2]
            agent_odom.pose.pose.orientation.w = quat[3]
            agent_odom_list.append(agent_odom)
        return agent_odom_list
    
    def get_target_positions(self, time, frame):
        target_poses = GroundTruthTargets()
        target_poses.header.frame_id = frame
        target_poses.header.stamp = time

        for target in self.sim_env.targets:
            target_pose = GroundTruthTarget()

            target_pose.id = target.id
            target_pose.x = target.x
            target_pose.y = target.y
            target_pose.heading = target.heading
            target_pose.linear_speed = target.linear_speed
            target_pose.angular_speed = target.angular_speed
            target_pose.is_detected = target.is_detected

            # print target_pose
            target_poses.targets.append(target_pose)
        
        return target_poses
    
    def get_camera_pose(self, time, frame):
        camera_pose_list = []
        for id_num in range(self.num_agents):
            camera_pose = Odometry()
            camera_pose.header.frame_id = frame
            camera_pose.header.stamp = time
            camera_pose.pose.pose.position.x = self.sim_env.agent[id_num].x
            camera_pose.pose.pose.position.y = self.sim_env.agent[id_num].y
            camera_pose.pose.pose.position.z = self.sim_env.agent[id_num].z
            quat = quaternion_from_euler(0, self.sim_env.sensor_pitch, self.sim_env.agent[id_num].psi)
            camera_pose.pose.pose.orientation.x = quat[0]
            camera_pose.pose.pose.orientation.y = quat[1]
            camera_pose.pose.pose.orientation.z = quat[2]
            camera_pose.pose.pose.orientation.w = quat[3]
            cov_matrix = np.zeros((6,6))
            cov_matrix[0,0] = self.sim_env.get_agent_uncertainty(id_num)[0]
            cov_matrix[1,1] = self.sim_env.get_agent_uncertainty(id_num)[1]
            cov_matrix[2,2] = self.sim_env.get_agent_uncertainty(id_num)[2]
            cov_matrix[3,3] = self.sim_env.get_agent_uncertainty(id_num)[3]
            cov_matrix[4,4] = self.sim_env.get_agent_uncertainty(id_num)[4]
            cov_matrix[5,5] = self.sim_env.get_agent_uncertainty(id_num)[5]
            camera_pose.pose.covariance = cov_matrix.flatten().tolist()
            camera_pose_list.append(camera_pose)
        return camera_pose_list
    
    def get_waypoint_num(self):
        waypoint_num_list = []
        for id_num in range(self.num_agents):
            waypoint_number = UInt32()
            waypoint_number.data = self.sim_env.curr_waypoint_num[id_num]
            waypoint_num_list.append(waypoint_number)
        return waypoint_num_list

    def get_remaining_budget(self):
        remaining_budget_list = []
        for id_num in range(self.num_agents):
            remaining_budget = Float32()
            remaining_budget.data = self.sim_env.remaining_budget[id_num]
            remaining_budget_list.append(remaining_budget)
        return remaining_budget_list
    
    def get_target_detections(self, time, frame):
        detections_msg_list = []
        detected_targets, camera_projection = self.sim_env.get_sensor_measurements()
        for i in range(self.num_agents):
            detection_msg = Detections()
            detection_msg.header.frame_id = frame
            detection_msg.header.stamp = time
            for target in detected_targets[i]:
                detection_msg.headings.append(self.sim_env.get_target_heading_noise(target.heading))
                target_camera_unit_vector = Point()
                range_to_target = np.linalg.norm(np.array([target.x, target.y, 0]) - 
                                                np.array([self.sim_env.agent[i].x, 
                                                        self.sim_env.agent[i].y, self.sim_env.agent[i].z]))
                i_hat = (target.x - self.sim_env.agent[i].x) / range_to_target
                j_hat = (target.y - self.sim_env.agent[i].y) / range_to_target
                k_hat = - self.sim_env.agent[i].z / range_to_target
                # R = np.matmul(self.sim_env.sensor.Rz(self.sim_env.agent.phi), self.sim_env.sensor.Ry(self.sim_env.sensor_pitch))
                # print("current pitch is", self.sim_env.sensor_pitch)
                # print("current psi is", self.sim_env.agent.psi)
                R = np.matmul(self.sim_env.sensor.Rz(self.sim_env.agent[i].psi),self.sim_env.sensor.Ry(self.sim_env.sensor_pitch))
                R_inv = np.linalg.inv(R)
                camera_frame_pose = np.matmul(R_inv, [i_hat, j_hat, k_hat])
                target_camera_unit_vector.x = camera_frame_pose[0]
                target_camera_unit_vector.y = camera_frame_pose[1]
                target_camera_unit_vector.z = camera_frame_pose[2]
                detection_msg.target_camera_vectors.append(target_camera_unit_vector)
                detection_msg.target_ids.append(target.id)
            detections_msg_list.append(detection_msg)

        return detections_msg_list, camera_projection
    
    def planner_callback(self, msg, id_num):
        self.sim_env.update_waypoints(msg, id_num)
        self.waiting_for_plan = False

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



    def get_agent_marker(self, time, frame, agent_odom):
        agent_marker_list = MarkerArray()
        for id_num in range(self.num_agents):
            odom = agent_odom[id_num]
            agent_marker = Marker()
            agent_marker.header.frame_id = frame
            agent_marker.header.stamp = time
            agent_marker.ns = "agent_mesh"
            agent_marker.id = id_num
            agent_marker.type = Marker.MESH_RESOURCE
            agent_marker.action = Marker.ADD
            agent_marker.mesh_use_embedded_materials = False
            agent_marker.mesh_resource = "package://simple_ipp_sim/meshes/vtol_to_scale.dae"
            agent_marker.lifetime = rospy.Duration()
            agent_marker.pose.position = odom.pose.pose.position
            agent_marker.pose.orientation = odom.pose.pose.orientation
            agent_marker.color.r = .8
            agent_marker.color.g = 0.95
            agent_marker.color.b = 1.0
            agent_marker.color.a = .99
            agent_marker.scale.x = 1.0 * visualization_scale
            agent_marker.scale.y = 1.0 * visualization_scale
            agent_marker.scale.z = 1.0 * visualization_scale
            agent_marker_list.markers.append(agent_marker)
        return agent_marker_list

    def get_agent_trajectory_marker(self, time, frame, agent_odom):
        trajectory_list = MarkerArray()
        for id_num in range(self.num_agents):
            odom = agent_odom[id_num]
            agent_traj = self.agent_traj_list[id_num]

            trajectory_marker = Marker()
            trajectory_marker.header.frame_id = frame
            trajectory_marker.header.stamp = time
            trajectory_marker.ns = "agent_trajectory"
            trajectory_marker.id = 0
            trajectory_marker.type = Marker.LINE_STRIP
            trajectory_marker.action = Marker.ADD
            trajectory_marker.lifetime = rospy.Duration(10.0)

            agent_traj.append([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z])
            if len(agent_traj) > 500:  # setting traj length to 100
                agent_traj.pop(0)

            trajectory_marker.pose.position.x = 0
            trajectory_marker.pose.position.y = 0
            trajectory_marker.pose.position.z = 0
            
            for i in range(1, len(agent_traj)):
                trajectory_marker.points.append(Point(agent_traj[i][0], 
                                    agent_traj[i][1], agent_traj[i][2]))
            
            trajectory_marker.color.r = 1
            trajectory_marker.color.g = 69/255
            trajectory_marker.color.b = 0
            trajectory_marker.color.a = 0.3
            trajectory_marker.scale.x = 0.1 * visualization_scale
            trajectory_marker.scale.y = 0.1 * visualization_scale
            trajectory_marker.scale.z = 0.1 * visualization_scale

            trajectory_list.markers.append(trajectory_marker)

        return trajectory_list

    """
    Four points of the sensor footprint polygon. 
    """
    def get_projection_points_marker(self, time, frame, agent_odom, camera_projection):
        projection_points_list = MarkerArray()
        for id_num in range(self.num_agents):
            odom = agent_odom[id_num]
            projection = camera_projection[id_num]

            projection_points_marker = Marker()
            projection_points_marker.header.frame_id = frame
            projection_points_marker.header.stamp = time
            projection_points_marker.ns = "projection_marker"
            projection_points_marker.id = id_num
            projection_points_marker.type = Marker.POINTS
            projection_points_marker.action = Marker.ADD
            projection_points_marker.color.r = 1.0
            projection_points_marker.color.g = 1.0
            projection_points_marker.color.b = 0.0
            projection_points_marker.color.a = 1.0
            projection_points_marker.scale.x = 0.2 * visualization_scale
            projection_points_marker.scale.y = 0.2 * visualization_scale
            projection_points_marker.scale.z = 0.2 * visualization_scale

            points = []
            for np_point in projection:
                ros_point = Point()
                ros_point.x = np_point[0]
                ros_point.y = np_point[1]
                ros_point.z = np_point[2]
                points.append(ros_point)
            projection_points_marker.points = points

            projection_points_list.markers.append(projection_points_marker)

        return projection_points_list

    def get_projection_marker(self, time, frame, agent_odom, camera_projection):
        marker_list = MarkerArray()
        for id_num in range(self.num_agents):
            odom = agent_odom[id_num]
            projection = camera_projection[id_num]

            projection_marker = Marker()
            projection_marker.header.frame_id = frame
            projection_marker.header.stamp = time
            projection_marker.ns = "projection_marker"
            projection_marker.id = id_num
            projection_marker.type = Marker.LINE_STRIP
            projection_marker.action = Marker.ADD
            projection_marker.color.r = 1
            projection_marker.color.g = 69/255
            projection_marker.color.b = 0
            projection_marker.color.a = 1.0
            projection_marker.scale.x = 0.1 * visualization_scale  # in meters
            projection_marker.scale.y = 0.1 * visualization_scale
            projection_marker.scale.z = 0.1 * visualization_scale

            points = []
            agent_point = Point()
            agent_point.x = odom.pose.pose.position.x
            agent_point.y = odom.pose.pose.position.y
            agent_point.z = odom.pose.pose.position.z

            # connect the projected camera bounds
            
            for edge in range(len(projection)):
                point_a = Point()
                point_a.x = projection[edge][0]
                point_a.y = projection[edge][1]
                point_a.z = projection[edge][2]

                point_b = Point()
                point_b.x = projection[(edge + 1) % len(projection)][0]
                point_b.y = projection[(edge + 1) % len(projection)][1]
                point_b.z = projection[(edge + 1) % len(projection)][2]

                points.append(point_b)
                points.append(point_a)
                points.append(agent_point)
                points.append(point_b)
                points.append(point_a)

            projection_marker.points = points
            marker_list.markers.append(projection_marker)

        return marker_list
        
    
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
            target_marker.mesh_use_embedded_materials = False
            target_marker.mesh_resource = os.path.join("package://simple_ipp_sim", rospy.get_param("/sim_manager_node/target_mesh"))

            target_marker.lifetime = rospy.Duration()
            quat = quaternion_from_euler(0, 0, target.heading)
            target_marker.pose = Pose(Point(target.x, 
                                            target.y,
                                            0),  # z offset to make it appear above grid-map
                                            Quaternion(quat[0], quat[1], quat[2], quat[3]))

            target_marker.color = get_color(target.id)
            target_marker.scale.x = 0.2 * visualization_scale
            target_marker.scale.y = 0.2 * visualization_scale
            target_marker.scale.z = 0.2 * visualization_scale
            targets_marker_array.markers.append(target_marker)
        
        return targets_marker_array
    
    def plan_request_callback(self, plan_request,id_num):
        self.waiting_for_plan = True
        self.sim_env.agent[id_num].vel = plan_request.desired_speed
        self.sim_env.hvel = plan_request.desired_speed
        self.sim_env.remaining_budget[id_num] = plan_request.maximum_range

        priority_list = {}
        for target in plan_request.target_priors:
            if target.target.header.frame_id != "":
                priority_list[str(target.target.id)] = target.target.priority
        # print(priority_list)
        rospy.set_param("~priority", priority_list)

        if rospy.get_param("/sim_manager_node/set_agent_pose_to_plan_request"):
            rospy.loginfo("Teleporting agent to plan request position")
            self.agent_traj_list[id_num] = []
            agent_pose = plan_request.start_pose
            self.sim_env.agent[id_num].x = agent_pose.position.x
            self.sim_env.agent[id_num].y = agent_pose.position.y
            self.sim_env.agent[id_num].z = agent_pose.position.z
            self.sim_env.prev_agentxyz[id_num] = [agent_pose.position.x, agent_pose.position.y, agent_pose.position.z]
            # https://github.com/ros/geometry/issues/109#issuecomment-344702754
            explicit_quat = [agent_pose.orientation.x, agent_pose.orientation.y, agent_pose.orientation.z, agent_pose.orientation.w]
            roll, pitch, yaw = euler_from_quaternion(explicit_quat)
            self.sim_env.agent[id_num].psi = yaw  # yaw angle

        if rospy.get_param("/sim_manager_node/sample_targets_from_plan_request"):
            rospy.loginfo("Sampling true simulated target states from plan request prior distributions")
            self.sample_target_state_from_target_priors(plan_request.target_priors)
        elif rospy.get_param("/sim_manager_node/set_targets_to_plan_request"):
            rospy.loginfo("Set true simulated target states from plan request prior distributions")
            self.init_target_state_from_target_priors(plan_request.target_priors)

    
    def sample_target_state_from_target_priors(self, target_priors):
        """
        Given the list of target priors, sample the true simulated target state from the multivariate Gaussian distribution.
        """
        self.sim_env.targets[:] = []
        for prior in target_priors:
            t = prior.target
            if t and not (t.x == 0 and t.y == 0 and t.heading == 0 and t.linear_speed == 0 and t.angular_speed == 0):
                means = t.x, t.y, t.heading, t.linear_speed, t.angular_speed
                covs = np.array(t.covariance).reshape(5,5)
                # covs[2, 2]  = covs[3,3] = covs[4,4] = 0.0001  # ignore heading, speed, angle
                target_state = np.random.multivariate_normal([0,0,0,0,0], covs, 1)[0]
                # target_state *= 3/4  # scale down the variance
                sim_target = Target(
                    id=t.id,
                    init_x=target_state[0] + t.x,
                    init_y=target_state[1] + t.y,
                    heading=target_state[2] + t.heading,
                    linear_speed=target_state[3] + t.linear_speed,
                    angular_speed=target_state[4] + t.angular_speed,
                    linear_speed_std=0.00,
                    angular_speed_std=0.000
                )
                self.sim_env.targets.append(sim_target)
        rospy.loginfo("Added " + str(len(self.sim_env.targets)) + " simulated targets")

    def init_target_state_from_target_priors(self, target_priors):
        """
        Given the list of target priors, init the true simulated target state.
        """
        self.sim_env.targets[:] = []
        for prior in target_priors:
            t = prior.target
            if t and not (t.x == 0 and t.y == 0 and t.heading == 0 and t.linear_speed == 0 and t.angular_speed == 0):
                sim_target = Target(
                    id=t.id,
                    init_x=t.x,
                    init_y=t.y,
                    heading=t.heading,
                    linear_speed=t.linear_speed,
                    angular_speed=t.angular_speed,
                    linear_speed_std=0.0,
                    angular_speed_std=0.0
                )
                self.sim_env.targets.append(sim_target)
        rospy.loginfo("Added " + str(len(self.sim_env.targets)) + " simulated targets")

    def main(self):
        
        odom_pubs = [[] for i in range(self.num_agents)]
        sensor_detection_pubs = [[] for i in range(self.num_agents)]
        camera_pose_pubs = [[] for i in range(self.num_agents)]
        remaining_budget_pubs = [[] for i in range(self.num_agents)]
        if not rospy.get_param( self.agent_names[0] + "/onr_ipp_node/use_own_waypoint_manager"):
            waypoint_num_pubs = [[] for i in range(self.num_agents)]

        for idx in range(self.num_agents):
            odom_pubs[idx] = rospy.Publisher(agent_names[idx] + "/odom", Odometry, queue_size=10)
            sensor_detection_pubs[idx] = rospy.Publisher(agent_names[idx] + '/sensor_measurement', Detections, queue_size=10)
            camera_pose_pubs[idx] = rospy.Publisher(agent_names[idx] + '/camera_pose', Odometry, queue_size=10)
            remaining_budget_pubs[idx] = rospy.Publisher(agent_names[idx] + '/remaining_budget', Float32, queue_size=10)
            rospy.Subscriber(agent_names[idx] + "/planner/plan_request", PlanRequest, self.plan_request_callback, (idx))
            rospy.Subscriber(agent_names[idx] + self.planner_path_topic, Plan, self.planner_callback, (idx))
            if not rospy.get_param(agent_names[idx] + "/onr_ipp_node/use_own_waypoint_manager"):
                waypoint_num_pubs[idx] = rospy.Publisher(agent_names[idx] + '/waypoint_num', UInt32, queue_size=10)
        target_pose_pub = rospy.Publisher('simulator/target_poses', GroundTruthTargets, queue_size=10)
        
        #visualization publishers
        ocean_marker_pub = rospy.Publisher('simulator/markers/ocean_plane', Marker, queue_size=2)
        agent_marker_pub = rospy.Publisher('simulator/markers/agent_mesh', MarkerArray, queue_size=10)
        projection_marker_pub = rospy.Publisher('simulator/markers/camera_projection', MarkerArray, queue_size=10)
        projection_points_marker_pub = rospy.Publisher('simulator/markers/camera_projection_points', MarkerArray, queue_size=10)
        targets_marker_pub = rospy.Publisher('simulator/markers/targets', MarkerArray, queue_size=10)
        agent_trajectory_pub = rospy.Publisher('simulator/markers/agent_trajectory', MarkerArray, queue_size=10)

        rate = rospy.Rate(1.0/self.sim_env.del_t)  
        br = tf.TransformBroadcaster()
        counter = 0

        # filename = "./data/" + rospy.get_param('/experiment', 'blank_sim_manager') + "_target_positions.csv"
        # with open(filename, 'w') as f:
        #     f.write("time_stamp,target_id,x,y,heading,linear_speed,angular_speed\n")

        # visualization markers
        projection_marker_pub = rospy.Publisher('simulator/markers/camera_projection', MarkerArray, queue_size=10)

        start_time = rospy.Time.now()
        print("\nSim is ready to go\n")

        while not rospy.is_shutdown():
            if self.pause_while_planning and self.waiting_for_plan:
                pass # do nothing while waiting for plan
            else:
                counter += 1
                self.sim_env.update_states()

            time = rospy.Time.now()
            frame = "local_enu"
            agent_odom = self.get_agent_odometry(time, frame)
            target_positions = self.get_target_positions(time, frame)
            target_detections, camera_projections = self.get_target_detections(time, frame)
            camera_pose = self.get_camera_pose(time, frame)
            remaining_budget = self.get_remaining_budget()
            waypoint_num = self.get_waypoint_num()

            for id_num in range(self.num_agents):
                br.sendTransform(
                    (agent_odom[id_num].pose.pose.position.x, agent_odom[id_num].pose.pose.position.y, agent_odom[id_num].pose.pose.position.z),
                    (agent_odom[id_num].pose.pose.orientation.x, agent_odom[id_num].pose.pose.orientation.y, 
                    agent_odom[id_num].pose.pose.orientation.z, agent_odom[id_num].pose.pose.orientation.w),
                    rospy.Time.now(),
                    self.agent_names[id_num] + "_base_link",
                    "local_enu")
                odom_pubs[id_num].publish(agent_odom[id_num])
                camera_pose_pubs[id_num].publish(camera_pose[id_num])
                sensor_detection_pubs[id_num].publish(target_detections[id_num])
                remaining_budget_pubs[id_num].publish(remaining_budget[id_num])
                if not rospy.get_param( self.agent_names[0] + "/onr_ipp_node/use_own_waypoint_manager"): 
                    waypoint_num_pubs[id_num].publish(waypoint_num[id_num])

            target_pose_pub.publish(target_positions)

            # visualizations
            ocean_marker_pub.publish(self.get_ocean_marker(time, frame))
            agent_marker_pub.publish(self.get_agent_marker(time, frame, agent_odom))
            projection_marker_pub.publish(self.get_projection_marker(time, frame, agent_odom, camera_projections))
            targets_marker_pub.publish(self.get_targets_marker(time, frame, target_positions))
            projection_points_marker_pub.publish(self.get_projection_points_marker(time, frame, agent_odom, camera_projections))
            if counter % 10 == 0:
                agent_trajectory_pub.publish(self.get_agent_trajectory_marker(time, frame, agent_odom))

            # calculate the velocity of the target
            # curr_time  = rospy.get_time()
            # if self.prev_time != -1 and len(self.prev_target_positions.targets) != 0:
            #     for i in range(len(target_positions.targets)):
            #         # print(target_positions)
            #         # print()
            #         delta_x = target_positions.targets[i].x - self.prev_target_positions.targets[i].x
            #         delta_y = target_positions.targets[i].y - self.prev_target_positions.targets[i].y
                    
            #         delta_t = curr_time - self.prev_time
            #         velocity = np.sqrt(delta_x**2 + delta_y**2) / delta_t
            #         print("Target " + str(i) + " velocity: " + str(velocity))

            # self.prev_time = curr_time
            # self.prev_target_positions = target_positions

            rate.sleep()


if __name__ == '__main__':
    rospy.init_node("sim_manager_node")
    agent_names = rospy.get_param("~agent_names")
    if len(agent_names) == 0:
        agent_names = [""]
    obj = SimManager(agent_names)
    obj.main()
        
            