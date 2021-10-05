#!/usr/bin/env python2

import rospy
from rospkg import RosPack
from rospy.core import is_shutdown
from planner_map_interfaces.msg import Plan
from src.environment import *
from geometry_msgs.msg import PoseStamped
from simple_ships_simulator.msg import TargetsPose
from tf.transformations import quaternion_from_euler

package = RosPack()
package_path = package.get_path("simple_ships_simulator")


class SimManager:
    def __init__(self):
        self.planner_path_topic = rospy.get_param("~planner_path")
        self.sim_env = self.env_setup()

    def env_setup(self):
        targets_list = rospy.get_param("/env_setup/targets")
        
        init_x = rospy.get_param("/env_setup/init_x")
        init_y = rospy.get_param("/env_setup/init_y")
        init_z = rospy.get_param("/env_setup/init_z")
        init_phi = rospy.get_param("/env_setup/init_phi")

        max_omega = rospy.get_param("/env_setup/max_omega")

        vehicle_l = rospy.get_param("/env_setup/vehicle_l")

        vel = rospy.get_param("/env_setup/vel")

        n_rand_targets = rospy.get_param("/env_setup/n_rand_targets")

        del_t = rospy.get_param("/env_setup/del_t")

        return Environment(targets_list, 
                            max_omega, 
                            init_x, 
                            init_y,
                            init_z,
                            init_phi,
                            vehicle_l,
                            vel,
                            n_rand_targets,
                            del_t)
    
    def get_vehicle_position(self):
        vehicle_pose = PoseStamped()
        vehicle_pose.header.frame_id = "local_enu"
        vehicle_pose.header.stamp = rospy.Time.now()
        vehicle_pose.pose.position.x = self.sim_env.vehicle.x
        vehicle_pose.pose.position.y = self.sim_env.vehicle.y
        vehicle_pose.pose.position.z = self.sim_env.vehicle.z

        quat = quaternion_from_euler(0, 0, self.sim_env.vehicle.phi)
        vehicle_pose.pose.orientation.x = quat[0]
        vehicle_pose.pose.orientation.y = quat[1]
        vehicle_pose.pose.orientation.z = quat[2]
        vehicle_pose.pose.orientation.w = quat[3]
        return vehicle_pose
    
    def get_target_positions(self):
        targets_pose = TargetsPose()
        targets_pose.header.frame_id = "local_enu"
        targets_pose.header.stamp = rospy.Time.now()

        for target in self.sim_env.targets:
            target_pose = []
            
            target_pose.append(target.x)
            target_pose.append(target.y)
            target_pose.append(target.data)

            targets_pose.targets.append(target_pose)
        
        return targets_pose
    
    def planner_callback(self):
        pass

    def main(self):
        vehicle_pose_pub = rospy.Publisher('/ship_simulator/vehicle_pose', PoseStamped, queue_size=10)
        target_pose_pub = rospy.Publisher('/ship_simulator/target_poses', TargetsPose, queue_size=10)

        waypt_sub = rospy.Subscriber("/global_path", Plan, self.planner_callback)
        rospy.init_node('sim_manager', anonymous=True)
        rate = rospy.Rate(10)  # 10 Hz
        rospy.spin()
        while not rospy.is_shutdown():
            vehicle_pose_pub.publish(self.get_vehicle_position())
            target_pose_pub.publish(self.get_target_positions())
            rate.sleep()
        
            