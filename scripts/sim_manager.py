#!/usr/bin/env python2

import rospy
from rospkg import RosPack
from rospy.core import is_shutdown
from planner_map_interfaces.msg import Plan
from src.environment import *

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
    
    def main(self):
        