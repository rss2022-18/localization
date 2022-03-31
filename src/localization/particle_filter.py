#!/usr/bin/env python2

# from concurrent.futures import thread ## look into this more, dont know what this does.
import rospy
import numpy as np
from scipy import stats
from sensor_model import SensorModel
from motion_model import MotionModel
import threading  # useful for actually threading the particle filter.

import tf

from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, PoseStamped
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker

from tf.transformations import euler_from_quaternion, quaternion_from_euler


class ParticleFilter:

    def __init__(self):
        # Get parameters
        self.particle_filter_frame = rospy.get_param(
            "~particle_filter_frame", "/base_link_pf")
        self.num_particles = rospy.get_param("~num_particles", 200)
        self.num_beams_per_particle = rospy.get_param(
            "~num_beams_per_particle", 100)

        # Initialize publishers/subscribers
        #
        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        # Other Initializations:
        self.particle_pos = np.zeros((self.num_particles, 3))  # 3 is x,y,z
        self.particle_prob = np.ones(
            (self.num_particles))*1.0/float(self.num_particles)

        self.t_minus_1 = None
        self.lock_thread = threading.RLock()

        # Transform Listeners
        self.br_transform = tf.TransformBroadcaster()
        self.tf_listener = tf.TransformListener()

        # Initialize the models
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel()

        # Debugging and visualization
        self.visuals = True
        self.noise = rospy.get_param('~deterministic')
        #self.noise   = False

        scan_topic = rospy.get_param("~scan_topic", "/scan")
        odom_topic = rospy.get_param("~odom_topic", "/odom")
        map_topic = rospy.get_param("~map_topic", "/map")

        self.laser_sub = rospy.Subscriber(scan_topic, LaserScan,
                                          self.lidar_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber(odom_topic, Odometry,
                                         self.odom_cb, queue_size=1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.
        self.pose_sub = rospy.Subscriber("/initialpose", PoseWithCovarianceStamped,
                                         self.pose_init_cb, queue_size=1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.
        self.odom_pub = rospy.Publisher(
            "/pf/pose/odom", Odometry, queue_size=1)
        self.visualize_pub = rospy.Publisher(
            "/particle_points", Marker, queue_size=1)
        self.visualize_pos = VisualizePos()

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

    def publish_odometry(self, x_avg, y_avg, quat):

        p = Odometry()
        p.header.frame_id = "/map"
        p.header.stamp = rospy.Time.now()
        p.pose.pose.position.x = x_avg
        p.pose.pose.position.y = y_avg
        p.pose.pose.position.z = 0

        p.pose.pose.orientation.x = quat[0]
        p.pose.pose.orientation.y = quat[1]
        p.pose.pose.orientation.z = quat[2]
        p.pose.pose.orientation.w = quat[3]

        self.br_transform.sendTransform((x_avg, y_avg, 0),
                                        quat,
                                        p.header.stamp,
                                        self.particle_filter_frame,
                                        "/map")

        self.odom_pub.publish(p)

    """
    Function: lidar_cb
    Inputs: Self, laser_scan data

    Whenever we get sensor data, use the sensor model to compute the particle probabilities.
    Use these probabilities to resample the particles based on these probabilites. 

    When the particles are updated:
    Determine the "average" particle pose and publish that transform. 
    """

    def lidar_cb(self, data):
        # print("Lidar!")
        with self.lock_thread:  # ROS Callbacks are not thread safe!
            angle_step = int(
                np.ceil(len(data.ranges)/float(self.num_beams_per_particle)))
            # TODO Look into this more
            data_downsized = np.array(data.ranges)[::angle_step]

            # Sensor Model
            self.particle_prob = self.sensor_model.evaluate(
                self.particle_pos, data_downsized)
            self.particle_prob = self.particle_prob/np.sum(self.particle_prob)

            # Resampling ->  x,y,theta
            # use np.random.choice!
            index_sample = np.random.choice(np.arange(
                0, self.num_particles), size=self.num_particles, p=self.particle_prob)
            self.particle_pos = self.particle_pos[index_sample, :]

            # Change particle positions and publish to pose odom.
            x_avg = stats.mode(self.particle_pos[:, 0]).mode[0]
            y_avg = stats.mode(self.particle_pos[:, 1]).mode[0]

            th_avg = stats.mode(self.particle_pos[:, 2]).mode[0]  # a yaw value

            sum_of_sin = np.sum(np.sin(self.particle_pos[:, 2]))
            sum_of_cos = np.sum(np.cos(self.particle_pos[:, 2]))
            th_avg_ang = np.arctan2(sum_of_sin, sum_of_cos)

            pub_quat = quaternion_from_euler(0, 0, th_avg_ang)

            #pub_quat = quaternion_from_euler(0,0,th_avg)

            # TODO: Visualize Particles.
            if self.visuals:
                #print('Hi! Visualizing')
                self.visualize_pos.draw(self.particle_pos)
                self.visualize_pub.publish(self.visualize_pos.line)

            # publish our odometry!
            self.publish_odometry(x_avg, y_avg, pub_quat)

    """
    Function: odom_cb
    Inputs: self, odometry data

    Whenever we get Odometry data, use the motion model to update the particle positions. 

    When the particles are updated, determine the "average" particle pose and publish the transform.
    
    """

    def odom_cb(self, data):

        # print("Odometry!")
        with self.lock_thread:  # ROS Callbacks are not thread safe!
            odom_to_world = PoseStamped()
            odom_to_world.header = data.header
            odom_to_world.pose = data.pose.pose
            pt_trans = self.tf_listener.transformPose("/map", odom_to_world)

            # initialization
            if self.t_minus_1 is None:
                self.t_minus_1 = rospy.get_time()

                #state is x,y,theta
                th = self.quat_to_yaw(pt_trans.pose.orientation)

                state = [pt_trans.pose.position.x,
                         pt_trans.pose.position.y, th]
                print("Initial Point:" + str(state))
                self.particle_pos = np.random.normal(loc=state,
                                                     # Try with no noise  [1.5,1.5,np.pi/6]
                                                     scale=[0.5, 0.5, np.pi/8],
                                                     size=(self.num_particles, 3))
                return

            # Have the option to add noise to odometry, in this case normal noise
            if self.noise:
                # normal distribution noise
                noise_x = np.random.normal(scale=0.125)
                noise_y = np.random.normal(scale=0.125)
                noise_th = np.random.normal(
                    scale=np.pi/16)  # Try with no noise
            else:
                noise_x = 0
                noise_y = 0
                noise_th = 0

            delta_t = rospy.get_time() - self.t_minus_1
            self.t_minus_1 = rospy.get_time()

            # Propagate forward!

            odom_data = np.array([(data.twist.twist.linear.x+noise_x)*delta_t,
                                  (data.twist.twist.linear.y+noise_y)*delta_t,
                                  (data.twist.twist.angular.z+noise_th)*delta_t])

            self.particle_pos = self.motion_model.evaluate(
                self.particle_pos, odom_data)

            # Position:
            (x_avg, y_avg) = np.mean(self.particle_pos[:, :2], axis=0)

            if self.visuals:
                #print('Hi! Visualizing')
                self.visualize_pos.draw(self.particle_pos)
                self.visualize_pub.publish(self.visualize_pos.line)

            """
            Consider the mean of circular quantities. 
            """
            sum_of_sin = np.sum(np.sin(self.particle_pos[:, 2]))
            sum_of_cos = np.sum(np.cos(self.particle_pos[:, 2]))
            th_avg_ang = np.arctan2(sum_of_sin, sum_of_cos)

            quat_pub = quaternion_from_euler(0, 0, th_avg_ang)

            self.publish_odometry(x_avg, y_avg, quat_pub)

    """
    Consider how to initialize the particles. Use some of the interactive points in Rviz to set an
    initialize guess of the robots location with a random spread of locations spread around a
    clicked point or pose. Make it simple on yourself and avoid using the kidnapped robot problem. 
    """

    def pose_init_cb(self, data):
        print("Pose Initialization!")
        with self.lock_thread:  # ROS callbacks are not thread safe!
            x = data.pose.pose.position.x
            y = data.pose.pose.position.y
            th = self.quat_to_yaw(data.pose.pose.orientation)
            self.particle_pos = np.random.normal(loc=[x, y, th], scale=[
                                                 0.5, 0.5, np.pi/8], size=(self.num_particles, 3))  # [1.5,1.5,np.pi/6]

    def quat_to_yaw(self, quaternion):

        my_quat = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
        rpy = euler_from_quaternion(my_quat)
        roll = rpy[0]
        pitch = rpy[1]
        yaw = rpy[2]

        return yaw


"""
Used for visualizing our particles. 
"""


class VisualizePos:
    def __init__(self):

        # Set parameters for the published message.
        self.line = Marker()
        self.line.type = Marker.POINTS
        self.line.header.frame_id = "/map"
        self.line.scale.x = 0.1
        self.line.scale.y = 0.1
        self.line.color.a = 1.
        self.line.color.r = 1
        self.line.color.g = 1
        self.line.color.b = 0

    def draw(self, positions):
        x_pos = positions[:, 0]
        y_pos = positions[:, 1]

        self.line.points = []

        for i in range(len(x_pos)):
            point = Point()
            point.x = x_pos[i]
            point.y = y_pos[i]

            self.line.points.append(point)


if __name__ == "__main__":
    rospy.init_node("particle_filter")
    pf = ParticleFilter()
    rospy.spin()
