import numpy as np
from scan_simulator_2d import PyScanSimulator2D

import rospy
import tf
from nav_msgs.msg import OccupancyGrid
from tf.transformations import quaternion_from_euler

class SensorModel:


    def __init__(self):
        # Fetch parameters
        self.map_topic = rospy.get_param("~map_topic", "/map")
        self.num_beams_per_particle = rospy.get_param("~num_beams_per_particle", )
        self.scan_theta_discretization = rospy.get_param("~scan_theta_discretization")
        self.scan_field_of_view = rospy.get_param("~scan_field_of_view")
        self.lidar_scale_to_map_scale = rospy.get_param("~lidar_scale_to_map_scale")

        ####################################
        # TODO
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0
        self.eta = 1
        self.epsilon = 0.1

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        self.z_max = self.table_width - 1

        # Precompute the sensor model table
        self.sensor_model_table = None
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
                self.num_beams_per_particle,
                self.scan_field_of_view,
                0, # This is not the simulator, don't add noise
                0.01, # This is used as an epsilon
                self.scan_theta_discretization) 

        # Subscribe to the map
        self.map = None
        self.map_set = False
        rospy.Subscriber(
                self.map_topic,
                OccupancyGrid,
                self.map_callback,
                queue_size=1)
        self.map_resolution = None

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.
        
        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A
        
        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """
        self.sensor_model_table = np.empty(shape=(self.table_width,self.table_width))
        #do p hit
        #TODO: broadcast indices instead of using a for loop
        for z in range(self.table_width):
            for d in range(self.table_width):
                self.sensor_model_table[z,d] = self.calculate_p_hit(z,d)
        
        # normalize p hit
        self.sensor_model_table = self.alpha_hit * self.sensor_model_table / self.sensor_model_table.sum(axis=0, keepdims=1)

        #TODO: broadcast indices instead of using a for loop
        for z in range(self.table_width):
            for d in range(self.table_width):
                rest = self.calculate_p_short(z,d) + self.calculate_p_max(z,d) + self.calculate_p_rand(z,d)
                self.sensor_model_table[z,d] += rest
        
        #normalize
        self.sensor_model_table = self.sensor_model_table / self.sensor_model_table.sum(axis=0, keepdims=1)

    def calculate_p_hit(self, z, d):
        if z >= 0 and z <= self.z_max:
            return self.eta/np.sqrt(2*np.pi*self.sigma_hit**2)*np.exp(-(z-d)**2/(2*self.sigma_hit**2))
        return 0
    
    def calculate_p_short(self, z, d):
        if z <= d and z >= 0 and d != 0:
            return self.alpha_short * 2/d * (1-z/d)
        return 0

    def calculate_p_max(self, z, d):
        if self.z_max - self.epsilon <= z and z <= self.z_max:
            return self.alpha_max/self.epsilon
        return 0 

    def calculate_p_rand(self, z, d):
        if z <= self.z_max and z >= 0:
            return self.alpha_rand/self.z_max 
        return 0


    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:
            
                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar.

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans = self.scan_sim.scan(particles)

        #conversion from meters to pixels
        observation = np.clip(observation/(self.map_resolution*self.lidar_scale_to_map_scale), 0, 200)
        scans = np.clip(scans/(self.map_resolution*self.lidar_scale_to_map_scale), 0, 200)

        #rounding for indexing
        scans = np.round(scans).astype(int)  
        observation = np.round(observation).astype(int)

        #get probabilities
        result = self.sensor_model_table[observation, scans]

        #multiply probabilities
        result = np.product(result, axis=1)

        return result**(1/2.2)

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double)/100.
        self.map = np.clip(self.map, 0, 1)
        self.map_resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position 
        origin_o = map_msg.info.origin.orientation
        origin_o = tf.transformations.euler_from_quaternion((
                origin_o.x,
                origin_o.y,
                origin_o.z,
                origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
                self.map,
                map_msg.info.height,
                map_msg.info.width,
                map_msg.info.resolution,
                origin,
                0.5) # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
