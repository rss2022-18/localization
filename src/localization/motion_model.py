import numpy as np
import rospy


class MotionModel:

    def __init__(self, mean=0, sigma=.1):

        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.
        self.mean = mean
        self.sigma = sigma
        self.deterministic = rospy.get_param('/localization/deterministic')
        print("[Insert joke for graders here]")
        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        n, _ = particles.shape
        updated_particles = np.zeros([n, 3])
        if not self.deterministic:
            noise = np.random.normal(self.mean, self.sigma, (n, 3))
            odometry = odometry + noise
        for i in range(n):
            R = self.make_rotation_matrix(particles[i, 2])
            T = self.make_transformation_matrix(R, particles[i, :])
            updated_particles[i, :2] = np.matmul(T, odometry[i, :])[:, :2]
            updated_particles[i, 2] = particles[i, 2] + odometry[i, 2]
        return updated_particles

    def make_transformation_matrix(self, R, p):
        """
        Creates a 3x3 transformation matrix from
        the rotation matrix R and the position vector p
        """
        return np.array([[R[0, 0], R[0, 1], p[0]],
                         [R[1, 0], R[1, 1], p[1]],
                         [0, 0, 1]])

    def make_rotation_matrix(self, theta):
        """
        Creates a 2x2 rotation matrix from theta
        """
        return np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
