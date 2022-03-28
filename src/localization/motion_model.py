#!/usr/bin/env python2
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
        self.alpha = [0.1, 0.1, 0.1, 0.1]
        self.deterministic = rospy.get_param('~deterministic')
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
        for i in range(n):
            R = self.make_rotation_matrix(particles[i, 2])
            T = self.make_transformation_matrix(R, particles[i, :])
            p_t = np.array([odometry[0], odometry[1], 1])
            updated_particles[i, :2] = np.matmul(T, p_t)[:2]
        updated_particles[:, 2] = particles[:, 2] + odometry[2]
        # Noise model based on Probabilistic Robotics Section 5.4
        # Key Assumptions
        # - Three separate transformation steps: rotation, translation, rotation
        #  - Our motion model assumes that each transformation step is corrupted by independent noise.

        if not self.deterministic:
            # Calculate relative motion parameters
            odometry = updated_particles - particles
            rot1 = np.arctan2(odometry[:, 1], odometry[:, 0]) - particles[:, 2]
            trans = np.sqrt(odometry[:, 0]**2 + odometry[:, 1]**2)
            rot2 = odometry[:, 2] - rot1

            # Add noise
            noisy_rot1 = rot1 - \
                self.sample(self.alpha[0]*rot1 + self.alpha[1] * trans)
            noisy_trans = trans - \
                self.sample(self.alpha[2]*trans + self.alpha[3]*(rot1+rot2))
            noisy_rot2 = rot2 - \
                self.sample(self.alpha[0]*rot2 + self.alpha[1]*trans)
            rot1, rot2, trans = noisy_rot1, noisy_rot2, noisy_trans

            # Calculate updated particle position based on random noise sample
            updated_particles[:, 0] = particles[:, 0] + \
                trans*np.cos(particles[:, 2] + rot1)
            updated_particles[:, 1] = particles[:, 1] + \
                trans*np.sin(particles[:, 2] + rot1)
            updated_particles[:, 2] = particles[:, 2] + rot1 + rot2
        return updated_particles

    def sample(self, var):
        """
        Sample from a normal distribution with mean 0 and variance var
        """
        return np.random.normal(0, np.sqrt(var), var.shape)

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
