import rospy


class TransformPublisher:
    """ 
    Publishes transform between map and base link frames
    """

    def __init__(self):
        """
        Initializes the publisher
        """
        self.publisher = rospy.Publisher("/map_to_base_link",
                                         TransformStamped,
                                         queue_size=10)
        self.rate = rospy.Rate(10)

    def publish_transform(self):
        """
        Publishes the transform to the publisher
        """
        transform = TransformStamped()
        transform.header.frame_id = "map"
        transform.child_frame_id = "base_link"
        transform.transform.translation.x = 0.0
        transform.transform.translation.y = 0.0
        transform.transform.translation.z = 0.0
        transform.transform.rotation.x = 0.0
        transform.transform.rotation.y = 0.0
        transform.transform.rotation.z = 0.0
        transform.transform.rotation.w = 1.0
        self.publisher.publish(transform)
        self.rate.sleep()


if __name__ == '__main__':
    rospy.init_node('transform_publisher')
    transform_publisher = TransformPublisher()
    rospy.spin()
