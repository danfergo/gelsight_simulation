#!/usr/bin/env python
import rospy

import numpy as np
import cv2
from cv_bridge import CvBridge

from sensor_msgs.msg import Image


def main_loop():
    # rate = rospy.Rate(30)
    cap = cv2.VideoCapture(0)

    bridge = CvBridge()

    while not rospy.is_shutdown() \
            and cap.isOpened():
        ret, frame = cap.read()

        # cv2.imshow('frame', frame)

        image_message = bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        pub.publish(image_message)
        cv2.waitKey(1)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
        # rate.sleep()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    pub = rospy.Publisher('/gelsight/tactile_image', Image, queue_size=10)
    rospy.init_node('gelsight_driver', anonymous=True)
    print('DRIVER RUNNING')
    main_loop()
