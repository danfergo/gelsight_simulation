#!/usr/bin/env python

import rospy
import time
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import math
import numpy as np
# import rospack
# PKG_PATH = rospack.get_path('gelsight_gazebo')


bridge = CvBridge()
pub = None
rate = None
previous_position = (0, 0, 0)
# printer_speed = 0.004
printer_speed = 0.001

WS_MAX = (0.32, 0.32, 0.42)
# WS_MIN = (0, 0, 0.06)
WS_MIN = (0, 0, 0.01)
gelsight_img = None
gelsight_depth = None


def show_normalized_img(name, img):
    draw = img.copy()
    draw -= np.min(draw)
    draw = draw / np.max(draw)
    cv2.imshow(name, draw)
    return draw


def euclidean_dist(t1, t2):
    return math.sqrt(math.pow(t1[0] - t2[0], 2) + math.pow(t1[1] - t2[1], 2) + math.pow(t1[2] - t2[2], 2))


def gelsight_callback(img_msg):
    global gelsight_img

    camera_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
    gelsight_img = camera_img
    cv2.imshow('tactile_img', camera_img)
    cv2.waitKey(1)


def depth_callback(depth_msg):
    global gelsight_depth

    img = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="32FC1")
    img[np.isnan(img)] = np.inf
    gelsight_depth = img
    show_normalized_img('depth_map', gelsight_depth)
    cv2.waitKey(1)


def move(x, y, z, force=False, wait=None):
    if rospy.is_shutdown():
        exit(2)

    if not force and (
            x > WS_MAX[0] or y > WS_MAX[1] or z > WS_MAX[2] or x < WS_MIN[0] or y < WS_MIN[1] or z < WS_MIN[2]):
        print('ERROR. Attempted to move to invalid position.', (x, y, z))
        exit(2)

    global previous_position
    print('Move to:', (x, y, z))
    pos = Float64MultiArray()
    pos.data = [x, y, z]
    pub.publish(pos)

    if wait is None:
        s = 3 + euclidean_dist(previous_position, (x, y, z)) / printer_speed
    else:
        s = wait
    print('Waiting seconds: ', s)
    time.sleep(s)
    previous_position = (x, y, z)


def collect_data():
    global previous_position

    x_steps = 3
    y_steps = 3
    z_steps = 10
    h_step_size = 0.001
    z_step_size = 0.0001

    # SIM
    start_x = 0.185 - 0.045
    start_y = 0.165 + 0.007
    start_z = 0.022 + 0.027

    previous_position = (0.0, 0.0, 0.0)

    # REAL
    # start_x = 0.1625
    # start_y = 0.16
    # # start_z = 0.0641
    # start_z = 0.065

    starting_position = previous_position
    # starting_position = (0.1, 0.1, start_z + 0.01)
    # previous_position = starting_position
    # previous_position = (start_x, start_y, start_z + 0.01)
    print('START.')
    move(0, 0, start_z, force=True, wait=20)

    # TEST
    # move(start_x, start_y, start_z + 2*z_step_size)
    # move(start_x, start_y, start_z + 1*z_step_size)
    # move(start_x, start_y, start_z)
    # move(start_x, start_y, start_z - 1 * z_step_size)
    move(start_x, start_y, start_z + 3 * z_step_size, wait=20)  # 20 for sim, 60 for real

    # return
    #
    # move(*starting_position)

    k = 0
    BASE = '/home/danfergo/Projects/PhD/gelsight_simulation/dataset/sim/exp3'
    BASE_DEPTH = '/home/danfergo/Projects/PhD/gelsight_simulation/dataset/sim/depth3'
    # BASE = '/home/danfergo/Projects/gelsight_simulation/dataset/demo/sim_test'
    solid = 'wave1'
    # solid = 'dots'
    # solid = 'cross_lines'
    # solid = 'flat_slab'
    # solid = 'curved_surface'
    # solid = 'parallel_lines'
    # solid = 'pacman'
    # solid = 'torus'
    # solid = 'cylinder_shell'
    # solid = 'sphere2'
    # solid = 'line'
    # solid = 'cylinder_side'
    # solid = 'moon'
    # solid = 'random'
    # solid = 'prism'
    # solid = 'dot_in'
    # solid = 'triangle'
    # solid = 'sphere'
    # solid = 'hexagon'
    # solid = 'cylinder'
    # solid = 'cone'

    for x in range(-(x_steps // 2), (x_steps // 2) + 1):
        for y in range(-(y_steps // 2), (y_steps // 2) + 1):
            p = start_x + x * h_step_size, start_y + y * h_step_size
            move(*(p + (start_z + z_step_size,)))
            print('========================================>')
            for z in range(z_steps + 1):
                print('--START ---->')
                print('POINT:', (x, y, z))

                pp = p + (start_z - z * z_step_size,)
                move(*pp)

                k += 1
                if gelsight_img is not None:
                    cv2.imwrite(
                        BASE + '/' + solid + '__' + str(k) + '__' + str(x) + '_' + str(y) + '_' + str(z) + '.png',
                        gelsight_img)
                    print('...>', np.max(gelsight_depth), np.min(gelsight_depth), gelsight_depth.dtype,
                          np.shape(gelsight_depth), np.shape(gelsight_depth))
                    # cv2.imwrite(
                    #     BASE_DEPTH + '/' + solid + '__' + str(k) + '__' + str(x) + '_' + str(y) + '_' + str(z) + '.bmp',
                    #     gelsight_depth)
                    np.save(BASE_DEPTH + '/' + solid + '__' + str(k) + '__' + str(x) + '_' + str(y) + '_' + str(z) + '.npy', gelsight_depth)



                else:
                    print('warn. tactile img not received')

                print('--END   ---->')
                time.sleep(1)

            print('========================================>')
            move(*(p + (start_z + z_step_size,)))

    # move(*starting_position)


if __name__ == '__main__':
    rospy.init_node('gelsight_simulation_dc')

    pub = rospy.Publisher('/fdm_printer/xyz_controller/command', Float64MultiArray, queue_size=10)
    rospy.Subscriber("/gelsight/tactile_image", Image, gelsight_callback)
    rospy.Subscriber("/gelsight/depth/image_raw", Image, depth_callback)

    rate = rospy.Rate(1)
    # rate.sleep()
    # rate.sleep()
    # rate.sleep()
    rate.sleep()
    rate.sleep()
    print('--------------------_>>> Data Collection start.')
    collect_data()
