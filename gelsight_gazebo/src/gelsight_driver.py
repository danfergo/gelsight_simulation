#!/usr/bin/env python
import rospy
import rospkg

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

bridge = CvBridge()
rospack = rospkg.RosPack()
PKG_PATH = rospack.get_path('gelsight_gazebo')

import cv2
import numpy as np

import scipy.ndimage.filters as fi

""" 
    Utils section
"""


def show_normalized_img(name, img):
    draw = img.copy()
    draw -= np.min(draw)
    draw = draw / np.max(draw)
    cv2.imshow(name, draw)


def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen // 2, kernlen // 2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)


def derivative(mat, direction):
    assert (direction == 'x' or direction == 'y'), "The derivative direction must be 'x' or 'y'"
    kernel = None
    if direction == 'x':
        kernel = [[-1.0, 0.0, 1.0]]
    elif direction == 'y':
        kernel = [[-1.0], [0.0], [1.0]]
    kernel = np.array(kernel, dtype=np.float64)
    return cv2.filter2D(mat, -1, kernel) / 2.0


def tangent(mat):
    dx = derivative(mat, 'x')
    dy = derivative(mat, 'y')
    img_shape = np.shape(mat)
    _1 = np.repeat([1.0], img_shape[0] * img_shape[1]).reshape(img_shape).astype(dx.dtype)
    unormalized = cv2.merge((-dx, -dy, _1))
    norms = np.linalg.norm(unormalized, axis=2)
    return (unormalized / np.repeat(norms[:, :, np.newaxis], 3, axis=2))


def solid_color_img(color, size):
    image = np.zeros(size + (3,), np.float64)
    image[:] = color
    return image


def add_overlay(rgb, alpha, color):
    s = np.shape(alpha)

    opacity3 = np.repeat(alpha, 3).reshape((s[0], s[1], 3))  # * 10.0

    overlay = solid_color_img(color, s)

    foreground = opacity3 * overlay
    background = (1.0 - opacity3) * rgb.astype(np.float64)
    res = background + foreground

    res[res > 255.0] = 255.0
    res[res < 0.0] = 0.0
    res = res.astype(np.uint8)

    return res


""" 
    GelSight Simulation
"""


class SimulationApproach:

    def __init__(self, **config):
        self.light_sources = config['light_sources']
        self.background = config['background_img']
        self.px2m_ratio = config['px2m_ratio']
        self.elastomer_thickness = config['elastomer_thickness']
        self.min_depth = config['min_depth']

        self.default_ks = 0.15
        self.default_kd = 0.5

        self.max_depth = self.min_depth + self.elastomer_thickness

    def protrusion_map(self, original, not_in_touch):
        protrusion_map = np.copy(original)
        protrusion_map[not_in_touch >= self.max_depth] = self.max_depth
        return protrusion_map

    def segments(self, depth_map):
        not_in_touch = np.copy(depth_map)
        not_in_touch[not_in_touch < self.max_depth] = 0.0
        not_in_touch[not_in_touch >= self.max_depth] = 1.0

        in_touch = 1 - not_in_touch

        return not_in_touch, in_touch

    def internal_shadow(self, elastomer_depth, in_touch):
        elastomer_depth_inv = self.max_depth - elastomer_depth
        elastomer_depth_inv[elastomer_depth_inv < 0] = 0.0
        elastomer_depth_inv = np.interp(elastomer_depth_inv, (self.min_depth, self.max_depth), (0.0, 1.0))
        return 5 * in_touch * (0.1 + elastomer_depth_inv)

    def apply_elastic_deformation(self, protrusion_depth, not_in_touch, in_touch):
        kernel = gkern2(15, 7)
        blur = self.max_depth - protrusion_depth

        for i in range(5):
            blur = cv2.filter2D(blur, -1, kernel)
        return 3 * -blur * not_in_touch + (protrusion_depth * in_touch)

    def phong_illumination(self, T, source_dir, kd, ks):
        dot = np.dot(T, np.array(source_dir)).astype(np.float64)
        difuse_l = dot * kd
        difuse_l[difuse_l < 0] = 0.0

        dot3 = np.repeat(dot[:, :, np.newaxis], 3, axis=2)

        R = 2.0 * dot3 * T - source_dir
        V = [0.0, 0.0, 1.0]

        spec_l = np.power(np.dot(R, V), 5) * ks

        return difuse_l + spec_l

    def generate(self, obj_depth):
        not_in_touch, in_touch = self.segments(obj_depth)
        protrusion_depth = self.protrusion_map(obj_depth, not_in_touch)
        elastomer_depth = self.apply_elastic_deformation(protrusion_depth, not_in_touch, in_touch)
        # show_normalized_img('elastomer depth', elastomer_depth)

        out = self.background
        out = add_overlay(out, self.internal_shadow(protrusion_depth, in_touch), (0.0, 0.0, 0.0))

        T = tangent(elastomer_depth / self.px2m_ratio)
        # show_normalized_img('tangent', T)
        for light in self.light_sources:
            ks = light['ks'] if 'ks' in light else self.default_ks
            kd = light['ks'] if 'ks' in light else self.default_kd
            out = add_overlay(out, self.phong_illumination(T, light['position'], kd, ks), light['color'])

        cv2.imshow('tactile img', out)

        return out


""" 
    ROS Driver 
"""


class GelSightDriver:

    def __init__(self, name, sim_approach):
        self.simulation_approach = sim_approach
        self.depth_img = None
        self.visual_img = None

        rospy.init_node(name, )
        rospy.Subscriber("/gelsight/depth/image_raw", Image, self.on_depth_img)
        rospy.Subscriber("/gelsight/image/image_raw", Image, self.on_rgb_img)
        self.publisher = rospy.Publisher("/gelsight/tactile_image", Image, queue_size=1)

        self.rate = rospy.Rate(30)

    def on_depth_img(self, img_msg):
        img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="32FC1")
        img[np.isnan(img)] = np.inf
        self.depth_img = img

    def on_rgb_img(self, img_msg):
        self.visual_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

    def publish(self, tactile_img):
        self.publisher.publish(bridge.cv2_to_imgmsg(tactile_img, "bgr8"))

    def run(self):
        while not rospy.is_shutdown():
            if self.depth_img is None or self.visual_img is None:
                continue

            tactile_img = self.simulation_approach.generate(self.depth_img)
            self.publish(tactile_img)
            self.rate.sleep()
            cv2.waitKey(1)


def main():
    # light position: x,y,z, color BGR
    light_sources = [
        {'position': [0, 1, 0.25], 'color': (240, 240, 240)},
        {'position': [-1, 0, 0.25], 'color': (255, 139, 78)},
        {'position': [0, -1, 0.25], 'color': (108, 82, 255)},
        {'position': [1, 0, 0.25], 'color': (100, 240, 150)},
    ]
    background_img = cv2.imread(PKG_PATH + '/assets/background.png')
    px2m_ratio = 5.4347826087e-05
    elastomer_thickness = 0.004
    min_depth = 0.026  # distance from the image sensor to the rigid glass outer surface

    simulation_approach = SimulationApproach(
        light_sources=light_sources,
        background_img=background_img,
        px2m_ratio=px2m_ratio,
        elastomer_thickness=elastomer_thickness,
        min_depth=min_depth
    )

    driver = GelSightDriver('gelsight_node', simulation_approach)
    driver.run()


if __name__ == '__main__':
    main()
