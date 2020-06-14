#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
import serial
import threading


def gcode_parse_pos(res):
    arr = res.split(' ')
    pos_d = {a.split(':')[0]: float(a.split(':')[1]) for a in arr}
    return pos_d['X'], pos_d['Y'], pos_d['Z']


class FDMPrinterDriver:

    def __init__(self):
        rospy.init_node('fdm_printer', anonymous=True)
        rospy.Subscriber("move", Point, self.move_to)

        self.ser_con = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)

        # time.sleep(5)

        self.MAX_X = 320
        self.MAX_Y = 320
        self.MAX_Z = 420
        self.initialized = False
        self.position = None

        self.continuously_empty = 0
        self.wait_to_be_ready = 5

        self.access_serial = threading.Lock()
        self.move_printer = threading.Lock()

        self.wait_until_ready()

        # initialize
        self.initialized = True
        self.on_init()

        # watch position
        self.watch_position()

    def on_init(self):
        print('GO HOME.')
        self.go_home()

    def wait_until_ready(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            ln = self.ser_con.readline().rstrip()
            self.continuously_empty = self.continuously_empty + 1 if ln == '' else 0

            if not self.initialized and self.continuously_empty == self.wait_to_be_ready:
                break
            r.sleep()

    def watch_position(self):
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            # self.position = gcode_parse_pos(self.send_cmd('M114'))
            # print('self pos', self.position)
            r.sleep()

    def move_center(self):
        pt = Point()
        pt.x = 0
        pt.y = 0
        pt.z = 0
        self.move_to(pt)

        # center_pt = Point(self.MAX_X / 2, 0, 160)
        # self.move_to(center_pt)

    def send_cmd(self, cmd):
        self.access_serial.acquire()

        self.ser_con.write(str.encode(cmd + ' \n'))
        self.ser_con.flush()
        rospy.loginfo("G-code sent: " + cmd)

        res = None
        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            ln = self.ser_con.readline().rstrip()
            if ln == 'ok':
                self.access_serial.release()
                return res
            res = ln
            r.sleep()

    def move_to(self, position):
        if not self.initialized:
            return

        # self.move_printer.acquire()

        x, y, z = position.x, position.y, position.z
        x = max(0, min(x, self.MAX_X))
        y = max(0, min(y, self.MAX_Y))
        z = max(0, min(z, self.MAX_Z))
        print(x, y, z)
        self.send_cmd("G0 X%.2f Y%.2f Z%.2f" % (x, y, z))

        r = rospy.Rate(10)
        while not rospy.is_shutdown():
            print(self.position, (x, y, z))
            if self.position == (x, y, z):
                print('TRUE!!!')
                # self.move_printer.release()
                break
            self.position = gcode_parse_pos(self.send_cmd('M114'))
            print('->>>>---', self.position)
            r.sleep()

    def go_home(self):
        self.send_cmd("G28")


if __name__ == '__main__':
    driver = FDMPrinterDriver()
