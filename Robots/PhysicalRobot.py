import sys

# Edit the system path as needed
sys.path.append('/home/pi/DeepRobots/Robots')

import math
import serial
from time import sleep
from adafruit_servokit import ServoKit
import time
from math import cos, sin, pi
import numpy as np
import random
# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


class PhysicalRobot(object):

    def __init__(self, a1=90, a2=90, a_upper=150, a_lower=30, delay=0.015):
        """
        :param a1: initial joint angle of proximal link
        :param a2: initial joint angle of distal link
        :param a_upper: upper bound of joint angles
        :param a_lower: lower bound of joint angles
        :param t_interval: time interval between each action
        """
        self.encoder_val = 0 # sum of the displacements detected by the two encoders
        self.a1 = a1
        self.a2 = a2
        self.a_upper = a_upper
        self.a_lower = a_lower
        self.a1dot = 0
        self.a2dot = 0
        self.state = (self.a1, self.a2)

        # arduino and servo controller
        self.kit = ServoKit(channels=16)
        self.arduino = serial.Serial('/dev/ttyACM0', 9600)

        # constants
        self.delay = delay
    
    def get_encoder(self):
        self.arduino.write(str.encode('LE'))
        left_encoder = int(self.arduino.readline())
        self.arduino.write(str.encode('RE'))
        right_encoder = int(self.arduino.readline())
        return left_encoder, right_encoder

    def move(self, action):
        a1dot, a2dot = action
        # print("action: ", action)
        a1, a2 = self.a1, self.a2
        a1_target = a1 + int(a1dot)
        a2_target = a2 + int(a2dot)
        
        # enforce joint limits
        if a1_target > self.a_upper or a1_target < self.a_lower:
            a1_target = a1
        if a2_target > self.a_upper or a2_target < self.a_lower:
            a2_target = a2
            
        # ensure that current state is initial position
        self.kit.servo[1].angle = a1 
        self.kit.servo[2].angle = a2
        
        while a1 != a1_target or a2 != a2_target: 
            if a1 < a1_target:
                a1 += 1
            elif a1 > a1_target:
                a1 -= 1
                
            if a2 < a2_target:
                a2 += 1
            elif a2 > a2_target:
                a2 -= 1
                
            # move joints accordingly
            self.kit.servo[1].angle = a1
            self.kit.servo[2].angle = a2
            # print(a1,a2, a1_target, a2_target)
            sleep(self.delay) # time delay of 0.015 or 0.025 seems to work best
      
        assert a1 == a1_target and a2 == a2_target, "Problem with moving joint angles to target positions"
        left_encoder,right_encoder = self.get_encoder() 
        encoder_val = left_encoder + right_encoder
        # print('reward: {}'.format(encoder_val))
        self.update_params(a1, a2, a1dot, a2dot, encoder_val)
        return self.encoder_val, self.a1, self.a2

    def reset_state(self, state):
        a1_target, a2_target = state
        a1dot = a1_target - self.a1
        a2dot = a2_target - self.a2
        action = (a1dot, a2dot)
        self.move(action)
        return self.state
    
    def update_params(self, a1, a2, a1dot, a2dot, encoder_val):
        self.a1 = a1
        self.a2 = a2
        self.state = (self.a1, self.a2)
        self.encoder_val = encoder_val

    def print_state(self):
        print('\nthe current state is: ' + str(self.state) + '\n')


if __name__ == "__main__":
    robot = PhysicalRobot()

    time = [0]
    a1 = [robot.a1]
    a2 = [robot.a2]
    displacement = [robot.encoder_val]
    print('initial a1: {} a2: {}'.format(robot.a1, robot.a2))
    for t in range(6):
        print('Executing iteration number {}'.format(t+1))
        if t % 2 == 0:
            a1dot = 60
            a2dot = -60
        else:
            a1dot = -60
            a2dot = 60
        action = [a1dot, a2dot]
        robot.move(action)
        print('action taken: {}'.format(action))
        print('robot a1: {} a2: {}'.format(robot.a1, robot.a2))
        print('robot encoder displacement: {}'.format(robot.encoder_val))
        time.append(t + 1)
        a1.append(robot.a1)
        a2.append(robot.a2)
        displacement.append(robot.encoder_val)

    plt.plot(time, a1)
    plt.ylabel('a1 displacements')
    plt.show()

    plt.plot(time, a2)
    plt.ylabel('a2 displacements')
    plt.show()

    plt.plot(time, displacement)
    plt.ylabel('encoder displacements')
    plt.show()



