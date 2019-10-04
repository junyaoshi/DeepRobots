import math
import serial
import time
from math import cos, sin, pi
import numpy as np
import random
# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


class PhysicalRobot(object):

    def __init__(self, a1=0, a2=0, t_interval=125.0):
        """
        :param a1: initial joint angle of proximal link
        :param a2: initial joint angle of distal link
        :param t_interval: time interval between each action
        """

        # self.theta_displacement = 0
        self.encoder_displacement = 0 # sum of the displacements detected by the two encoders
        self.a1 = a1
        self.a2 = a2
        self.a1dot = 0
        self.a2dot = 0
        self.state = (self.a1, self.a2)

        # arduino controler
        self.baudrate = 19200
        self.arduino = serial.Serial('com11', self.baudrate)

        # constants
        self.t_interval = t_interval

    # accessor methods
    def get_state(self):
        return self.state
    '''
    # Transition Function Example For Physical Bot - Accepts action tuple (a1dot,a2dot) and fixed time interval t_interval 
    # - returns joint angles and simple_reward using displacment from encoder values
    def move(self, action):
        a1dot, a2dot = action

        # SEND RATES OF CHANGE AND CONSTANT TIME INTERVAL = "Execution Time" to ARDUINO IDE via Serialport

        self.arduino.write(chr(a1dot))  # send action
        self.arduino.write(chr(a2dot))  # send action
        time.sleep(self.t_interval)  # wait for action to finish then get new servo positioins

        # GET SERVO ANGLES from Arduino IDE
        self.arduino.write('pos1')  # Request servo position(s)
        a1 = round(float(self.arduino.readline()), 2)  # Read serial port and save data
        self.arduino.write('pos2')
        a2 = round(float(self.arduino.readline()), 2)

        # GET ENCODER VALUES from Arduino IDE
        self.arduino.write('LE')  # Request Encoder value(s), NB they are returned here as the delta, not cumulative
        Left_Encoder_Count = int(self.arduino.readline())
        self.arduino.write('RE')
        Right_Encoder_Count = int(self.arduino.readline())

        # CALCULATE REWARD PARAMETER(S) FROM ENCODER VALUES
        displacement = Left_Encoder_Count + Right_Encoder_Count  # Example of "simple reward" based on enocder values

        self.update_alpha_dots(a2dot, a2dot)
        self.update_params(a1, a2, displacement)

        return self.state, self.encoder_displacement  # return new_state and displacment/reward
    '''

    # This funtion accepts "actions" as a tuple containing 3 variables (a1dot,a2dot,t_interval)
    # can update which variables you want sent to the function
    # make sure that a1dot and a2dot values range between (0-92-180)
    # = (Full Speed CW - No Speed - Full Speed CCW), and any integers inbetween
    # time interval range should stay between 125ms - 250ms
    def move(self, action):
        a1dot, a2dot = action
        action_command = str(a1dot) + " " + str(a2dot) + " " + str(self.t_interval)
        self.arduino.write(("start"))
        validate = self.arduino.readline()
        if len(validate) == len("ready") + 2:
            self.arduino.write(action_command)
            while True:
                validate = self.arduino.readline()
                if len(validate) == len("done") + 2:
                    a1 = self.arduino.readline()
                    a2 = self.arduino.readline()
                    LE = self.arduino.readline()
                    RE = self.arduino.readline()

                    # calculate total encoder displacement
                    displacement = LE + RE  # Example of "simple reward" based on enocder values

                    self.update_alpha_dots(a1dot, a2dot)
                    self.update_params(a1, a2, displacement)

                    return self.encoder_displacement, self.a1, self.a2

    # update the return portion however you would like currently return(servo1 angle, servo2 angle, Left Encoder count, Right encoder count)

    def update_alpha_dots(self, a1dot, a2dot):
        self.a1dot = a1dot
        self.a2dot = a2dot

    def update_params(self, a1, a2, displacement):
        self.a1 = a1
        self.a2 = a2
        self.state = (self.a1, self.a2)
        self.encoder_displacement = displacement

    def print_state(self):
        print('\nthe current state is: ' + str(self.state) + '\n')


if __name__ == "__main__":
    robot = PhysicalRobot(t_interval=125.0)

    time = [0]
    a1 = [robot.a1]
    a2 = [robot.a2]
    displacement = [robot.encoder_displacement]
    print('initial a1: {} a2: {}'.format(robot.a1, robot.a2))
    for t in range(100):
        print('Executing iteration number {}',format(t+1))
        a1dot = 1/3 * cos(t/5)
        a2dot = -1/3 * sin(t/5)
        action = [a1dot, a2dot]
        robot.move(action)
        print('action taken: {}'.format(action))
        print('robot a1: {} a2: {}'.format(robot.a1, robot.a2))
        print('robot encoder displacement: {}'.format(robot.encoder_displacement))
        time.append(t + 1)
        a1.append(robot.a1)
        a2.append(robot.a2)
        displacement.append(robot.encoder_displacement)

    plt.plot(time, a1)
    plt.ylabel('a1 displacements')
    plt.show()

    plt.plot(time, a2)
    plt.ylabel('a2 displacements')
    plt.show()

    plt.plot(time, displacement)
    plt.ylabel('encoder displacements')
    plt.show()



