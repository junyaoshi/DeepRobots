import csv
from math import sin, cos
from Robots.WheeledRobot_v1 import ThreeLinkRobot
from Robots.ContinuousSwimmingBot import SwimmingRobot
from math import pi


def generate_csv(robot_params, filename):
    with open(filename, 'w') as file:
        w = csv.writer(file)
        w.writerows(robot_params)


if __name__ == "__main__":

    robot_params = []
    # robot = ThreeLinkRobot(a1=-0.01, a2=0.01, t_interval=0.02)
    robot = SwimmingRobot(t_interval=1, a1=0, a2=0)
    robot_param = [robot.x, robot.y, robot.theta, float(robot.a1), float(robot.a2), robot.a1dot, robot.a2dot]
    robot_params.append(robot_param)
    # for i in range(50):
    #     print('i: ', i)
    #     if i%2 == 0:
    #         action = (-pi/2, pi/2)
    #     else:
    #         action = (pi/2, -pi/2)
    #     for j in range(40):
    #         print('j: ', j)
    #         print('a1 a2: ', robot.a1, robot.a2)
    #         robot.move(action)
    #         robot_param = [robot.x, robot.y, robot.theta, float(robot.a1), float(robot.a2), robot.a1dot, robot.a2dot]
    #         robot_params.append(robot_param)

    for t in range(1000):
        print(t + 1, 'th iteration')
        a1dot = 0
        a2dot = 1/5 * cos(t/5)
        action = (a1dot, a2dot)
        robot.move(action)
        print('action taken(a1dot, a2dot): ', action)
        print('robot x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
        robot_param = [robot.x, robot.y, robot.theta, float(robot.a1), float(robot.a2), robot.a1dot, robot.a2dot]
        robot_params.append(robot_param)

    print(robot_params)
    generate_csv(robot_params, 'csv_outputs/swimming_test_a1_idle.csv')

