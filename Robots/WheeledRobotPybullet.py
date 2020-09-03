import pybullet as p
import pybullet_data
import numpy as np
import time
import matplotlib.pyplot as plt

import Snakebot_PyBullet.pybullet_functions as pf


class WheeledRobotPybullet(object):

    def __init__(self,
                 use_GUI=False,             # False
                 gravity= -9.81,            # -9.81
                 tracking_cam=True,         # True
                 timestep= 1. / 60,         # 1./240
                 decision_interval=.25,     # 1.0,
                 center_link_mass=None,     # None
                 outer_link_mass=None,      # None
                 wheel_mass=None,           # None
                 rolling_friction=None,     # None
                 lateral_fricition= None,   # None
                 spinning_fricition=None,   # None
                 friction_anchor= None,     # 1
                 wheel_restitution=None,    # 0
                 a_upper= 7*np.pi/18,       # np.pi/2 # limits for robot 7*np.pi/18
                 a_lower=-7*np.pi/18):      # -np.pi/2 # -7*np.pi/18

        self.tracking_cam = tracking_cam
        self.timestep = timestep
        self.decision_interval = decision_interval
        self.a_upper = a_upper
        self.a_lower = a_lower

        # initalize pybullet simulation enviroment,
        if use_GUI:
            p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            p.connect(p.DIRECT)
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load "floor", otherwise unfixed based will "fall through plane"
        p.loadURDF("plane.urdf")
        p.createCollisionShape(p.GEOM_PLANE)

        # Define starting position and orientation for robot
        StartingOrientation = p.getQuaternionFromEuler([0, 0, 0])
        StartingPosition = [1, 1, .03]  # x,y,z

        # Load robot URDF file created in solidworks
        self.robot = p.loadURDF(r'C:\Users\Jesse\Desktop\DeepRobots\Snakebot_PyBullet\Snakebot_urdf.SLDASM\Snakebot_urdf.SLDASM\urdf\Snakebot_urdf.SLDASM.urdf',
                                StartingPosition,
                                StartingOrientation,
                                useFixedBase=0)
        pf.color_code_joints(self.robot)

        # Modify default robot model parameters
        if center_link_mass is not None:
            p.changeDynamics(self.robot, linkIndex=-1, mass=center_link_mass)
        if outer_link_mass is not None:
            p.changeDynamics(self.robot, linkIndex=0, mass=outer_link_mass)
            p.changeDynamics(self.robot, linkIndex=3, mass=outer_link_mass)
        if wheel_mass is not None:
            pf.add_wheels_mass(self.robot, wheel_mass=wheel_mass)
        if rolling_friction is not None:
            pf.add_rolling_friction(self.robot, friction=rolling_friction)
        if lateral_fricition is not None:
            pf.add_lateral_friction(self.robot, friction=lateral_fricition)
        if spinning_fricition is not None:
            pf.add_spinning_friction(self.robot, friction=spinning_fricition)
        if friction_anchor is not None:
            pf.add_friction_anchor(self.robot, friction_anchor=friction_anchor)
        if wheel_restitution is not None:
            pf.add_wheel_restitution(self.robot, wheel_restitution=wheel_restitution)

        # Initalize Physics Engine parameters (always after modifiying robot parameters)
        p.setGravity(0, 0, gravity)
        p.setTimeStep(timestep)
        p.setRealTimeSimulation(1)  # 1 = True, 0 = False
        p.setPhysicsEngineParameter(fixedTimeStep=timestep,
                                    numSolverIterations=1,
                                    useSplitImpulse=1,
                                    numSubSteps=1,
                                    enableConeFriction=1,
                                    collisionFilterMode=0)

        # Get current state of robot
        self.update_system_params()
        self._set_init_system_params()
        self.theta_displacement = 0  # for theta reward functions
        self.a1dot = 0
        self.a2dot = 0
        self.state = (self.theta, self.a1, self.a2)

    def update_system_params(self):
        self.position, self.orientation = p.getBasePositionAndOrientation(self.robot)
        self.x, self.y, _ = self.position
        self.theta, *_ = pf.quat_to_euler(self.orientation)
        self.a1, self.a2 = self.get_joint_angles()

    def _set_init_system_params(self):
        self.init_position = self.position
        self.init_orientation = self.orientation
        self.init_a1 = self.a1
        self.init_a2 = self.a2

    def get_joint_angles(self):
        a1, *_ = p.getJointState(self.robot, 0)
        a2, *_ = p.getJointState(self.robot, 3)
        return a1, a2

    def _set_tracking_cam(self):
        position, orientation = p.getBasePositionAndOrientation(self.robot)
        p.resetDebugVisualizerCamera(cameraDistance=.1, cameraYaw=0, cameraPitch=-95.0,
                                     cameraTargetPosition=[position[0], position[1], position[2] + 1])

    def move(self, action):

        a1dot, a2dot = action
        a1, a2 = self.get_joint_angles()
        theta_prev = self.theta

        # execute action
        for i in range(int(self.decision_interval / self.timestep)):  # calculate # of iterations for decision interval
            if self.tracking_cam:
                self._set_tracking_cam()

            # Updated Check Boundries / Overide action if violates joint limits and hold at current limit position
            if a1 >= self.a_upper and a1dot >= 0 or a1 <= self.a_lower and a1dot <= 0:
                a1dot = 0
                p.setJointMotorControl2(self.robot, 0, p.POSITION_CONTROL,targetPosition = a1)

            if a2 >= self.a_upper and a2dot >= 0 or a2 <= self.a_lower and a2dot <= 0:
                a2dot = 0
                p.setJointMotorControl2(self.robot, 3, p.POSITION_CONTROL,targetPosition = a2)


            p.setJointMotorControl2(self.robot, 0, p.VELOCITY_CONTROL,targetVelocity=a1dot)
            p.setJointMotorControl2(self.robot, 3, p.VELOCITY_CONTROL,targetVelocity=a2dot)

            p.stepSimulation()
            #time.sleep(self.timestep) # uncomment for real time speed view

            # Update at each sim step
            a1, a2 = self.get_joint_angles()


        # update class params
        self.update_system_params()
        self.theta_displacement = self.theta - theta_prev
        self.a1dot = a1dot # 0
        self.a2dot = a2dot # 0
        self.state = (self.theta, self.a1, self.a2)

        return self.state

    def set_system_params(self, position, orientation, a1, a2):
        p.resetBasePositionAndOrientation(self.robot, position, orientation)
        p.setJointMotorControl2(self.robot, 0, p.POSITION_CONTROL, targetPosition=a1)
        p.setJointMotorControl2(self.robot, 3, p.POSITION_CONTROL, targetPosition=a2)
        p.stepSimulation()
        time.sleep(self.timestep)

        # update class params
        self.update_system_params()
        self.state = (self.theta, self.a1, self.a2)

    def reset(self):
        self.set_system_params(self.init_position, self.init_orientation, self.init_a1, self.init_a2)
        return self.state


if __name__ == "__main__":

    # create a robot simulation
    robot = WheeledRobotPybullet(decision_interval=0.2)

    x_pos = [robot.x]
    y_pos = [robot.y]
    thetas = [robot.theta]
    steps = [0]
    a1 = [robot.a1]
    a2 = [robot.a2]
    print('initial x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
    for t in range(100):
        print(t + 1, 'th iteration')
        a1dot = np.cos(t)
        a2dot = np.sin(t)
        action = (a1dot, a2dot)
        robot.move(action)
        print('action taken(a1dot, a2dot): ', action)
        print('robot x y theta a1 a2: ', robot.x, robot.y, robot.theta, robot.a1, robot.a2)
        x_pos.append(robot.x)
        y_pos.append(robot.y)
        thetas.append(robot.theta)
        steps.append(t + 1)
        a1.append(robot.a1)
        a2.append(robot.a2)
        robot_param = [robot.x,
                       robot.y,
                       robot.theta,
                       float(robot.a1),
                       float(robot.a2),
                       robot.a1dot,
                       robot.a2dot]

    plt.plot(x_pos, y_pos)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y vs x')
    plt.show()

    plt.plot(steps, a1)
    plt.ylabel('a1 displacements')
    plt.show()

    plt.plot(steps, a2)
    plt.ylabel('a2 displacements')
    plt.show()

    plt.plot(steps, x_pos)
    plt.ylabel('x positions')
    plt.show()

    plt.plot(steps, y_pos)
    plt.ylabel('y positions')
    plt.show()

    plt.plot(steps, thetas)
    plt.ylabel('thetas')
    plt.show()
    plt.close()
