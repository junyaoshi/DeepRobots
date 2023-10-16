"""
Chaplygin Beanie Model File
"""


from math import cos, sin, pi
import numpy as np
import random
from scipy.integrate import quad, odeint

# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
# from utils.csv_generator import generate_csv
import csv
import time


class ChaplyginBeanie(object):
	def __init__(self,
		x = 0.0,
		y = 0.0,
		theta = 0.0,
		xdot = 0.0,
		ydot = 0.0,
		thetadot = 0.0,
		phidot = 0.0,
		m = 1.0,
		a = 1.0,
		B = 1.0,
		C = 1.0,
		t_interval=0.25,
		timestep=1):

		self.init_x = x
		self.init_y = y
		self.init_theta = theta
		self.init_phidot = phidot
		self.init_JLT = m*xdot*cos(theta) + m*ydot*sin(theta)
		self.init_JRW = (m*a**2+B+C)*thetadot + B*phidot

		self.x = x
		self.y = y
		self.theta = theta
		self.phidot = phidot
		self.JLT = self.init_JLT
		self.JRW = self.init_JRW

		self.time = 0

		#constants
		self.t_interval = t_interval
		self.timestep = timestep
		self.m = m
		self.a = a
		self.B = B
		self.C = C

		self.state = (self.x, self.y, self.theta, self.JLT, self.JRW)


	def reset(self):
		self.x = self.init_x
		self.y = self.init_y
		self.theta = self.init_theta
		self.phidot = self.init_phidot
		self.JLT = self.init_JLT
		self.JRW = self.init_JRW
		self.state = (self.x, self.y, self.theta, self.JLT, self.JRW)

		self.time = 0
		return self.state

	def reset_x(self):
		self.x = self.init_x
		return self.x

	def reset_y(self):
		self.y = self.init_y
		return self.y

	def set_state(self, x, y, theta, JLT, JRW):
		self.x, self.y, self.theta, self.JLT, self.JRW = x, y, theta, JLT, JRW
		self.state = (x, y, theta, JLT, JRW)
		return self.state

	def get_position(self):
		return self.x, self.y


	def robot(self, state, t, phiddot):
		x, y, theta, JLT, JRW = state
		thetadot = (JRW-self.B*phiddot)/(self.m*self.a**2+self.B+self.C)
		xdot = -(-self.a*JLT*cos(theta) + JRW*sin(theta) - self.B*sin(theta)*phiddot - (self.B+self.C)*sin(theta)*thetadot) / (self.a*self.m)
		ydot = -(-self.a*JLT*sin(theta) - JRW*cos(theta) + self.B*cos(theta)*phiddot + (self.B+self.C)*cos(theta)*thetadot) / (self.a*self.m)
		JLTdot = self.m*self.a*thetadot**2
		JRWdot = -self.a*thetadot*JLT
		dynamics = [xdot, ydot, thetadot, JLTdot, JRWdot]
		return dynamics

	def perform_integration(self, action, t_interval):
		if(t_interval == 0):
			return self.x, self.y, self.theta, self.JLT, self.JRW
		state = [self.x, self.y, self.theta, self.JLT, self.JRW]
		t = np.linspace(0, t_interval, 11)
		sol = odeint(self.robot, state, t, args=(action,))
		x, y, theta, JLT, JRW = sol[-1]
		return x, y, theta, JLT, JRW


	def enforce_angle_range(self, angle_name):
		if angle_name == 'theta':
			angle = self.theta
		elif angle_name == 'phi':
			angle = self.phi
		if angle > pi:
			angle = angle % (2 * pi)
			if angle > pi:
				angle = angle - 2 * pi
		elif angle < -pi:
			angle = angle % (-2 * pi)
			if angle < -pi:
				angle = angle + 2 * pi
		if angle_name == 'theta':
			self.theta = angle
		elif angle_name == 'phi':
			self.phi = angle

	def update_position(self, x, y, theta):
		self.x = x
		self.y = y
		self.theta = theta
		self.enforce_angle_range('theta')

	def update_momentum(self, phidot, JLT, JRW):
		self.phidot = phidot
		self.JLT = JLT
		self.JRW = JRW


	def move(self, action, timestep = 1):
		self.timestep = timestep
		t = self.timestep * self.t_interval
		phidot = self.phidot + action * t
		x, y, theta, JLT, JRW = self.perform_integration(action, t)

		self.update_position(x, y, theta)
		self.update_momentum(phidot, JLT, JRW)
		self.state = (self.JLT, self.JRW)


if __name__ == "__main__":
	# create a robot simulation
	robot = ChaplyginBeanie(t_interval=1)
	
	x_pos = [robot.x]
	y_pos = [robot.y]
	thetas = [robot.theta]
	JLT = [robot.JLT]
	JRW = [robot.JRW]

	times = [0]
	robot_params = []
	print('initial x y theta: ', robot.x, robot.y, robot.theta)

	for t in range(100):
		print(t + 1, 'th iteration')
		phidot = 1 / 3 * cos(t / 5)
		action = phidot
		start = time.time()
		robot.move(action)
		end = time.time()
		print("Move time: {}".format(end - start))
		print('action taken(phidot): ', action)
		print('robot x y theta: ', robot.x, robot.y, robot.theta)

		x_pos.append(robot.x)
		y_pos.append(robot.y)
		thetas.append(robot.theta)
		JLT.append(robot.JLT)
		JRW.append(robot.JRW)
		times.append(t + 1)

		robot_param = [robot.x,
					   robot.y,
					   robot.theta,
					   robot.JLT,
					   robot.JRW]
		robot_params.append(robot_param)

	# generate_csv(robot_params, "../results/wheelchair_results/" + "result.csv")
	# with open("out.csv", "w", newline="") as f:
	#     writer = csv.writer(f)
	#     writer.writerows(robot_params)

	# view results
	# print('x positions are: ' + str(x_pos))
	# print('y positions are: ' + str(y_pos))
	# print('thetas are: ' + str(thetas))

	plt.plot(x_pos, y_pos)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.title('y vs x')
	plt.show()

	plt.plot(times, x_pos)
	plt.ylabel('x positions')
	plt.show()

	plt.plot(times, y_pos)
	plt.ylabel('y positions')
	plt.show()

	plt.plot(times, thetas)
	plt.ylabel('theta')
	plt.show()
	plt.close()

	plt.plot(times, JLT)
	plt.ylabel('JLT')
	plt.show()
	plt.close()

	plt.plot(times, JRW)
	plt.ylabel('JRW')
	plt.show()
	plt.close()