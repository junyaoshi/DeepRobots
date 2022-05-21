"""
Wheelchair Model File
"""


from math import cos, sin, pi
import numpy as np
import random
from scipy.integrate import quad, odeint

# SET BACKEND
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
#from utils.csv_generator import generate_csv
import csv
import time


class WheelChairRobot(object):
	def __init__(self,
		x = 0.0,
		y = 0.0,
		theta = 0.0,
		phi = 0.0,
		psi = 0.0,
		rho = 3.0,
		w = 3.0,
		t_interval=0.25,
		timestep=1):

		self.init_x = x
		self.init_y = y
		self.init_theta = theta
		self.init_psi = psi
		self.init_phi = phi

		self.x = x
		self.y = y
		self.theta = theta 
		self.theta_displacement = 0
		self.psi = psi
		self.phi = phi

		self.psidot = 0
		self.phidot = 0
		self.time = 0


		#constants
		self.t_interval = t_interval
		self.timestep = timestep
		self.rho = rho
		self.w = w

		self.state = (self.theta, self.phi, self.psi)


	def reset(self):
		self.x = self.init_x
		self.y = self.init_y
		self.theta = self.init_theta
		self.psi = self.init_psi
		self.phi = self.init_phi
		self.state = (self.theta, self.phi, self.psi)

		self.theta_displacement = 0
		self.psidot = 0
		self.phidot = 0
		self.time = 0
		return self.state

	def set_state(self, theta, phi, psi):
		self.theta, self.phi, self.psi = theta, phi, psi

		self.state = (theta, phi, psi)
		return self.state

	def get_position(self):
		return self.x, self.y

	@staticmethod
	def TeLg(theta):
		"""
		:param theta: the inertial angle in radians
		:return: the lifted left action matrix given the angle
		"""
		arr = np.array([[cos(theta), -sin(theta), 0],
						[sin(theta), cos(theta), 0],
						[0, 0, 1]])
		return arr

	def J(self):
		return np.array([
			[-1*self.rho/2, -1*self.rho/2],
			[0, 0],
			[-1*self.rho/self.w, self.rho/self.w]
		])

	def M(self, theta, phi, psi, dphi, dpsi):
		da = np.array([
			[dphi],
			[dpsi]
		])

		f = -self.TeLg(theta) @ (self.J() @ da)
		xdot = f[0, 0]
		ydot = f[1, 0]
		thetadot = f[2, 0]
		M = [xdot, ydot, thetadot, dphi, dpsi]

		return M

	def robot(self, v, t, dphi, dpsi):
		_, _, theta, phi, psi = v
		dvdt = self.M(theta, phi, psi, dphi, dpsi)
		return dvdt

	def perform_integration(self, action, t_interval):
		if(t_interval == 0):
			return self.x, self.y, self.theta, self.phi, self.psi
		phidot, psidot = action
		v0 = [self.x, self.y, self.theta, self.phi, self.psi]
		t = np.linspace(0, t_interval, 11)
		sol = odeint(self.robot, v0, t, args=(phidot, psidot))
		x, y, theta, phi, psi = sol[-1]
		return x, y, theta, phi, psi

	def enforce_angle_range(self, angle_name):
		if angle_name == 'theta':
			angle = self.theta
		elif angle_name == 'phi':
			angle = self.phi
		else:
			angle = self.psi
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
		elif angle_name == 'psi':
			self.psi = angle

	def update_params(self, x, y, theta, phi, psi):
		self.x = x
		self.y = y
		self.theta = theta
		self.enforce_angle_range('theta')

		self.psi = psi
		self.enforce_angle_range('psi')
		self.phi = phi
		self.enforce_angle_range('phi')

	def update_dots(self, 
		phidot1, 
		psidot1, 
		t1 = None, 
		phidot2 = 0,
		psidot2 = 0,
		t2 = None):

		c3 = 1

		# one move made
		if t2 is None:
			t2 = 0

		if t1 + t2 < self.t_interval + self.timestep:
			c3 = (t1 + t2) / (self.t_interval * self.timestep)

		if t1 + t2 == 0:
			c1 = 0
			c2 = 0
		else:
			c1 = t1 / (t1 + t2)
			c2 = t2 / (t1 + t2)
		self.psidot = (c1 * psidot1 + c2 * psidot2) * c3
		self.phidot = (c1 * phidot1 + c2 * phidot2) * c3

	def move(self, action, timestep = 1):
		self.timestep = timestep

		phidot, psidot = action

		t = self.timestep * self.t_interval

		psi = self.psi + psidot * t
		phi = self.phi + phidot * t

		d_theta = 0

		old_theta = self.theta
		x, y, theta, phi, psi = self.perform_integration(action, t)
		d_theta += theta - old_theta
		self.update_params(x, y, theta, phi, psi)
		self.update_dots(phidot, psidot, t)

		self.theta_displacement = d_theta
		self.state = (self.theta, self.phi, self.psi)

if __name__ == "__main__":

	# create a robot simulation
	robot = WheelChairRobot(t_interval=1)
	
	x_pos = [robot.x]
	y_pos = [robot.y]
	thetas = [robot.theta]
	times = [0]
	phi = [robot.phi]
	psi = [robot.psi]
	robot_params = []
	print('initial x y theta phi psi: ', robot.x, robot.y, robot.theta, robot.phi, robot.psi)
	for t in range(100):
		print(t + 1, 'th iteration')
		phidot = 1 / 3 * cos(t / 5)
		psidot = 1 / 3 * sin(t / 5)
		action = (phidot, psidot)
		start = time.time()
		robot.move(action)
		end = time.time()
		print("Move time: {}".format(end - start))
		print('action taken(aheaddot, ataildot): ', action)
		print('robot x y theta ahead atail: ', robot.x, robot.y, robot.theta, robot.phi, robot.psi)
		x_pos.append(robot.x)
		y_pos.append(robot.y)
		thetas.append(robot.theta)
		times.append(t + 1)
		phi.append(robot.phi)
		psi.append(robot.psi)
		robot_param = [robot.x,
					   robot.y,
					   robot.theta,
					   float(robot.phi),
					   float(robot.psi),
					   robot.phidot,
					   robot.psidot]
		robot_params.append(robot_param)

	#generate_csv(robot_params, "results/RobotTestResults/SwimmerIdealFluid/" + "result.csv")
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

	plt.plot(times, phi)
	plt.ylabel('phi displacements')
	plt.show()

	plt.plot(times, psi)
	plt.ylabel('psi displacements')
	plt.show()

	plt.plot(times, x_pos)
	plt.ylabel('x positions')
	plt.show()

	plt.plot(times, y_pos)
	plt.ylabel('y positions')
	plt.show()

	plt.plot(times, thetas)
	plt.ylabel('thetas')
	plt.show()
	plt.close()



