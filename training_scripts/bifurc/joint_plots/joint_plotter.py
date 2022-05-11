import csv
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from mpl_toolkits.mplot3d import Axes3D

csv_name = "rollout10.csv"
#csv_name2 = "rollout11.csv"

def get_joint_lists(csv_name):
	a1_joint_data = []
	a2_joint_data = []
	with open(csv_name, newline='') as rollout_file:
		reader = csv.reader(rollout_file)
		x_dis = 0
		for row in reader:
			a1_joint_data.append(float(row[3]))
			a2_joint_data.append(float(row[4]))
			x_dis = float(row[0])
	print("final x displacement:", x_dis)
	return a1_joint_data, a2_joint_data


def plot_joints(a1_joint_data, a2_joint_data, viz="temp"):
	time = np.arange(len(a1_joint_data))	
	if(viz == "3d"):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(a1_joint_data, a2_joint_data, time, linestyle='dashed', marker='o')
		plt.show()
	elif(viz == "temp"):
		cm = plt.cm.ScalarMappable(cmap = 'viridis', norm=plt.Normalize(vmin=0, vmax=len(time) - 1))
		plt.colorbar(cm)
		cm.set_array([])
		for i, a2 in enumerate(a2_joint_data):
			plt.plot(a1_joint_data[i], a2, marker='o', c=cm.to_rgba(i+1))
		plt.plot(a1_joint_data, a2_joint_data, linestyle='dashed', linewidth=0.3)
		#plt.show()

def plot_snapshots(a1_joint_data, a2_joint_data, rows=2, cols=5):
	offset = int(len(a1_joint_data)/(rows*cols))
	print(offset)
	figure, axis = plt.subplots(rows, cols, figsize = (15, 15))


	start = 0

	for i in range(rows):
		for j in range(cols):
			axis[i, j].plot(a1_joint_data[start:start+offset], a2_joint_data[start:start+offset])
			title = "t: [" + str(start+1) + "," + str(start+offset+1) + "]"
			axis[i, j].title.set_text(title)
			start += offset

	figure.tight_layout(pad = 4.0)

	plt.show()
	

a1, a2 = get_joint_lists(csv_name)
#print(a1)
#print(a2)
# t = np.arange(100)
# y = [np.sin(x) for x in t]
plot_joints(a1, a2, viz = "temp")

#a1, a2 = get_joint_lists(csv_name2)

#plot_joints(a1, a2, viz = "temp")

plot_snapshots(a1, a2)

plt.show()