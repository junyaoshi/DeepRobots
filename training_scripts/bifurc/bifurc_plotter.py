import csv
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from math import pi

USE_ABS = True

NUM_BATCHES = 12
exclude = {4, 6}

a_max_arr = np.linspace(pi/2, 3, NUM_BATCHES, endpoint = True)

avg_a1s = []; avg_a2s = []; 
a1_mins = []; a1_maxs = []; a2_mins = []; a2_maxs = [];

batch = 1

for i in range(NUM_BATCHES):
	csv_name = "batch" + str(batch) + "/rollout" + str(i) + ".csv"

	with open(csv_name, newline='') as rollout_file:
		reader = csv.reader(rollout_file)
		a1_tot = 0; a2_tot = 0
		count = 0
		a1_min = float('inf'); a1_max = -1*float('inf');
		a2_min = float('inf'); a2_max = -1*float('inf');
		for row in reader:
			a1_tot = a1_tot + float(row[3])
			a2_tot = a2_tot + float(row[4])
			a1_min = min(float(row[3]), a1_min);
			a2_min = min(float(row[4]), a2_min);
			a1_max = max(float(row[3]), a1_max);
			a2_max = max(float(row[4]), a2_max);
			count += 1
		if(USE_ABS):
			avg_a1s.append(abs(a1_tot/count))
			avg_a2s.append(abs(a2_tot/count))
			if(a1_tot < 0):
				a1_min, a1_max = -1*a1_max, -1*a1_min
			if(a2_tot < 0):
				a2_min, a2_max = -1*a2_max, -1*a2_min
		else:
			avg_a1s.append((a1_tot/count))
			avg_a2s.append((a2_tot/count))
		a1_mins.append(a1_min); a1_maxs.append(a1_max);
		a2_mins.append(a2_min); a2_maxs.append(a2_max);

plot_style= "--bo"
marker_size = 3

y = []
a1x = []
a2x = []

a1mx = []
a1mn = []

a2mx = []
a2mn = []
for i in range(NUM_BATCHES):
	if(i in exclude):
		continue
	y.append(a_max_arr[i])
	a1x.append(avg_a1s[i])
	a2x.append(avg_a2s[i])
	a1mx.append(a1_maxs[i])
	a1mn.append(a1_mins[i])
	a2mx.append(a2_maxs[i])
	a2mn.append(a2_mins[i])

a1bot = np.array(a1x) - np.array(a1mn)
a1top = np.array(a1mx) - np.array(a1x)

a2bot = np.array(a2x) - np.array(a2mn)
a2top = np.array(a2mx) - np.array(a2x)


plt.errorbar(y, a1x, linestyle='--', color = 'b', marker = 'o', lolims = False, yerr=(a1bot, a1top), markersize=marker_size)
plt.xlabel('a_max')
plt.ylabel('abs() of avg a1')
plt.show()

plt.errorbar(y, a2x, linestyle='--', color = 'b', marker = 'o', lolims = False, yerr=(a1bot, a1top), markersize=marker_size)
plt.xlabel('a_max')
plt.ylabel('abs() of avg a2')
plt.show()

