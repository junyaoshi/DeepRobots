import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt

fname = "trial_12_59_trial_1_log.txt"

df = pd.read_csv(fname)

a1 = df['ahead'].values
a2 = df['atail'].values

plt.hist2d(a1, a2, bins = 30, cmap = 'inferno')
plt.show()



fname = "trial_12_59_trial_0_log.txt"

df = pd.read_csv(fname)

a1 = df['ahead'].values
a2 = df['atail'].values

plt.hist2d(a1, a2, bins = 30, cmap = 'inferno')
plt.show()


a = hi()


def hi():
	return 2