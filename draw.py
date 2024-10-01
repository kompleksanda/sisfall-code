import matplotlib.pyplot as plt
import numpy as np


fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])

approaches = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")

x2 = np.array([0.8975871313672922, 0.8855227882037533, 0.9131367292225201, 0.8927613941018767, 0.8941018766756033, 0.8975871313672922, 0.8994638069705094])
tes = np.array([0.5294117647058824, 0.4, 0.4117647058823529, 0.5647058823529412, 0.5235294117647059, 0.5705882352941176, 0.5647058823529412])
#sd2 = np.array([0.0747104254, 0.0530682598, 0.0657771, 0.0435936791, 0.0567289541, 0.0667974181, 0.0687332269, 0.0593793767, 0.0539394972, 0.049810905, 0.0452937096, 0.0416791154, 0.0570768792, 0.0602386223, 0.0520705879])

ax.bar(approaches, x2, align='center', color='gray')
ax.set_title('SisFall Dataset')
#ax.set_yticks(np.arange(0,1,0.1))
#ax.set_xticks(np.arange(len(approaches)), np.arange(0,1,0.1))
ax.set_ylim([0, 1])

fig.text(0.5, 0.01, 'Cross Validation Accuracy for training', ha='center', fontsize=12)

#plt.show()

plt.savefig("fig/LOSO_Accuracy_train.pdf", bbox_inches="tight", pad_inches=0)
"""
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 1])

approaches = ("S1", "S2", "S3", "S4", "S5", "S6", "S7")

x2 = np.array([0.8975871313672922, 0.8855227882037533, 0.9131367292225201, 0.8927613941018767, 0.8941018766756033, 0.8975871313672922, 0.8994638069705094])
tes = np.array([0.5294117647058824, 0.4, 0.4117647058823529, 0.5647058823529412, 0.5235294117647059, 0.5705882352941176, 0.5647058823529412])
#sd2 = np.array([0.0747104254, 0.0530682598, 0.0657771, 0.0435936791, 0.0567289541, 0.0667974181, 0.0687332269, 0.0593793767, 0.0539394972, 0.049810905, 0.0452937096, 0.0416791154, 0.0570768792, 0.0602386223, 0.0520705879])

ax.bar(approaches, tes, align='center', color='gray')
ax.set_title('SisFall Dataset')
#ax.set_yticks(np.arange(0,1,0.1))
#ax.set_xticks(np.arange(len(approaches)), np.arange(0,1,0.1))
ax.set_ylim([0, 1])

fig.text(0.5, 0.01, 'Cross Validation Accuracy for testing', ha='center', fontsize=12)

#plt.show()

plt.savefig("fig/LOSO_Accuracy_test.pdf", bbox_inches="tight", pad_inches=0)
"""