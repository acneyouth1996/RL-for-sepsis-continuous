import matplotlib.pyplot as plt
import numpy as np
plt.rc('font', size=14)

data = np.genfromtxt('./res.csv',delimiter=',')

temp = [data[:, 0], data[:, 1], data[:,2], data[:, 3]]

fig, ax = plt.subplots()
ax.set_title('Off policy evaluation')
ax.boxplot(temp)

ax.set_xticklabels(['RL policy', 'Clinician policy', 'Random action policy', 'No action policy'])

plt.show()