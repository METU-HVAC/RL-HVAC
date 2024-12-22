
import math
import matplotlib.pyplot as plt
total_training_steps = 1500
eps_threshold = 0
eps_end = 0.05
eps_start = 0.9
eps_decay = 5
eps_list = []
for i in range(total_training_steps):


    eps_threshold = eps_end + (eps_start - eps_end) * \
                math.exp(-1. * (i / total_training_steps) * eps_decay)
    eps_list.append(eps_threshold)
plt.plot(eps_list)
plt.show()