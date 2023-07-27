import numpy as np

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from copy import deepcopy
from time import time
from plotfunc import plot_data
from Agents import n_sarsa, q_learning, montecarlo_control, nQsigma
from Models import Q, ACTION_SET, step_func, simu_func, BEGINNING, play_states

from parameters import *

## (function, (*args))
AGENTS = [(nQsigma, (ACTION_SET, step_func, BEGINNING,
                    play_states, N, ALPHA, GAMMA, EPSILON,
                    1.0, LEARNING_EPISODES,
                    False, True, 1)),
          (nQsigma, (ACTION_SET, step_func, BEGINNING,
                    play_states, N, ALPHA, GAMMA, EPSILON,
                    0.5, LEARNING_EPISODES,
                    False, True, 1)),
          (nQsigma, (ACTION_SET, step_func, BEGINNING,
                    play_states, N, ALPHA, GAMMA, EPSILON,
                    0.0, LEARNING_EPISODES,
                    False, True, 1))]
NAMES = ["Q(1)sigma (CV-SARSA)", "Q(.5)Sigma", "Q(0)sigma (Tree Backup)"]

true_start_time = time()

times = [0.0 for _ in range(len(AGENTS))]

true_data = [np.zeros((LEARNING_EPISODES, 2)) for _ in range(len(AGENTS))]
data = []

for i in range(REPEATS):
    actual_time = time()
    n = 0
    for agent in AGENTS:
        d, _ = agent[0](deepcopy(Q), *agent[1])
        data.append(d)
        times[n] += (1/(i+1)) * (time() - actual_time - times[n])
        actual_time = time()
        true_data[n] += (1/(i+1)) * (data[n] - true_data[n])
        n += 1
    print("Repeat {}/{}".format(i+1, REPEATS))

for n in range(len(NAMES)):
    print("AVG {} TIME:".format(NAMES[n]), times[n])
print("TOTAL TIME:", time() - true_start_time)

TITLE = 'Recompensa Promedio AVG{}'.format(REPEATS)
plot_data(true_data, TITLE, NAMES)
plt.show()