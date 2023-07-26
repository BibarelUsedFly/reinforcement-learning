import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
from testbed import *

MEAN = 0.0
STDEV = 1.0
ARMS = 10
TIMECYCLES = 2000 ## Tiempo
TESTPROBLEMS = 500  ## Cantidad de ejecuciones a promediar
EPSILONS = [0.00, 0.01, 0.10] ## Agentes a comparar
COLORS = ["#33a02c", "#a6cee3", "#1f78b4"]

testbed = make_testbed(TESTPROBLEMS, MEAN, STDEV, ARMS)

final_data = []
final_datap = []
for agent in range(len(EPSILONS)):
    final_data.append(np.zeros(TIMECYCLES))
    final_datap.append(np.zeros(TIMECYCLES))

for i in range(TESTPROBLEMS):
    for agent in range(len(EPSILONS)):
        data = egreedy(testbed[i], TIMECYCLES, EPSILONS[agent])
        final_data[agent] += (1/(i+1)) * (data[1] - final_data[agent])
        final_datap[agent] += (1/(i+1)) * (data[2] - final_datap[agent])
    if (i % 100 == 0) and (i != 0):
        print("Ciclo {}/{} completado".format(i, TESTPROBLEMS))

## Graficar
matplotlib.style.use('seaborn')
fig1, axes = plt.subplots(2)
fig1.canvas.manager.set_window_title(
    '{}-Armed Testbed {} Steps Results'.format(ARMS, TIMECYCLES))

plt.subplot(2, 1, 1)
for d in range(len(final_data)):
    plt.plot(final_data[d], color=COLORS[d%len(COLORS)])
plt.ylabel('Recompensa Promedio')

plt.subplot(2, 1, 2)
for d in range(len(final_datap)):
    plt.plot(final_datap[d], color=COLORS[d%len(COLORS)])
plt.ylabel('Porcentaje de elección óptima')

fontP = FontProperties()
fontP.set_size('small')
fig1.legend(['${}-Greedy$'.format(EPSILONS[agent]) if EPSILONS[agent] > 0 \
              else '$Greedy$' for agent in range(len(EPSILONS))],
    loc='center',
    bbox_to_anchor=(0.535, 0.635),
    fancybox=True,
    frameon= True,
    shadow=True,
    prop=fontP)
axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
fig1.tight_layout()
plt.show()
