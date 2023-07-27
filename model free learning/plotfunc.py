import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def plot_data(data_list, title, label_list):
    matplotlib.style.use('seaborn')
    fig1, _ = plt.subplots(len(data_list))
    fig1.canvas.manager.set_window_title(title)
    colors = ["#1f78b4", "#33a02c", "#a6cee3", "#1f78b4"]

    plt.subplot(1, 1, 1) ## (N, 1, 1)
    plt.ylabel("Recompensa Promedio")
    plt.xlabel("Episodios de entrenamiento")
    data = data_list[0]
    plt.plot([x for x in range(1, len(data)+1)], data[:, 1],
            color=colors[0])

    data = data_list[1]
    plt.plot([x for x in range(1, len(data)+1)], data[:, 1],
            color=colors[1])
    
    data = data_list[2] ## 0: SARSA 1: Q-LEA
    plt.plot([x for x in range(1, len(data)+1)], data[:, 1],
            color=colors[2])
    
    fontP = FontProperties()
    fontP.set_size('small')
    fig1.legend(label_list, loc='center',
                bbox_to_anchor=(0.400, 0.900),
                fancybox=True,
                frameon= True,
                shadow=True,
                prop=fontP)