import matplotlib.pyplot as plt

import numpy as np


def draw_plot(episodes, losses, rewards, path):
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    ax1.plot(episodes, losses)
    ax1.legend(['losses'])

    ax2.plot(episodes, rewards)
    ax2.legend(['rewards'])

    plt.xlabel('Episodes')
    plt.savefig(path)

    plt.close()
