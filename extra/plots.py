import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import FuncFormatter

from .utils import *

def basic_plot(datas, fixed_colors={}, title=None, filename=None):
    for name in datas:
        epochs = list(range(len(datas[name][0])))
        color = fixed_colors[name]
        _, low, high = mean_confidence_interval(datas[name], 0.95)

        plt.plot(np.array(datas[name]).mean(axis=0), color=hsv_to_rgb(*color), label=name)
        plt.fill_between(epochs, low, high, color=hsv_to_rgb(*color), alpha=0.3)
    plt.legend()
    # plt.xlim(-5, 150)
    # plt.ylim(10, 80)
    plt.xlabel("Communication round")
    plt.ylabel("Accuracy")
    plt.yticks(plt.yticks()[0], [f"{int(i)}%" for i in plt.yticks()[0]])
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    if title is not None: plt.title(title)
    if filename:
        plt.savefig(filename)
        plt.close()
    else: plt.show()

def close_plot(datas, fixed_colors={}, title=None, option=None, filename=None):
    for name in datas:
        color = fixed_colors[name]
        plt.plot(np.array(datas[name]).mean(axis=0), color=hsv_to_rgb(*color), label=name)
    plt.legend()
    plt.xlim(4, 22)
    plt.xlabel("Communication round")
    plt.ylim(54, 72)
    plt.ylabel("Accuracy")

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    if title is not None: plt.title(title)
    if filename:
        plt.savefig(filename)
        plt.close()
    else: plt.show()

def doublezoom_plot(datas, fixed_colors={}, title=None, option={}, filename=None):
    fig = plt.figure(figsize=(6,5))
    sub1 = fig.add_subplot(2,2,1)
    sub2 = fig.add_subplot(2,2,2)
    sub3 = fig.add_subplot(2,2,(3,4))

    x0 = option["x0"] if ("x0" in option) else 0
    x1 = option["x1"] if ("x1" in option) else 20
    x2 = option["x2"] if ("x2" in option) else 80
    x3 = option["x3"] if ("x3" in option) else 100
    y0 = option["y0"] if ("y0" in option) else 10
    y1 = option["y1"] if ("y1" in option) else 80
    y2 = option["y2"] if ("y2" in option) else 50
    y3 = option["y3"] if ("y3" in option) else 80

    plot_error = True

    sub1.set_xlim(x0, x1)
    sub1.set_ylim(y0, y1)
    sub2.set_xlim(x2, x3)
    sub2.set_ylim(y2, y3)

    sub3.set_ylim(10, max(y1,y3))

    sub3.fill_between((x0,x1), 0, 100, facecolor='green', alpha=0.3)
    sub3.fill_between((x2,x3), 0, 100, facecolor='orange', alpha=0.3)

    for name in datas:
        epochs = list(range(len(datas[name][0])))
        color = fixed_colors[name]
        _, low, high = mean_confidence_interval(datas[name], 0.95)

        sub1.plot(np.array(datas[name]).mean(axis=0), color=hsv_to_rgb(*color), label=name)
        sub1.fill_between(epochs, low, high, color=hsv_to_rgb(*color), alpha=0.1)
        sub2.plot(np.array(datas[name]).mean(axis=0), color=hsv_to_rgb(*color), label=name)
        sub2.fill_between(epochs, low, high, color=hsv_to_rgb(*color), alpha=0.1)
        sub3.plot(np.array(datas[name]).mean(axis=0), color=hsv_to_rgb(*color), label=name)
        sub3.fill_between(epochs, low, high, color=hsv_to_rgb(*color), alpha=0.1)

    con1 = ConnectionPatch(xyA=(x0, y0), coordsA=sub1.transData,
                           xyB=(x0, max(y1,y3)), coordsB=sub3.transData, color="green")
    con2 = ConnectionPatch(xyA=(x1, y0), coordsA=sub1.transData,
                           xyB=(x1, max(y1,y3)), coordsB=sub3.transData, color="green")

    con3 = ConnectionPatch(xyA=(x2, y2), coordsA=sub2.transData,
                           xyB=(x2, max(y1,y3)), coordsB=sub3.transData, color="orange")
    con4 = ConnectionPatch(xyA=(x3, y2), coordsA=sub2.transData,
                           xyB=(x3, max(y1,y3)), coordsB=sub3.transData, color="orange")
    fig.add_artist(con1)
    fig.add_artist(con2)
    fig.add_artist(con3)
    fig.add_artist(con4)

    sub1.set_ylabel("Accuracy")
    sub3.set_xlabel("Communication round")
    sub3.set_ylabel("Accuracy")

    sub1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    sub2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.1f}%"))
    sub3.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    sub2.yaxis.tick_right()

    sub1.legend()
    sub2.legend()
    sub3.legend()
    if title is not None: plt.title(title)
    if filename:
        plt.savefig(filename)
        plt.close()
    else: plt.show()