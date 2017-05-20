############################################################################
# -*- coding:utf8 -*-                                                      #
# created by tengxing on 2017.5.19                                         #
# mail tengxing7452@163.com                                                #
# github github.com/tengxing                                               #
# description Matplot画图  时间序列                                          #
############################################################################

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cv2
import numpy as np
import sys
reload(sys)
sys.setdefaultencoding('utf8')

# 显示连续函数
def continue_fun():
    x = np.linspace(0, 30, 100)
    plt.figure("幂函数")  #
    plt.plot(x, np.exp(x / 3))
    plt.figure("三角函数")  #
    ax1 = plt.subplot(2, 1, 1)  # sin
    ax2 = plt.subplot(2, 1, 2)  # cos
    plt.sca(ax1)  #
    plt.plot(x, np.sin(x))
    plt.sca(ax2)  #
    plt.plot(x, np.cos(x))

    plt.show()


def base_graph():
    # 1D data
    x = [1, 2, 3, 4, 5]
    y = [2.3, 3.4, 1.2, 6.6, 7.0]

    plt.figure("基础图形",figsize=(12, 6))

    plt.subplot(231)
    plt.plot(x, y)
    plt.title("plot")

    plt.subplot(232)
    plt.scatter(x, y)
    plt.title("scatter")

    plt.subplot(233)
    plt.pie(y)
    plt.title("pie")

    plt.subplot(234)
    plt.bar(x, y)
    plt.title("bar")

    plt.show()


def animal_graph():
    fig = plt.figure("动态图形")
    ax1 = fig.add_subplot(2, 1, 1, xlim=(0, 2), ylim=(-4, 4))
    ax2 = fig.add_subplot(2, 1, 2, xlim=(0, 2), ylim=(-4, 4))
    line, = ax1.plot([], [], lw=2)
    line2, = ax2.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        return line, line2


    def update(i):
        print i
        x = np.linspace(0, 2, 100)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        line.set_data(x, y)

        x2 = np.linspace(0, 2, 100)
        y2 = np.cos(2 * np.pi * (x2 - 0.01 * i)) * np.sin(2 * np.pi * (x - 0.01 * i))
        line2.set_data(x2, y2)
        return line, line2

    anim1 = animation.FuncAnimation(fig, update, init_func=init, frames=500, interval=10)
    plt.show()

#animal_graph()
#continue_fun()
base_graph()


