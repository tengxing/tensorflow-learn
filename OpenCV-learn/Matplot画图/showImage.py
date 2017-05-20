############################################################################
# -*- coding:utf8 -*-                                                      #
# created by tengxing on 2017.5.19                                         #
# mail tengxing7452@163.com                                                #
# github github.com/tengxing                                               #
# description Matplot画图  显示图片                                          #
############################################################################

import matplotlib.pyplot as plt
import cv2
import os

image_path = "./data/008.jpg"
image_dir = "data"


# 显示一张图片
def show(image_path):
    img = cv2.imread(image_path)
   # plt.figure(image_path[-7:])  # 创建图表,截取字符
    plt.imshow(img)
    #plt.title(image_path[-7:-4])
    #plt.axis("off")
    plt.show()
    return


# 显示文件夹下的图片
def showDir(dir):
    images = eachFile(dir)
    plt.figure(dir, figsize=(15, 15))
    for i, name in enumerate(images):
        # show(name)
        img = cv2.imread(name)
        plt.subplot(2, 2, i+1) # A grid of 2 rows x 2 columns
        plt.axis('off')
        plt.imshow(img)
        plt.title(name[-7:-4])
    plt.show()
    return


# 遍历指定目录
def eachFile(filepath):
    pathDir = os.listdir(filepath)
    result = []
    for allDir in pathDir:
        child = os.path.join('%s/%s' % (filepath, allDir))
        result.append(child)
    return result

#show(image_path)

showDir(image_dir)






