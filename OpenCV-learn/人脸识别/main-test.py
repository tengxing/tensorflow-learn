############################################################################
# -*- coding:utf8 -*-                                                      #
# created by tengxing on 2017.5.19                                         #
# mail tengxing7452@163.com                                                #
# github github.com/tengxing                                               #
# description opencv 人脸识别   测试类                                            #
############################################################################

import cv2
import matplotlib
import matplotlib.pyplot as plt
import sys
import time

from faceUtil import *

filename = "./input/008.jpg"
face_model = ""
output_name = "./out/%s.jpg" % (time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

#image_file = sys.argv[1]
#if not (image_file):
#    img_name = filename
img_name = filename

# 图像检测
images,img = detect_faces(model_face, img_name)

# 显示图片
#plt.imshow(img)  # 显示图片
#plt.axis('off')  # 不显示坐标轴
#plt.show()
# 保存图像
save_image(output_name, img)



