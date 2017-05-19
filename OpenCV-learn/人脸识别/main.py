############################################################################
# -*- coding:utf8 -*-                                                      #
# created by tengxing on 2017.5.19                                         #
# mail tengxing7452@163.com                                                #
# github github.com/tengxing                                               #
# description opencv 人脸识别   测试类                                            #
############################################################################

import cv2
import matplotlib as plt
import sys
import time

from faceUtil import *

filename = "./input/002.jpg"
face_model = ""
output_name = "./out/%s.jpg" % (time.time())

#image_file = sys.argv[1]
#if not (image_file):
#    img_name = filename
img_name = filename

images,img = detect_faces(model_face, img_name)

save_image(output_name,img)



