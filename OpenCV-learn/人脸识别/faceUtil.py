############################################################################
# -*- coding:utf8 -*-                                                      #
# created by tengxing on 2017.5.19                                         #
# mail tengxing7452@163.com                                                #
# github github.com/tengxing                                               #
# description opencv 人脸识别                                               #
############################################################################

import cv2
import matplotlib as plt

# 默认值
model_face = '/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml'
min_height_dec=20
min_width_dec=20
min_height_thresh=50
min_width_thresh=50
FACE_PAD = 50

## face model
def get_model_type(model_name):
    if not(model_face):
        return cv2.CascadeClassifier(model_face)
    return cv2.CascadeClassifier(model_name)

## 人脸识别
## return faces and img
def detect_faces(face_model,image_name):
    face_cascade = get_model_type(face_model)
    print image_name
    img = cv2.imread(image_name)
    print img
    min_h = int(max(img.shape[0] / min_height_dec, min_height_thresh))
    min_w = int(max(img.shape[1] / min_width_dec, min_width_thresh))

    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5, minSize=(min_h, min_w))

    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x + width, y + height))
        draw_rect(img, x, y, width, height)  # 画线标记
    return result,img


## 画图
def draw_rect(img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        cv2.rectangle(img, (lower_cut[1], lower_cut[0]), (upper_cut[1], upper_cut[0]), (255, 0, 0), 2)


## 截图字图片
def sub_image(name, img, x, y, w, h):
        upper_cut = [min(img.shape[0], y + h + FACE_PAD), min(img.shape[1], x + w + FACE_PAD)]
        lower_cut = [max(y - FACE_PAD, 0), max(x - FACE_PAD, 0)]
        roi_color = img[lower_cut[0]:upper_cut[0], lower_cut[1]:upper_cut[1]]
        cv2.imwrite(name, roi_color)
        return name


## save image
def save_image(filename, img):
    cv2.imwrite(filename, img)