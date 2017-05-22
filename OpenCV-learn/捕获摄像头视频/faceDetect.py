############################################################################
# -*- coding:utf8 -*-                                                      #
# created by tengxing on 2017.5.22                                         #
# mail tengxing7452@163.com                                                #
# github github.com/tengxing                                               #
# description 摄像头人脸检测 并保存                                           #
############################################################################

import sys
reload(sys)
sys.setdefaultencoding("utf-8")
import cv2


face_model = "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"
out_file = "output.avi"


def faceDetect(img, face_cascade):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img


def main():
    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    face_cascade = cv2.CascadeClassifier(face_model)
    fourcc = cv2.VideoWriter_fourcc(*'flv1')  # 'F', 'L', 'V', '1'
    video = cv2.VideoWriter(out_file, fourcc, 20.0, (width, height))
    while(True):
        ret, frame = cap.read()
        if ret == True:
            frame = faceDetect(frame, face_cascade)
        cv2.imshow("头像检测", frame)
        video.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video.release()
    cap.release()
    cv2.destroyAllWindows()


main()