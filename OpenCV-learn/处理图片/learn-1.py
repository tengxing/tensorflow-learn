###################################################################
# -*- coding:utf8 -*-
# created by tengxing on 2017.5.18
# mail tengxing7452@163.com
# github github.com/tengxing
# description opencv读取，缩放，生成图片
####################################################################
import cv2
import matplotlib as plt

filename = "test.jpg"

##read image
img = cv2.imread(filename=filename)
print "height:%spx"%img.shape[0] # out: width:978px
print type(img) # out: <type 'numpy.ndarray'>
print img.size  # out: 5304672
print img
#[[[ 4  4  4]
#  ....
#  [ 4  4  4]]]

## show image
image = cv2.imshow(winname="image",mat=img)
#plt.imshow(img)  # 显示图片
#plt.axis('off')  # 不显示坐标轴
#plt.show()
cv2.waitKey(0)
"""
cv2.waitKey() 是一个键盘绑定函数。函数等待特定的几毫秒，看是否有键盘输入。特定的几毫秒之内，如果
按下任意键，这个函数会返回按键的ASCII 码值，程序将会继续运行。如果没
有键盘输入，返回值为-1，如果我们设置这个函数的参数为0，那它将会无限
期的等待键盘输入。
"""
cv2.destroyWindow(image)
"""
cv2.destroyAllWindows() 可以轻易删除任何我们建立的窗口。如果
你想删除特定的窗口可以使用cv2.destroyWindow()，在括号内输入你想删
除的窗口名。
"""

## resize image
img = cv2.resize(img,(200,100),interpolation=None)
"""
interplolation为缩放时的插值方式，有三种插值方式：
cv2.INTER_AREA:使用象素关系重采样。当图像缩小时候，该方法可以避免波纹出现。当图像放大时，类似于 CV_INTER_NN方法　　　　
cv2.INTER_CUBIC: 立方插值
cv2.INTER_LINEAR: 双线形插值　
cv2.INTER_NN: 最近邻插值
"""

print img.shape # out: (100, 200, 3)
## rewrite image
cv2.imwrite('out.png',img)



