说明：这个模型灵感来源于inception,在其基础上进行改造

特点：

1：这是个通用学习模型，图片保存在内存中

2：输入模型自定义大小

train.py:训练

pre_data.py 准备数据

image_pooling.py 大图片快速pool

suf_data.py 批量train样本

train_model.py 训练的核心模型(lay)

convolutional_model.py 卷积模型封装库

