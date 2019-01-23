import argparse
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body



def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    # Step 1: Compute box scores
    ### START CODE HERE ### (≈ 1 line)
    # 计算所有小格子５个anchor box的分类结果的概率
    box_scores = box_confidence * box_class_probs
    ### END CODE HERE ###
    
    # Step 2: Find the box_classes thanks to the max box_scores, keep track of the corresponding score
    ### START CODE HERE ### (≈ 2 lines)
    # https://stackoverrun.com/cn/q/12993258
    # 计算所有小格子５个anchor box的分类结果中　概率最高的一类　所在的坐标.   shape:（19,19,5）
    box_classes = K.argmax(box_class_probs, axis=-1)  
    # 计算所有小格子５个anchor box的分类结果中　最高的概率值.   shape:（19,19,5）
    box_class_scores = K.max(box_scores, axis=-1) 
    ### END CODE HERE ###
    
    # Step 3: Create a filtering mask based on "box_class_scores" by using "threshold". The mask should have the
    # same dimension as box_class_scores, and be True for the boxes you want to keep (with probability >= threshold)
    ### START CODE HERE ### (≈ 1 line)
    # 计算所有小格子５个anchor box的分类结果中，概率大于threshold的设置为　Ｔrue，否则，设置为False, shape:(19,19,5)
    filtering_mask = box_class_scores >= threshold 
    ### END CODE HERE ###
    
    # Step 4: Apply the mask to scores, boxes and classes
    ### START CODE HERE ### (≈ 3 lines)
    # 在所有小格子的５个anchor box中，去除box_class_scores中　概率得分小于threshold　的部分
    # box_class_scores是三维矩阵，filtering_mask是三维的矩阵，mask之后返回3-3+1=1维的向量
    # 但是这个一维向量的个数是不确定的,由filtering_mask的True的个数决定
    # scores的shape为(?,) 表示这是一维的向量,且元素个数不确定
    scores = tf.boolean_mask(box_class_scores, filtering_mask)
    
    # 在所有小格子的５个anchor box中，去除boxes中　概率得分小于threshold　的部分
    # boxes是四维矩阵，filtering_mask是三维的矩阵，mask之后返回4-3+1=2维的矩阵
    # 但是这个二维向量的 第一个维度  的元素个数是不确定的,由filtering_mask的True的个数决定
    # boxes的shape为(?,4) 表示这是二维的向量,且第一维元素个数不确定, 第二维元素个数为4个
    
    boxes = tf.boolean_mask(boxes, filtering_mask)
    # 在所有小格子的５个anchor box中，去除box_classes中　概率得分小于threshold　
    # classes是三维矩阵，filtering_mask是三维的矩阵，mask之后返回3-3+1=1维的向量
    # 但是这个一维向量的个数是不确定的,由filtering_mask的True的个数决定
    # classes的shape为(?,) 表示这是一维的向量,且元素个数不确定
    classes = tf.boolean_mask(box_classes, filtering_mask)
    ### END CODE HERE ###
    
    # 最终保留所有19*19个格子中概率较大的anchor box的分数scores, 边框boxes, 类别classes
    return scores, boxes, classes



def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32') 
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) 
    
    # tf.image.non_max_suppression 用法可以参考以下网址的内容
    # https://blog.csdn.net/m0_37393514/article/details/81777244
    # 得到 去除交并比大于0.5的数据 后留下来的 边框在boxes里的下标
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes_tensor, iou_threshold=iou_threshold)
    
    # 通过tf.gather函数, 传入上面得到的下标, 即可得到保留下来的相关数据:坐标boxes, 得分scores, 类别classes
    boxes = tf.gather(boxes, nms_indices)
    scores = tf.gather(scores, nms_indices)
    classes = tf.gather(classes, nms_indices)
    
    return scores, boxes, classes



def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=10, score_threshold=.6, iou_threshold=.5):
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs
	
    # 转换成用顶点表示的坐标
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # 过滤所有 可信度 低于score_threshold的 anchor box
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    
    # 由于YOLO是用608*608的图片训练出来的, 如果想测试其他分辨率的图片, 
    # 我们需要通过下面的函数来对anchor box 的位置和尺寸进行相应的缩放.
    boxes = scale_boxes(boxes, image_shape)

    # 进行 非最大值抑制 
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    
    return scores, boxes, classes

# 创建一个session
sess = K.get_session()

# 读取类别
class_names = read_classes("model_data/coco_classes.txt")
# 读取 anchor box 的长度和宽度
anchors = read_anchors("model_data/yolo_anchors.txt")
# 设置我们要测试的图片尺寸
image_shape = (720., 1280.)    

"""
载入预先训练好的模型
yolo.h5文件获取方法:
    git clone https://github.com/allanzelener/yad2k.git
    cd yad2k
    wget http://pjreddie.com/media/files/yolo.weights
    wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolo.cfg
    ./yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
    
打开 https://github.com/allanzelener/YAD2K 低下有详细说明
"""
yolo_model = load_model("model_data/yolo.h5")

# yolo_model 的输出是 (m, 19, 19, 5, 85) 的张量 
# 利用 yolo_head函数 将 YOLO 模型的最后一层输出转换成(box_confidence, box_xy, box_wh, box_class_probs)的tuple形式
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))

# 过滤不需要的 anchor box
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

# 根据图片文件名,输出图片的检测结果
def predict(sess, image_file):

    # 对图片预处理, image_data 会增加一个维度, 变成 (1, 608, 608, 3), 这将作为CNN的输入
    image, image_data = preprocess_image("images/" + image_file, model_image_size = (608, 608))

    # 喂入数据, 运行 session
    out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes], feed_dict={yolo_model.input:image_data, K.learning_phase():0})


    # 打印预测信息
    print('Found {} boxes for {}'.format(len(out_boxes), image_file))
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join("out", image_file), quality=90)
    output_image = scipy.misc.imread(os.path.join("out", image_file))
    imshow(output_image)
    plt.show()
    
    return out_scores, out_boxes, out_classes

out_scores, out_boxes, out_classes = predict(sess, image_file="0059.jpg")