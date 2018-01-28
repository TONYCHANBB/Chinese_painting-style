# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:40:55 2017

@author: chenbin
"""


import tensorflow as tf
import numpy as np
import scipy.io

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

def load_net(data_path):
    data = scipy.io.loadmat(data_path)
    mean = data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))
    weights = data['layers'][0]
    return weights, mean_pixel

def net_preloaded(weights, input_image, pooling):
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))#转换高和宽的位置
            bias = bias.reshape(-1) #-1表示由实际情况确定
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)
'''
1. input是输入的样本，在这里就是图像。input的shape=[batch, height, width, channels]。 
- batch是输入样本的数量 
- height, width是每张图像的高和宽 
- channels是输入的通道，比如初始输入的图像是灰度图，那么channels=1，如果是rgb，那么channels=3。对于第二层卷积层，channels=32。 
2. tf.constant(weights)表示卷积核的参数，shape的含义是[height,width,in_channels,out_channels]。 
3. strides参数表示的是卷积核在输入x的各个维度下移动的步长。了解CNN的都知道，在宽和高方向stride的大小决定了卷积后图像的size。这里为什么有4个维度呢？因为strides对应的是输入x的维度，所以strides第一个参数表示在batch方向移动的步长，第四个参数表示在channels上移动的步长，这两个参数都设置为1就好。重点就是第二个，第三个参数的意义，也就是在height于width方向上的步长，这里也都设置为1。 
4. padding参数用来控制图片的边距，’SAME’表示卷积后的图片与原图片大小相同，’VALID’的话卷积以后图像的高为Heightout=Height原图−Height卷积核+1StrideHeight， 宽也同理。
'''

def _pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
'''                        
这里用2∗2的max_pool和avg。参数ksize定义pool窗口的大小，每个维度的意义与之前的strides相同
第一个参数value：需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape
第二个参数ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
第三个参数strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
第四个参数padding：和卷积类似，可以取'VALID' 或者'SAME'
返回一个Tensor，类型不变，shape仍然是[batch, height, width, channels]这种形式
这个函数的功能是将整个图片分割成2x2的块，
对每个块提取出最大值输出。可以理解为对整个图片做宽度减小一半，高度减小一半的降采样
'''




def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel