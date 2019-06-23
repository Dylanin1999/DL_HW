import matplotlib.image as img
import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import tensorflow as tf

def print_size(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def Weights(shape):
    Kernel = tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32))
    return Kernel


def Bias(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias


def prediction(pic):

    # 每个批次大小
    batch_size = 128
    # 一共有多少个批次

    data = tf.placeholder(tf.float32, [None, 28*28])
    label = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    image = tf.reshape(data, [-1, 28, 28, 1])


    # conv1
    W_conv1 = Weights([5, 5, 1, 6])
    B_conv1 = Bias([6])
    conv1 = tf.nn.conv2d(image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    relu_1 = tf.nn.relu(conv1+B_conv1)

    pooling1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print_size(pooling1)
    #conv2
    W_conv2 = Weights([5, 5, 6, 16])
    B_conv2 = Bias([16])
    conv2 = tf.nn.conv2d(pooling1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    relu_2 = tf.nn.relu(conv2+B_conv2)
    pooling2 = tf.nn.max_pool(relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    print_size(pooling2)
    flatten = tf.reshape(pooling2, [-1, 7*7*16])

    #FCL1
    W_FCL1 = Weights([7*7*16, 84])
    B_FCL1 = Weights([84])
    FCL1 = tf.matmul(flatten, W_FCL1) + B_FCL1

    #FCL2
    W_FCL2 = Weights([84, 10])
    B_FCL2 = Weights([10])
    FCL2 = tf.matmul(FCL1, W_FCL2) + B_FCL2

    prediction = tf.nn.softmax(FCL2)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=prediction))

    train_step = tf.train.AdamOptimizer(1e-2).minimize(loss)

    correct_preditction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))
    res = tf.argmax(prediction, 1)
    accuracy = tf.reduce_mean(tf.cast(correct_preditction, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './model/mnist_cnn.ckpt')
        print(sess.run(res, feed_dict={data: pic, keep_prob:1.0}))
        print('-----------------------------')

def img2data(path):
    pic = img.imread(path)
    image = np.reshape(pic, [1, 28*28])
   # print(image)
    return image

picture = img2data('./32.jpg')
prediction(picture)

