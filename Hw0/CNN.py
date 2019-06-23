from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def print_size(t):
    print(t.op.name, ' ', t.get_shape().as_list())


def Weights(shape):
    Kernel = tf.Variable(tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32))
    return Kernel


def Bias(shape):
    bias = tf.Variable(tf.constant(0.1, shape=shape))
    return bias


def CNN():

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # 每个批次大小
    batch_size = 128
    # 一共有多少个批次
    n_batch = mnist.train.num_examples // batch_size

    data = tf.placeholder(tf.float32, [None, 28*28], name='data')
    label = tf.placeholder(tf.float32, [None, 10], name='data')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    image = tf.reshape(data, [-1, 28, 28, 1])


    # conv1
    W_conv1 = Weights([5, 5, 1, 32])
    B_conv1 = Bias([32])
    conv1 = tf.nn.conv2d(image, W_conv1, strides=[1, 1, 1, 1], padding='SAME')
    relu_1 = tf.nn.relu(conv1+B_conv1)

    pooling1 = tf.nn.max_pool(relu_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    print_size(pooling1)
    #conv2
    W_conv2 = Weights([5, 5, 32, 64])
    B_conv2 = Bias([64])
    conv2 = tf.nn.conv2d(pooling1, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
    relu_2 = tf.nn.relu(conv2+B_conv2)
    pooling2 = tf.nn.max_pool(relu_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    print_size(pooling2)
    flatten = tf.reshape(pooling2, [-1, 7*7*64])

    #FCL1
    W_FCL1 = Weights([7*7*64, 1024])
    B_FCL1 = Weights([1024])
    FCL1 = tf.matmul(flatten, W_FCL1) + B_FCL1


    #FCL2
    W_FCL2 = Weights([1024, 10])
    B_FCL2 = Weights([10])
    FCL2 = tf.matmul(FCL1, W_FCL2) + B_FCL2

    prediction = tf.nn.softmax(FCL2, name='prediction')

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=prediction))
    tf.summary.scalar('loss', loss)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)

    correct_preditction = tf.equal(tf.argmax(prediction, 1), tf.argmax(label, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_preditction, tf.float32))
    linear_loss = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter("logs/", sess.graph)
        for epoch in range(21):
            for batch in range (n_batch):
                batch_xs, batch_ys = mnist.train.next_batch(batch_size)
                summary = sess.run([train_step, loss], feed_dict={data: batch_xs, label: batch_ys, keep_prob: 0.7})
                linear_loss.append(loss)
            writer.add_summary(summary,epoch)
            acc = sess.run(accuracy, feed_dict={data: mnist.test.images, label: mnist.test.labels, keep_prob: 1.0})
            print("Iter: " + str(epoch) + ", acc: " + str(acc))
        saver = tf.train.Saver()
        save_path = saver.save(sess, './model/mnist_cnn.ckpt')
        print('model saved')
        plt.plot(linear_loss,c='b', linestyle='--',marker='o')
        plt.savefig('loss.png',dpi=500)
        print('loss figure saved')


CNN()
