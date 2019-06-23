import tensorflow as tf
import numpy as np
import matplotlib.image as img


def pre(pic):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/mnist_cnn.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint("./model/"))
        graph = tf.get_default_graph()

        sess.run(tf.global_variables_initializer())

        X = graph.get_tensor_by_name('data:0')
        keep_prob = graph.get_tensor_by_name('keep_prob:0')
        result = graph.get_tensor_by_name('prediction:0')
        print(sess.run(result, feed_dict={X: pic, keep_prob: 1.0}))
        out = tf.argmax(result, axis=1)
        print('prediction label is: ', sess.run(out, feed_dict={X: pic, keep_prob:1.0}))

def img2data(path):
    pic = img.imread(path)
    image = np.reshape(pic, [1, 28 * 28])
    # print(image)
    return image

picture = img2data('./17.jpg')
pre(picture)

