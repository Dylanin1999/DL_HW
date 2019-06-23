from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np
import os
import scipy
import cv2 as cv


def load_data(save_dir):
    mnist = input_data.read_data_sets('./data/', one_hot=True)
    image_arr = mnist.train.images
    label_arr = mnist.train.labels
    for i in range(len(image_arr)):
        image = image_arr[i].reshape(28, 28)
        filename = save_dir + str(np.argmax(label_arr[i])) + '/' + str(i) + '.jpg'
        print(filename)
        scipy.misc.toimage(image, cmin=0.0, cmax=1.0).save(filename)


save_dir = './img/'
if os.path.exists(save_dir) is False:
    os.mkdir(save_dir)
    for num in range(10):
        os.mkdir(save_dir+str(num))

load_data(save_dir)