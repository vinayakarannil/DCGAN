import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import os
import cv2

from model import GAN
 

data_dir = 'data/'
real_size = (32,32,3)
z_size = 100
learning_rate = 0.0002
alpha = 0.2
beta1 = 0.5


# Create the network
net = GAN(real_size, z_size, learning_rate, alpha=alpha, beta1=beta1)

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    sample_z = np.random.uniform(-1, 1, size=(16, z_size))
    gen_samples = sess.run(generator(net.input_z, 3, reuse=True, training=False),
                                   feed_dict={net.input_z: sample_z})


    _ = view_samples(0, [gen_samples],4, 4, figsize=(5,5))
    plt.show()



