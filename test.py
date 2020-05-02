from time import process_time
print('Program started')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
from scipy.io import loadmat
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm
import os
import cv2

print('Librearies imported %f' %  process_time()  )

from model import GAN
from model import generator
from dataset import view_samples
from constants import *

print('Model and constants imported %f' %  process_time()  )

# Create the network
net = GAN(real_size, z_size, learning_rate, alpha=alpha, beta1=beta1)
print('GAN created %f' %  process_time()  )

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    print('Session restored %f' %  process_time()  )

    sample_z = np.random.uniform(-1, 1, size=(int(real_size[0]/2), z_size))
    gen_samples = sess.run(generator(net.input_z, 3, reuse=True, training=False),
                                   feed_dict={net.input_z: sample_z})
    _ = view_samples(0, [gen_samples],4, 4, figsize=(5,5))
