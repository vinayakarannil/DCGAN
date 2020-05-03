#!/usr/bin/python3
from time import process_time
print('Program started')

import warnings
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
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
from dataset import Dataset,view_samples
from constants import *

print('Model and Dataset imported %f' %  process_time()  )

if not isdir(data_dir):
    raise Exception("Data directory doesn't exist!")

imagesRaw = []
for image in os.listdir(data_dir) :
    img = cv2.imread(os.path.join(data_dir,image))
    if img is not None:
        imagesRaw.append(cv2.resize(img,(real_size[0],real_size[1]),interpolation = cv2.INTER_AREA))
    else :
        print(os.path.join(data_dir,image))

print('Images imported %f' %  process_time()  )

images=np.array(imagesRaw)
imagesRaw = []

trainset = images[:3000]
testset = images[3000:]

print('Numpy array created %f' %  process_time()  )

#idx = np.random.randint(0, len(trainset), size=36)
#fig, axes = plt.subplots(6, 6, sharex=True, sharey=True, #figsize=(5,5),)
#for ii, ax in zip(idx, axes.flatten()):
#    ax.imshow(trainset[ii], aspect='equal')
#    ax.xaxis.set_visible(False)
#    ax.yaxis.set_visible(False)

#plt.subplots_adjust(wspace=0, hspace=0)
#plt.show()


def train(net, dataset, epochs, batch_size, print_every=10, show_every=100, figsize=(5,5)):
    print('Training started %f' %  process_time()  )
    saver = tf.train.Saver()
    sample_z = np.random.uniform(-1, 1, size=(real_size[0]*2, z_size))

    samples, losses = [], []
    steps = 0

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        last = process_time()
        for e in range(epochs):
            for x in dataset.batches(batch_size):
                steps += 1
                # Sample random noise for G
                batch_z = np.random.uniform(-1, 1, size=(batch_size, z_size))
                # Run optimizers
                _ = sess.run(net.d_opt, feed_dict={net.input_real: x, net.input_z: batch_z})
                _ = sess.run(net.g_opt, feed_dict={net.input_z: batch_z, net.input_real: x})

                if steps % print_every == 0:
                    # At the end of each epoch, get the losses and print them out
                    train_loss_d = net.d_loss.eval({net.input_z: batch_z, net.input_real: x})
                    train_loss_g = net.g_loss.eval({net.input_z: batch_z})

                    print("Epoch {}/{}...".format(e+1, epochs),
                          "Discriminator Loss: {:.4f}...".format(train_loss_d),
                          "Generator Loss: {:.4f}".format(train_loss_g))
                    # Save losses to view after training
                    losses.append((train_loss_d, train_loss_g))

                if steps % show_every == 0:
                    gen_samples = sess.run(generator(net.input_z, 3, reuse=True, training=False),
                                   feed_dict={net.input_z: sample_z})
                    samples.append(gen_samples)
                    _ = view_samples(-1, samples, 8, 8, figsize=figsize,save=True,saveCount=len(samples))
            saver.save(sess, './checkpoints/generator {}.ckpt'.format(e+1))
            last = process_time() - last
            print('Epoch time is %f' % last)
            last = process_time()

        saver.save(sess, './checkpoints/generator.ckpt')
        print('Model trained %f' %  process_time()  )
    return losses, samples



# Create the network
net = GAN(real_size, z_size, learning_rate, alpha=alpha, beta1=beta1)
print('GAN created %f' %  process_time()  )

dataset = Dataset(trainset, testset)
print('Dataset created %f' %  process_time()  )

losses, samples = train(net, dataset, epochs, batch_size,print_every,show_every,figsize=(5,5))

_ = view_samples(0, samples, 4, 4, figsize=(10,5))
