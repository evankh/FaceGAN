import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
import tensorflow as tf
import random
import datetime
import os, sys
import time

img_rows = 128
img_columns = 128
channels = 1

path = os.path.dirname(__file__)
print(path)
imgpath = os.path.join(path,"00000")
print(imgpath)
tfPath = os.path.join(path,"tfrecord")

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


tfrecords_filename = 'image001.tfrecords'

writer = tf.compat.v1.python_io.TFRecordWriter(tfrecords_filename)

#Path to images from dir
path_to_images = imgpath

#List of images - method of accessing images
filenum = len([name for name in os.listdir(path_to_images) if os.path.isfile(os.path.join(path_to_images, name))])

for p in range(1, filenum):
    fname = "00000/00000.png"
    img = np.array(Image.open(fname))

    img_raw = img.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(img_raw)}))

    writer.write(example.SerializeToString())

writer.close()
