from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import time
import data_helper as data_helpers

beginTime = time.time()

#Parameter definition
batch_size = 100
learning_rate = 0.005
max_steps = 1000

# Prepare data
data_sets = data_helpers.load_data()

# Define input placeholders
images_placeholder = tf.placeholder(tf.float32, shape=[None, 3072])
labels_placeholder = tf.placeholder(tf.int64, shape=[None])



