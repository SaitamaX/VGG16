import numpy as np
import tensorflow as tf

def conv2d(x, shape, padding="SAME"):
    return tf.nn.conv2d(x, shape, strides=[1, 1, 1, 1], padding=padding)

def maxPool(x, shape=[1, 2, 2, 1], padding="SAME"):
    return tf.nn.max_pool(x, ksize=shape, strides=[1, 2, 2, 1], padding=padding)

def genWeights(shape):
    return tf.Variable(tf.truncated_normal(shape=shape, stddev=0.1))

def genBias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

convLayers = []
maxPoolLayers = []
placeHolders = []

sess = tf.InteractiveSession()
input_img = tf.placeholder(tf.float32, shape=[None, 224*224*3], name="input_img")
input_x = tf.reshape(input_img, shape=[-1, 224, 224, 3], name="input_img")

# 3 * 224 * 224 -> 64 * 224 * 224
conv1_w = genWeights([3, 3, 3, 64])
conv1_b = genBias([64])
conv1_h = tf.nn.relu(conv2d(input_x, conv1_w) + conv1_b)

# 64 * 224 * 224 -> 64 * 224 * 224
conv2_w = genWeights([3, 3, 64, 64])
conv2_b = genBias([64])
conv2_h = tf.nn.relu(conv2d(conv1_h, conv2_w) + conv2_b)

# maxpool: 64 * 224 * 224 -> 64 * 112 * 112
maxpool1 = maxPool(conv2_h)

# 64 * 112 * 112 -> 128 * 112 * 112
conv3_w = genWeights([3, 3, 64, 112])
conv3_b = genBias([112])
conv3_h = tf.nn.relu(conv2d(maxpool1, conv3_w) + conv3_b)

# 128 * 112 * 112 -> 128 * 112 * 112
conv4_w = genWeights([3, 3, 112, 112])
conv4_b = genBias([112])
conv4_h = tf.nn.relu(conv2d(conv3_h, conv4_w) + conv4_b)

# maxpool: 128 * 112 * 112 -> 128 * 56 * 56
maxpool2 = maxPool(conv4_h)

# 128 * 56 * 56 -> 256 * 56 * 56
conv5_w = genWeights([3, 3, 112, 256])
conv5_b = genBias([256])
conv5_h = tf.nn.relu(conv2d(maxpool2, conv5_w) + conv5_b)

# 256 * 56 * 56 -> 256 * 56 * 56
conv6_w = genWeights([3, 3, 112, 256])
conv6_b = genBias([256])
conv6_h = tf.nn.relu(conv2d(conv5_h, conv6_w) + conv6_b)

# 256 * 56 * 56 -> 256 * 56 * 56
conv7_w = genWeights([3, 3, 112, 256])
conv7_b = genBias([256])
conv7_h = tf.nn.relu(conv2d(conv6_h, conv7_w) + conv7_b)

# 256 * 56 * 56 -> 256 * 56 * 56
conv8_w = genWeights([3, 3, 112, 256])
conv8_b = genBias([256])
conv8_h = tf.nn.relu(conv2d(conv7_h, conv8_w) + conv8_b)

# 256 * 56 * 56 -> 256 * 28 * 28
maxpool3 = maxPool(conv8_h)

# 256 * 28 * 28 -> 512 * 28 * 28
conv9_w = genWeights([3, 3, 256, 512])
conv9_b = genBias([512])
conv9_h = tf.nn.relu(conv2d(maxpool3, conv9_w) + conv9_b)

# 512 * 28 * 28 -> 512 * 28 * 28
conv10_w = genWeights([3, 3, 512, 512])
conv10_b = genBias([512])
conv10_h = tf.nn.relu(conv2d(conv9_h, conv10_w) + conv10_b)

# 512 * 28 * 28 -> 512 * 28 * 28
conv11_w = genWeights([3, 3, 512, 512])
conv11_b = genBias([512])
conv11_h = tf.nn.relu(conv2d(conv10_h, conv11_w) + conv11_b)

# 512 * 28 * 28 -> 512 * 28 * 28
conv12_w = genWeights([3, 3, 512, 512])
conv12_b = genBias([512])
conv12_h = tf.nn.relu(conv2d(conv11_h, conv12_w) + conv12_b)



