import tensorflow as tf
import numpy as np
import scipy.misc as misc
import os, sys
from six.moves import urllib
import tarfile
import zipfile
import scipy.io


def get_model_data(dir_path, model_url):
    maybe_download_and_extract(dir_path, model_url)
    filename = model_url.split("/")[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        raise IOError("VGG Model not found!")
    data = scipy.io.loadmat(filepath)
    return data

def get_variable(weights, name):
    init = tf.constant_initializer(weights, dtype=tf.float32)
    var = tf.get_variable(name=name, initializer=init, shape=weights.shape)
    return var

def weight_variable(shape, stddev=0.02, name=None):
    initial = tf.truncated_normal(shape, stddev=stddev)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def maybe_download_and_extract(dir_path, url_name, is_tarfile = False, is_zipfile = False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    filename = url_name.split('/')[-1]
    filepath = os.path.join(dir_path, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write(
                '\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(url_name, filepath, reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
        if is_tarfile:
            tarfile.open(filepath, 'r:gz').extractall(dir_path)
        elif is_zipfile:
            with zipfile.ZipFile(filepath) as zf:
                zip_dir = zf.namelist()[0]
                zf.extractall(dir_path)

def bias_variable(shape, name=None):
    initial = tf.constant(0.0, shape=shape)
    if name is None:
        return tf.Variable(initial)
    else:
        return tf.get_variable(name, initializer=initial)

def get_tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def conv2d_basic(x, W, bias, keep_prob = 1.0):
    conv = tf.nn.dropout(tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME"), keep_prob)
    return tf.nn.bias_add(conv, bias)

def conv2d_strided(x, W, b, keep_prob = 1.0):
    conv = tf.nn.dropout(tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding="SAME"), keep_prob)
    return tf.nn.bias_add(conv, b)

def crop_and_concat(x1, x2, output_shape):
    offsets = tf.zeros(tf.pack([output_shape[0], 2]), dtype=tf.float32)
    x2_shape = tf.shape(x2)
    size = tf.pack((x2_shape[1], x2_shape[2]))
    x1_crop = tf.image.extract_glimpse(x1, size=size, offsets=offsets, centered=True)
    return tf.concat(3, [x1_crop, x2])

def conv2d_transpose_strided(x, W, b, output_shape=None, name = None, stride = 2, keep_prob = 1.0):
    if output_shape is None:
        output_shape = x.get_shape().as_list()
        output_shape[1] *= 2
        output_shape[2] *= 2
        output_shape[3] = W.get_shape().as_list()[2]

        conv = tf.nn.dropout(tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], name = name, padding="SAME"), \
                             keep_prob)
        return tf.nn.bias_add(conv, b)

def fully_connected(feature, output_channel, name, relu = Ture):
    input_shape = feature.get_shape().as_list()
    W = weight_variable([input_shape[1], output_channel], name = name + '_w')
    b = bias_variable([output_channel], name=name + '_b')
    transformed_feature = tf.matmul(feature, W) + b
    if relu==True:
        transformed_feature = tf.nn.relu(transformed_feature)
    return transformed_feature

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def max_pool(x, k_h, k_w, s_h, s_w, name = None, padding="SAME"):
    """ Apply a max pooling layer."""
    return tf.nn.max_pool(x, ksize=[1, k_h, k_w, 1], strides=[1, s_h, s_w, 1], padding=padding, name=name)

# def max_poolv2(x, k_h, k_w, s_h, s_w, name = None, padding="SAME" ):
    """Apply a dynamic max pooling layer."""

def avg_pool_2x2(x):
    return tf.nn.avg_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def local_response_norm(x):
    return tf.nn.lrn(x, depth_radius=5, bias=2, alpha=1e-4, beta=0.75)

def batch_norm(x, n_out, phase_train, scope='bn', decay=0.9, eps=1e-5):
    """batch normalization"""
    with tf.variable_scope(scope):
        beta = tf.get_variable(name='beta', shape=[n_out], initializer=tf.constant_initializer(0.0), trainable=True)
        gamma = tf.get_variable(name ='gamma', shape=[n_out], initializer=tf.random_normal_initializer(1.0, 0.02), trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay)
