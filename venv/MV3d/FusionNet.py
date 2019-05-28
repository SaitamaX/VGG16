# MV3D network
from __future__ import print_function
from os.path import expanduser
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import datetime
import random
import pdb
import model

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool('debug', 'True', 'Debug mode: True/ False')
tf.flags.DEFINE_float('learning_rate', '1e-3', 'Learning rate for Adam Optimizer')
tf.flags.DEFINE_integer('batch_size', '1', "batch size for training")
tf.flags.DEFINE_string('logs_dir', 'MV3D_logs/', 'path to logs directory')
tf.flags.DEFINE_bool('regularization', 'False', 'Debug mode: True/ False')
tf.flags.DEFINE_string('model_dir', 'Model_zoo/', 'Path to vgg model mat')

# training loss
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

# main function

# training_data: a list containing training data
# training_data[0] = data_birdview
# training_data[1] = data_frontview
# training_data[2] = data_birdview_rois
# training_data[3] = data_frontview_rois
# training_data[4] = data_birdview_box_ind
# training_data[5] = data_frontview_box_ind
# training_data[6] = data_ROI_labels
# training_data[7] = data_ROI_regs
# training_data[8] = data_cls_mask
# training_data[9] = data_reg_mask

def train_network(train_data, val_data, MAX_EPOCH, weight, reg_weight, keep_prob):
    data_birdview = train_data[0]
    data_frontview = train_data[1]
    data_birdview_rois = train_data[2]
    data_frontview_rois = train_data[3]
    data_birdview_box_ind = train_data[4]
    data_birdview_box_ind = train_data[5]
    data_ROI_labels = train_data[6]
    data_ROI_regs = train_data[7]
    data_cls_mask = train_data[8]
    data_reg_mask = train_data[9]

    val_data_birdview = val_data[0]
    val_data_frontview = val_data[1]
    val_data_birdview_rois = val_data[2]
    val_data_frontview_rois = val_data[3]
    val_data_birdview_box_ind = val_data[4]
    val_data_birdview_box_ind = val_data[5]
    val_data_ROI_labels = val_data[6]
    val_data_ROI_regs = val_data[7]
    val_data_cls_mask = val_data[8]
    val_data_reg_mask = val_data[9]

    BATCHES = np.shape(data_birdview)[0]
    BATCHES1 = np.shape(val_data_birdview)[0]
    NUM_OF_REGRESSION_VALUE = np.shape(data_ROI_regs)[2]
    IMAGE_SIZE_BIRDVIEW = [data_birdview.shape[1], data_birdview.shape[2]]
    IMAGE_SIZE_FRONTVIEW = [data_frontview.shape[1], data_frontview.shape[2]]
    BIRDVIEW_CHANNEL = data_birdview.shape[3]
    FRONTVIEW_CHANNEL = data_frontview.shape[3]
    NUM_OF_ROI = 200
    ROI_H = 10
    ROI_W = 2

    with tf.name_scope('Input'):
        birdview = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_BIRDVIEW[0], IMAGE_SIZE_BIRDVIEW[1], BIRDVIEW_CHANNEL], \
                                  name='birdview')
        frontview = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE_FRONTVIEW[0], IMAGE_SIZE_FRONTVIEW[1], FRONTVIEW_CHANNEL], \
                                                      name='frontview')
        frontview_rois = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * NUM_OF_ROI, 4], name='frontview_rois')
        birdview_rois = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * NUM_OF_ROI, 4], name='birdview_rois')

        frontview_box_ind = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI], name='frontview_box_ind')
        birdview_box_ind = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI], name='birdview_box_ind')

        before_sample_cls_mask = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI, 1],
                                                name="before_sample_cls_mask")
        after_sample_cls_mask = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI, 1],
                                               name="after_sample_cls_mask")
        reg_mask = tf.placeholder(tf.int32, shape=[FLAGS.batch_size * NUM_OF_ROI, 1], name="reg_mask")
        gt_ROI_labels = tf.placeholder(tf.int64, shape=[FLAGS.batch_size, NUM_OF_ROI], name="gt_ROI_labels")
        gt_ROI_regs = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, NUM_OF_ROI, NUM_OF_REGRESSION_VALUE],
                                     name="gt_ROI_regs")

    with tf.name_scope('MV3D'):
        Overall_loss, classification_loss, regression_loss, logits_cls, logits_reg = model.MV3D(birdview, frontview, after_sample_cls_mask,\
            reg_mask, gt_ROI_labels, gt_ROI_regs, birdview_rois, frontview_rois, birdview_box_ind, frontview_box_ind, ROI_H, ROI_W, \
                NUM_OF_REGRESSION_VALUE, FLAGS.model_dir, weight, reg_weight, FLAGS.debug, keep_prob)
        before_sample_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits_cls, dimension=1), tf.reshape(gt_ROI_labels, [-1])),\
                                                       tf.float32) * tf.cast(tf.reshape(before_sample_cls_mask, [-1]), tf.float32))\
            / tf.reduce_sum(tf.cast(tf.reshape(before_sample_cls_mask, [-1]), tf.float32))
        after_samplel_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(logits_cls, dimension=1), tf.reshape(gt_ROI_labels, [-1])), \
                                                       tf.float32) * tf.cast(tf.reshape(after_sample_cls_mask, [-1]), tf.float32))\
            / tf.reduce_sum(tf.cast(tf.reshape(after_sample_cls_mask, [-1]), tf.float32))

    with tf.name_scope('summary'):
        print("Setting up summary op...")
        tf.summary.image("bird_view_intensity", birdview[:, :, :, BIRDVIEW_CHANNEL - 2 : BIRDVIEW_CHANNEL - 1], max_outputs=2)
