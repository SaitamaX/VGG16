from __future__ import print_function
from os.path import expanduser
import tensorflow as tf
import numpy as np
import model
import TensorflowUtils as utils
import datetime
import random
import pdb

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
# to help add Optional parameters in command line
FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_bool("debug", "True", "Debug mode: True/ False")
tf.flags.DEFINE_float("learning_rate", "1e-3", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_integer("batch_size", "1", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "Birdview_proposal_logs/", "path to logs directory")
tf.flags.DEFINE_bool("regularization", "False", "Debug mode: True/ False")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")

# training loss
def train(loss_val, var_list):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads)

# main function
# train_data[0] = birdview; train_data[1] = anchor_label;
# train_data[2] = anchor_label_mask; train_data[3] = anchor_reg;
# train_data[4] = anchor_reg_mask;
# val_data has same structure as train data
# MAX_EPOCH: training epochs
# weight: weight of classification loss
# reg_weight: weight of l2 regularization term

def train_network(train_data, val_data, MAXEPOCH, weight, reg_weight, keep_prob):
    tr_birdview = train_data[0]
    tr_anchor_label = train_data[1]
    tr_anchor_reg = train_data[2]
    tr_anchor_label_mask = train_data[3]
    tr_anchor_reg_mask = train_data[4]

    val_birdview = val_data[0]
    val_anchor_label = val_data[1]
    val_anchor_reg = val_data[2]
    val_anchor_label_mask = val_data[3]
    val_anchor_reg_mask = val_data[4]

    BATCHES = tr_birdview.shape[0]
    BATCHES1 = val_birdview.shape[0]
    BIRDVIEW_CHANNEL = tr_birdview.shape[3]
    IMAGE_SIZE = [tr_birdview.shape[1], tr_birdview.shape[2]]
    OUTPUT_IMAGE_SIZE = [tr_birdview.shape[1] / 4, tr_birdview.shape[2] / 4]
    NUM_OF_REGRESSION_VALUE = tr_anchor_reg.shape[4]
    NUM_OF_ANCHOR = tr_anchor_reg.shape[3]

    with tf.name_scope("Input"):
        image = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, IMAGE_SIZE[0], IMAGE_SIZE[1], BIRDVIEW_CHANNEL], name="input_image")
        gt_anchor_labels = tf.placeholder(tf.int64, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1],\
                                                           NUM_OF_ANCHOR], name = "sample_anchor_masks")
        gt_anchor_regs = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1],\
                                                           NUM_OF_ANCHOR, NUM_OF_REGRESSION_VALUE], name="gt_anchor_regs")
        sample_anchor_cls_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1],\
                                                                    NUM_OF_ANCHOR], name="sample_anchor_masks")
        nonsample_anchor_cls_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1],\
                                                                       NUM_OF_ANCHOR],name="nonsample_anchor_cls_masks")
        anchor_reg_masks = tf.placeholder(tf.float32, shape=[FLAGS.batch_size * OUTPUT_IMAGE_SIZE[0] * OUTPUT_IMAGE_SIZE[1],\
                                                             NUM_OF_ANCHOR], name="anchor_reg_masks")

    with tf.name_scope("3D-RPN"):
        net, classification_loss, regression_loss, Overall_loss, all_rpn_logits_softmax, all_rpn_regs = \
            model.birdview_proposal_net(image, gt_anchor_labels, gt_anchor_regs, sample_anchor_cls_masks, anchor_reg_masks, weight, 
                                        reg_weight, FLAGS.model_dir, FLAGS.batch_size, FLAGS.debug, keep_prob)
        sample_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(all_rpn_logits_softmax, dimension=1), tf.reshape(gt_anchor_labels,\
            [-1])), tf.float32) * tf.reshape(sample_anchor_cls_masks, [-1])) / tf.reduce_sum(tf.reshape(sample_anchor_cls_masks, [-1]))
        non_sample_accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(all_rpn_logits_softmax, dimension=1), tf.reshape(gt_anchor_labels,\
            [-1])), tf.float32) * tf.reshape(nonsample_anchor_cls_masks, [-1])) / tf.reduce_sum(tf.reshape(nonsample_anchor_cls_masks, [-1]))


