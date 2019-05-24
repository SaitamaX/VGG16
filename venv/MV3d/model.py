# birdview proposal net (3D RPN)
# 3D proposal network
# region fusion
import tensorflow as tf
import TensorflowUtils as utils
import numpy as np
import random
import pdb

# Vgg net modified for birdview input
def vgg_net_birdview(weights, image, debug, keep_prob):
    layers = (
        'birdview_conv1_1', 'birdview_relu1_1', 'birdview_conv1_2', 'birdview_relu1_2', 'birdview_pool1',

        'birdview_conv2_1', 'birdview_relu2_1', 'birdview_conv2_2', 'birdview_relu2_2', 'birdview_pool2',

        'birdview_conv3_1', 'birdview_relu3_1', 'birdview_conv3_2', 'birdview_relu3_2', 'birdview_conv3_3',
        'birdview_relu3_3', 'birdview_pool3',

        'birdview_conv4_1', 'birdview_relu4_1', 'birdview_conv4_2', 'birdview_relu4_2', 'birdview_conv4_3',
        'birdview_relu4_3'

        # 'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        # 'relu5_3'
    )

    # output of retrained layer for vgg
    net = {}
    current = image
    channel = image.get_shape().as_list()[3]

    for i, name in enumerate(layers):
        kind = name[9 : 13]
        if kind == 'conv':
            if name == 'birdview_conv1_1':
                # Modify the first conv layer
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                kernels = np.concatenate((np.repeat(kernels[:, :, 0 : 1], channel / 3, axis = 2), np.repeat(kernels[:, :, 1 : 2], \
                    channel / 3, axis = 2), np.repeat(kernels[:, :, 2: 3], channel - 2 * (channel / 3), axis = 2)), axis = 2)
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
