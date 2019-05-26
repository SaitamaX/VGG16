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
            elif name == 'birdview_conv4_1':
                # Modify the senventh conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index = random.sample(range(512), 256)
                kernels = kernels[:, :, :, sample_index]
                bias = bias[:, sample_index]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'birdview_conv4_2':
                # Modify the eighth conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'birdview_conv4_3':
                # pdb.set_trace()
                # Modify the ninth conv layer
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)

            else:
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
        # pdb.set_trace()
        return net
# Vgg net modified for frontview input
def vgg_net_frontview(weights, image, debug, keep_prob):
    layers = (
        'frontview_conv1_1', 'frontview_relu1_1', 'frontview_conv1_2', 'frontview_relu1_2', 'frontview_pool1',

        'frontview_conv2_1', 'frontview_relu2_1', 'frontview_conv2_2', 'frontview_relu2_2', 'frontview_pool2',

        'frontview_conv3_1', 'frontview_relu3_1', 'frontview_conv3_2', 'frontview_relu3_2', 'frontview_conv3_3',
        'frontview_relu3_3', 'frontview_pool3',

        'frontview_conv4_1', 'frontview_relu4_1', 'frontview_conv4_2', 'frontview_relu4_2', 'frontview_conv4_3',
        'frontview_relu4_3'
    )
    # output of retrained layer for vgg
    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[10:14]
        if kind == 'conv':
            if name == 'frontview_conv4_1':
                # Modify the senventh conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index = random.sample(range(512), 256)
                kernels = kernels[:, :, :, sample_index]
                bias = bias[:, sample_index]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'frontview_conv4_2':
                # Modify the eighth conv layer
                # pdb.set_trace()
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            elif name == 'frontview_conv4_3':
                # pdb.set_trace()
                # Modify the ninth conv layer
                kernels, bias = weights[i][0][0][0][0]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                sample_index_1 = random.sample(range(512), 256)
                sample_index_2 = random.sample(range(512), 256)
                kernels = kernels[:, :, sample_index_1, :]
                kernels = kernels[:, :, :, sample_index_2]
                bias = bias[:, sample_index_2]
                kernels = utils.get_variable(kernels, name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
            else:
                kernels, bias = weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
                bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
                current = utils.conv2d_basic(current, kernels, bias, keep_prob)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current
    # pdb.set_trace()
    return net

# 3D proposal network without rgbview input
def Proposal_net(birdview, frontview, model_dir, MODEL_URL, debug, keep_prob):
    """ 3D region proposal network """
    # input birdview, dropout probability, weight of vgg, ground-truth labels, ground-truth regression value,
    # anchor classification mask and anchor regression mask
    # The birdview has more than three channel, thus we need to train first two conv layers in vgg-16
    MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(model_dir, MODEL_URL)
    weights = np.squeeze(model_data['layers'])
    with tf.name_scope("birdview-Vgg-16"):
        birdview_net = vgg_net_birdview(weights, birdview, debug, keep_prob)
        current = birdview_net["birdview_relu4_3"]
    # upsample, output 256 channels
    with tf.name_scope("birdview_Upsample_layer"):
        kernels = utils.weight_variable([3, 3, 256, 256], name="birdview_upsample_w")
        bias = utils.bias_variable([256], name="birdview_upsample_b")
        output_shape = current.get_shape().as_list()
        output_shape[1] *= 4
        output_shape[2] *= 4
        output_shape[3] = kernels.get_shape().as_list()[2]
        birdview_net['birdview_upsample'] = utils.conv2d_transpose_strided(current, kernels, bias,
                                                                           output_shape=output_shape, stride=4,
                                                                           name='birdview_upsample',
                                                                           keep_prob=keep_prob)
        current = birdview_net['birdview_upsample']
        if debug:
            utils.add_activation_summary(current)
        # vgg-frontview
    with tf.name_scope("frontview-Vgg-16"):
        frontview_net = vgg_net_frontview(weights, frontview, debug, keep_prob)
        current = frontview_net["frontview_relu4_3"]
        # pdb.set_trace()
        # upsample, output 256 channels
    with tf.name_scope("frontview_Upsample_layer"):
        kernels = utils.weight_variable([3, 3, 256, 256], name="frontview_upsample_w")
        bias = utils.bias_variable([256], name="frontview_upsample_b")
        output_shape = current.get_shape().as_list()
        output_shape[1] *= 4
        output_shape[2] *= 4
        output_shape[3] = kernels.get_shape().as_list()[2]
        frontview_net['frontview_upsample'] = utils.conv2d_transpose_strided(current, kernels, bias,
                                                                             output_shape=output_shape, stride=4,
                                                                             name='frontview_upsample',
                                                                             keep_prob=keep_prob)
        current = frontview_net['frontview_upsample']
        if debug:
            utils.add_activation_summary(current)

    return birdview_net, frontview_net

def region_pooling(birdview_feat, frontview_feat, birdview_rois, frontview_rois, birdview_rois_ind, frontview_rois_ind, ROI_H, ROI_W, debug):
    # dynamic region pooling
    birdview_channel = birdview_feat.get_shape().as_list()[3]
    frontview_channel = frontview_feat.get_shape().as_list()[3]
    birdview_region_list = []
    frontview_region_list = []

    birdview_pooling_ROI = tf.image.crop_and_resize(birdview_feat, birdview_rois, birdview_rois_ind, [ROI_H, ROI_W],
                                                    name='birdview_pooling_ROI')
    frontview_pooling_ROI = tf.image.crop_and_resize(frontview_feat, frontview_rois, frontview_rois_ind, [ROI_H, ROI_W],
                                                     name='frontview_pooling_ROI')
    if debug:
        utils.add_activation_summary(birdview_pooling_ROI)
        utils.add_activation_summary(frontview_pooling_ROI)

    return birdview_pooling_ROI, frontview_pooling_ROI

# fusion network
def region_fusion_net(birdview_region, frontview_region, NUM_OF_REGRESSION_VALUE, ROI_H, ROI_W):
    # flat
    birdview_flatregion = tf.reshape(birdview_region, [-1, ROI_W * ROI_H * 256], name='birdview_flatregion')
    frontview_flatregion = tf.reshape(frontview_region, [-1, ROI_W * ROI_H * 256], name='frontview_flatregion')

    with tf.name_scope("fusion-1"):
        # first fusion
        # feature transformation is implemented by fully connected netwok
        joint_1 = utils.join(birdview_flatregion, frontview_flatregion, name='joint_1')
        fusion_birdview_1 = utils.fully_connected(joint_1, 1024, name='fusion_birdview_1')
        fusion_frontview_1 = utils.fully_connected(joint_1, 1024, name='fusion_frontview_1')

    with tf.name_scope("fusion-2"):
        # second fusion
        joint_2 = util.join(fusion_birdview_1, fusion_frontview_1, name='joint_2')
        fusion_birdview_2 = utils.fully_connected(joint_2, 1024, name='fusion_birdview_2')
        fusion_frontview_2 = utils.fully_connected(joint_2, 1024, name='fusion_frontview_2')

    with tf.name_scope("fusion-3"):
        # third fusion
        joint_3 = utils.join(fusion_birdview_2, fusion_frontview_2, name='joint_3')
        fusion_birdview_3 = utils.fully_connected(joint_3, 1024, name='fusion_birdview_3')
        fusion_frontview_3 = utils.fully_connected(joint_3, 1024, name='fusion_frontview_3')

    with tf.name_scope("fusion-4"):
        joint_4 = utils.join(fusion_birdview_3, fusion_frontview_3, name='joint_4')
        logits_cls = utils.fully_connected(joint_4, 2, name='fusion_cls_4', relu=False)
        logits_reg = utils.fully_connected(joint_4, NUM_OF_REGRESSION_VALUE, name='fusion_reg_4', relu=False)

    return logits_cls, logits_reg

# l2 loss for regression
def l2_loss(s, t):
    """L2 loss function."""
    d = s - t
    x = d * d
    loss = tf.reduce_sum(x, 1)
    return loss

# MV3D network comprised of 3D proposal network and region fusion network
def MV3D(birdview, frontview, cls_mask, reg_mask, gt_ROI_labels, gt_ROI_regs, birdview_rois, frontview_rois, birdview_box_ind, frontview_box_ind, \
         ROI_H, ROI_W, NUM_OF_REGRESSION_VALUE, model_dir, weight, reg_weight, debug, keep_prob = 1.0):

    MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-16.mat'

    with tf.name_scope("3D-Proposal-net"):
        birdview_net, frontview_net = Proposal_net(birdview, frontview, model_dir, MODEL_URL, debug, keep_prob)
