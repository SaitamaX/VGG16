# prediction of region based fusion network

# data: a list of data
# data[0] = data_birdview
# data[1] = data_frontview
# data[2] = data_birdview_rois
# data[3] = data_frontview_rois
# data[4] = data_birdview_box_ind
# data[5] = data_frontview_box_ind

# validation_data has same structure of training_data

# The format of prepared data should be:

# data_birdview: Birdview generated from LIDAR point cloud. Fomat: .npy file. [number of images, row of a image, col of a image, channels]
# data_frontview: Frontview generated from LIDAR point cloud. Format: .npy file. [number of images, row of a image, col of a image, channels]

# data_birdview_rois: Region coordinate for birdview region pooling. Format: .npy file. [number of images * 200 regions, 4 (y1, x1, y2, x2)]
# data_frontview_rois: Region coordinate for frontview region pooling. Format: .npy file. [number of images * 200 regions, 4 (y1, x1, y2, x2)]

# data_birdview_box_ind: The value of data_birdview_box_ind[i]
# specifies the birdview that the i-th box refers to. Format: .npy file. [number of images * 200 regions]
# data_frontview_box_ind: The value of data_frontview_box_ind[i]
# specifies the frontview that the i-th box refers to. Format: .npy file [number of images * 200 regions]

import FusionNet
import numpy as np
import pdb
from os.path import expanduser

loadroot = expanduser("~") + '/Desktop/MV3D/Fusion-net/'

data_birdview = np.load(loadroot + 'birdview_set.npy')
data_frontview = np.load(loadroot + 'frontview_set.npy')

data_frontview_rois = np.load(loadroot + 'frontview_rois2.npy')
data_birdview_rois = np.load(loadroot + 'birdview_rois2.npy')

data_frontview_box_ind = np.zeros(len(data_frontview_rois))
data_birdview_box_ind = np.zeros(len(data_birdview_rois))

data.extend((data_birdview, data_frontview, data_birdview_rois, data_frontview_rois, data_birdview_box_ind, data_frontview_box_ind))

label, reg = FusionNet.prediction_network(data)


