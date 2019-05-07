from pycocotools.coco import COCO
import pylab
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import skimage.io as io
import numpy as np

pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir = 'E:/xbw/CocoData'
dataType = 'train2017'
annFile = '{0}/annotations/instances_{1}.json'.format(dataDir, dataType)

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
catIds = coco.getCatIds(catNms=['person'])
imgIds = coco.getImgIds(catIds=catIds)

for imgId in imgIds:
    imgArray = coco.loadImgs(imgId)
    imgDict = imgArray[0]
    I = io.imread('{0}/{1}/{2}'.format(dataDir, dataType, imgDict['file_name']))
    plt.axis('off')
    plt.imshow(I)
    plt.show()
    plt.pause(1)

