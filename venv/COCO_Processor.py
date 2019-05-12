from pycocotools.coco import COCO
import pylab
# GUI for select training picture
import tkinter as TK
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

import skimage.io as io
import numpy as np
import time

def _quit(root):
    root.quit()
    root.destroy()

def _save():


dataDir = 'E:/xbw/CocoData'
dataType = 'train2017'
annFile = '{0}/annotations/instances_{1}.json'.format(dataDir, dataType)

coco = COCO(annFile)

cats = coco.loadCats(coco.getCatIds())
catIds = coco.getCatIds(catNms=['cat', 'dog'])
imgIds = coco.getImgIds(catIds=catIds)

root = TK.Tk()
root.title("Select Picture")
fig = Figure(figsize=(20,20), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
quitButton = TK.Button(master=root, test='Quit', command=lambda:_quit(root))

root.mainloop()


for imgId in imgIds:
    imgArray = coco.loadImgs(imgId)
    imgDict = imgArray[0]
    I = io.imread('{0}/{1}/{2}'.format(dataDir, dataType, imgDict['file_name']))
    plt.axis('off')
    plt.imshow(I)
    annIds = coco.getAnnIds(imgIds=imgDict['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    coco.showAnns(anns)
    plt.pause(1)
    plt.clf()

