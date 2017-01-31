import cv2
import configuration as conf
import numpy as np
import glob
import sys
import os
import block_matching as match
import Image
import scipy.misc
import evaluateOpticalFlow as of
import math
import matplotlib
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import opticalFlowMethods as opticalflow
import evaluateOpticalFlow as evaluateOF

sys.path.append('../')
operativeSystem = os.name
(CVmajor, CVminor, _) = cv2.__version__.split(".")


ID = '45'
resultsPath = "./resultsSequence" + ID + '/'
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

folder = conf.folders[ID]
block_size = conf.block_size
area_size = conf.area_size
compensation = conf.compensation

folder = conf.folders[ID]
framesFiles = sorted(glob.glob(folder + '*.png'))
nFrames = len(framesFiles)

referenceImage = cv2.imread(framesFiles[0])
referenceImageBW = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)

currentImage = cv2.imread(framesFiles[1])
currentImageBW = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)


# Apply block matching
# OFimage = match.compute_block_matching(referenceImageBW, currentImageBW)
OFimage = opticalflow.opticalFlowBW(referenceImageBW, currentImageBW)

GTpath = ID+'GT'
OFgt = cv2.imread(conf.folders[GTpath], -1)

msenValues, error, image = of.msen(OFimage, OFgt)
plt.hist(msenValues, bins=25, normed=True)
formatter = FuncFormatter(of.to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.xlabel('MSEN value')
plt.ylabel('Number of Pixels')
plt.title("%s Histogram. \n Percentage of Erroneous Pixels in Non-occluded areas (PEPN): %d %%" % (ID, error))
plt.show()