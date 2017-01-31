import cv2
import configuration as conf
import numpy as np
import glob
import sys
import os
import block_matching as match

#import Image
from PIL import Image

import scipy.misc
import evaluateOpticalFlow as of
import math
import matplotlib
from numpy.random import randn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import opticalFlowMethods as opticalflow
import evaluateOpticalFlow as evaluateOF


def expand_motion_matrix(motion_matrix, block_size=8):

    pixel_motion_matrix=np.zeros([motion_matrix.shape[0]*block_size,motion_matrix.shape[1]*block_size,2])
    
    for xx in range(motion_matrix.shape[0]):
        for yy in range(motion_matrix.shape[1]):
            pixel_motion_matrix[xx*block_size:xx*block_size+block_size,yy*block_size:yy*block_size+block_size,0] = motion_matrix[xx,yy,0] #x motion
            pixel_motion_matrix[xx*block_size:xx*block_size+block_size,yy*block_size:yy*block_size+block_size,1] = motion_matrix[xx,yy,1] #y motion
    
    return pixel_motion_matrix
'''
OFimage_exp = expand_motion_matrix(OFimage, conf.block_size, conf.area_size)
print(OFimage_exp.shape)    
print(OFimage.shape)    
print(OFgt.shape)
''' 
    
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

referenceImage = cv2.imread(framesFiles[0], -1)
referenceImageBW = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)

currentImage = cv2.imread(framesFiles[1], -1)
currentImageBW = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)

# Reshape from 1241 to 1240
referenceImageBW = referenceImageBW[:,1:]
currentImageBW = currentImageBW[:,1:]


# Apply block matching
OFimage = match.compute_block_matching(referenceImageBW, currentImageBW)
# OFimage = opticalflow.opticalFlowBW(referenceImageBW, currentImageBW)
# OFimage = opticalflow.LukasKanade(referenceImage, currentImage)

OFimage_exp = expand_motion_matrix(OFimage, conf.block_size)

GTpath = ID+'GT'
OFgt = cv2.imread(conf.folders[GTpath], -1)

# Reshape from 1241 to 1240
OFgt = OFgt[:,1:,:]


print('---Showing OFimage and OFgt shapes...')
print(OFimage_exp.shape)
print(OFgt.shape)

msenValues, error, image = of.msen(OFimage_exp, OFgt)
plt.hist(msenValues, bins=25, normed=True)
formatter = FuncFormatter(of.to_percent)
plt.gca().yaxis.set_major_formatter(formatter)
plt.xlabel('MSEN value')
plt.ylabel('Number of Pixels')
plt.title("%s Histogram. \n Percentage of Erroneous Pixels in Non-occluded areas (PEPN): %d %%" % (ID, error))
plt.show()
