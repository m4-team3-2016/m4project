#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 21:11:13 2017

@author: group 3
"""
import sys
sys.path.append("../")
sys.path.append('../tools/')

import numpy as np
import configuration as conf
import glob
from skimage import morphology
import drawBoundingBox as dbb
import detectedObject as dO
import cv2.cv as cv


def getConnectedComponents(img_mask):
    # Binarize the image
    img_mask = np.where(img_mask > 1, 1, 0)
    # Detect connected components and assign an ID to each one
    connected_components_img = morphology.label(img_mask, background=0) 
    return connected_components_img
    
    
def getLabelCoordinates(img, connected_components_img, idx):
    # Find where labels in connected_components_img are equal to idx
    return np.where(connected_components_img == idx)
    
    
def drawAllBoundingBoxes(img_color, indexes, infractionIDList):
    # Draw all bounding boxes in scene
    topLeft = (min(indexes[1]),min(indexes[0]))
    bottomRight = (max(indexes[1]),max(indexes[0]))
    alpha = 0.5
    oimg = np.zeros(img_color.shape)
    
    for bb_idx in range(cc.max()):
        if bb_idx+1 in infractionIDList:
            color = (0,0,255)
        else:
            color = (0,255,0)
        oimg = dbb.drawBBoxWithText(img_color,'ID: ' + str(int(bb_idx+1)),topLeft,bottomRight,color,alpha)
    
    return oimg
    
                      
if __name__ == "__main__":
    
    import cv2

    startFrame = 44

    # Load the mask
    folderGT = conf.folders["HighwayGT"]
    framesFiles = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)
    img_mask = cv2.imread(framesFiles[startFrame])
    
    # Compute the connected components
    cc = getConnectedComponents(img_mask)
    
    # Get the pixel coordinates of the first connected component
    indexes = getLabelCoordinates(img_mask, cc, 1)
    
    # Load the color image
    # folder = conf.folders["Traffic"]
    # framesFiles = sorted(glob.glob(folder + '*'))
    # nFrames = len(framesFiles)
    # img_color = cv2.imread(framesFiles[startFrame])
    #
    # Draw all bounding boxes
    # Cars 1 and 5 are infringing the law!
    # infractionIDList = [2, 5]
    # oimg = drawAllBoundingBoxes(img_color, indexes, infractionIDList)
    #
    # Save the output image
    # cv2.imwrite('oimg.png',oimg)

    id = 1
    detectedObjects = []
    for idx in range(1,cc.max()+1):
        indexes = getLabelCoordinates(img_mask, cc, idx)
        topLeft = (min(indexes[1]), min(indexes[0]))
        bottomRight = (max(indexes[1]), max(indexes[0]))
        detectedObject = dO.detection(idx, startFrame, topLeft, bottomRight, indexes)
        detectedObjects.append(detectedObject)

########
    centroid = [sum(indexes[0]) / len(indexes[0]), sum(indexes[1]) / len(indexes[1])]
    kalman = cv.CreateKalman(4, 2, 0)
    # This happens only one time to initialize the kalman Filter with the first (x,y) point
    kalman.state_pre[0, 0] = centroid[0]
    kalman.state_pre[1, 0] = centroid[1]
    kalman.state_pre[2, 0] = 0
    kalman.state_pre[3, 0] = 0

    # set kalman transition matrix
    kalman.transition_matrix[0, 0] = 1
    kalman.transition_matrix[1, 1] = 1
    kalman.transition_matrix[2, 2] = 1
    kalman.transition_matrix[3, 3] = 1

    # set Kalman Filter
    cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
    cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(1e-5))  ## 1e-5
    cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(1e-1))
    cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(0.1))


    #
    secondFrame = startFrame+1

    # Load the mask
    img_mask = cv2.imread(framesFiles[secondFrame])

    # Compute the connected components
    cc = getConnectedComponents(img_mask)

    for idx in range(1,cc.max()+1):
        # Get the pixel coordinates of the first connected component
        indexes = getLabelCoordinates(img_mask, cc, idx)

        topLeft = (min(indexes[1]), min(indexes[0]))
        bottomRight = (max(indexes[1]), max(indexes[0]))
        centroid = [sum(indexes[0]) / len(indexes[0]), sum(indexes[1]) / len(indexes[1])]

        indexObjectsNumber = 0
        for element in detectedObjects:
            prediction = element.kalmanFilter.computePredictionKalmanFilter(centroid)
            # Kalman prediction with Kalman Correction with the points I have in trajectory_0000.txt
            kalman_prediction = cv.KalmanPredict(kalman)
            print "Kalman prediction: " + str(kalman_prediction[0,0]) + " - " + str(kalman_prediction[1,0])
            # print "CC:" + str(idx) + " Detected Object: " + str(indexObjectsNumber)
            print "Real values:       " + str( centroid[0] ) +  " - " + str( centroid[1] )
            # print "Kalman prediction: " + str(prediction[0]) + " - " + str(prediction[1])
            # print ""
            # indexObjectsNumber = indexObjectsNumber+1
