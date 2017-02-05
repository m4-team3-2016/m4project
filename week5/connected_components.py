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

    currentFrame = 44

    # Load the mask
    folderGT = conf.folders["HighwayGT"]
    framesFiles = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)
    img_mask = cv2.imread(framesFiles[currentFrame])


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

    # Parameters
    isFirstFrame = True
    id = 1
    detectedObjects = []
    currentFrame

    # Reset onScreenValues to detect which objects are in the current image
    for element in detectedObjects:
        element.setVisibleOnScreen(False)

    # Find elements in list, if there are no elements, we should create them.
    # Creating objects is only necessary for the first frame.
    if isFirstFrame:
        # Compute the connected components
        cc = getConnectedComponents(img_mask)
        # Get the pixel coordinates of the first connected component
        indexes = getLabelCoordinates(img_mask, cc, 1)
        for idx in range(1,cc.max()+1):
            indexes = getLabelCoordinates(img_mask, cc, idx)
            topLeft = (min(indexes[1]), min(indexes[0]))
            bottomRight = (max(indexes[1]), max(indexes[0]))
            detectedObject = dO.detection(idx, currentFrame, topLeft, bottomRight, indexes)
            detectedObjects.append(detectedObject)


    secondFrame = currentFrame+1

    # Load the mask
    img_mask = cv2.imread(framesFiles[secondFrame])

    # Compute the connected components
    cc = getConnectedComponents(img_mask)

    for idx in range(1,cc.max()+1):
        # Get the pixel coordinates of the first connected component
        indexes = getLabelCoordinates(img_mask, cc, idx)
        isFound = False
        topLeft = (min(indexes[1]), min(indexes[0]))
        bottomRight = (max(indexes[1]), max(indexes[0]))
        centroid = [sum(indexes[0]) / len(indexes[0]), sum(indexes[1]) / len(indexes[1])]

        indexObjectNumber = 0
        for element in detectedObjects:
            prediction = element.kalmanFilter.predictKalmanFilter()
            distance = element.computeDistance(centroid, prediction)
            if(distance < 10):
                element.kalmanFilter.updateMeasurement(centroid)
                prediction = element.kalmanFilter.predictKalmanFilter()
                indexObjectNumber = element.detectionID
                element.setVisibleOnScreen(True)
                isFound = True
                print "Kalman prediction: " + str(prediction[0]) + " - " + str(prediction[1])
                print "Real values:       " + str(centroid[0]) + " - " + str(centroid[1])
                break;

        if not isFound:
            indexObjectNumber = detectedObjects.__len__()+1
            detectedObject = dO.detection(indexObjectNumber, secondFrame, topLeft, bottomRight, indexes)
            detectedObjects.append(detectedObject)

        #Print Bounding boxes in image. Based on the indexObjectNumber detected or created
        indexObjectNumber


    # CHECK THAT !!!!!!!!!!!!!!!!!!!!!!!!!!

    # Find detectedObjects that are not showing in image since 10 frames ago.
    # And remove them
    for element in detectedObjects:
        if not element.getVisibleOnScreen() and currentFrame > (element.getCurrentFrame+10):
            detectedObjects.remove(element)


    # return detectedObjects, img1, img2
