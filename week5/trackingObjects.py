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
import cv2
import inspect
#import cv2.cv as cv


def getConnectedComponents(img_mask):
    # Binarize the image
    if img_mask.max()> 1:
        img_mask = np.where(img_mask > 1, 1, 0)
    # Detect connected components and assign an ID to each one
    connected_components_img = morphology.label(img_mask, background=0)
    return connected_components_img


def getLabelCoordinates(connected_components_img, idx):
    # Find where labels in connected_components_img are equal to idx
    return np.where(connected_components_img == idx)


def drawCurrentBoundingBox(img_color, index, topLeft, bottomRight, isInfracting=False):
    alpha = 0.5
    if isInfracting:
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    # Get the bbox image
    cv2.imwrite("testBefore.png", img_color)
    bb_mask = dbb.drawBBoxWithText(img_color, 'ID: ' + str(int(index)), topLeft, bottomRight, color, alpha)
    # Find pixels where the object is
    coords = np.where(bb_mask <> 0)
    # Replace non-zero pixels in the current image by the bb_mask pixel intensities
    oimg = img_color.copy()
    oimg[coords] = 0
    oimg = oimg + bb_mask
    cv2.imwrite("after.png", oimg)
    return oimg


def computeTrackingBetweenFrames(isFirstFrame, detectedObjects, frameID, img1, imgMask1, img2, imgMask2):

    # Find elements in list, if there are no elements, we should create them.
    # Creating objects is only necessary for the first frame.
    if isFirstFrame:
        # Compute the connected components
        cc = getConnectedComponents(imgMask1)
        # Get the pixel coordinates of the first connected component
        indexes = getLabelCoordinates(cc, 1)
        for idx in range(1,cc.max()+1):
            indexes = getLabelCoordinates(cc, idx)
            topLeft = (min(indexes[1]), min(indexes[0]))
            bottomRight = (max(indexes[1]), max(indexes[0]))
            detectedObject = dO.detection(idx, frameID, topLeft, bottomRight, indexes)
            detectedObjects.append(detectedObject)
            # Print Bounding boxes in image. Based on the indexObjectNumber detected or created
            isInfracting = False
            img1 = drawCurrentBoundingBox(img1, idx, topLeft, bottomRight, isInfracting)

    # Reset onScreenValues to detect which objects are in the current image
    for element in detectedObjects:
        element.setVisibleOnScreen(False)
    secondFrame = frameID+1
    # Compute the connected components
    cc = getConnectedComponents(imgMask2)

    for idx in range(1,cc.max()+1):
        # Get the pixel coordinates of the first connected component
        indexes = getLabelCoordinates(cc, idx)
        isFound = False
        topLeft = (min(indexes[1]), min(indexes[0]))
        bottomRight = (max(indexes[1]), max(indexes[0]))
        centroid = [sum(indexes[0]) / len(indexes[0]), sum(indexes[1]) / len(indexes[1])]

        indexObjectNumber = 0

        # Kalman Filter
        for element in detectedObjects:
            if element.getVisibleOnScreen():
                continue;

            prediction = element.kalmanFilter.predictKalmanFilter()
            distance = element.computeDistance(centroid, prediction)
            # print "Distance between centroids:       " + str(distance)
            if(distance < 10):
                element.kalmanFilter.updateMeasurement(centroid)
                prediction = element.kalmanFilter.predictKalmanFilter()
                indexObjectNumber = element.detectionID
                element.setVisibleOnScreen(True)
                isFound = True
                # print "Kalman prediction: " + str(prediction[0]) + " - " + str(prediction[1])
                # print "Real values:       " + str(centroid[0]) + " - " + str(centroid[1])
                break;

        # Coherence Between Centroids
        # for element in detectedObjects:
        #     if element.getVisibleOnScreen():
        #         continue;
        #     previousCentroid = element.centroid
        #     if element.isVectorValid(element.getVector(previousCentroid, centroid)):
        #         indexObjectNumber = element.detectionID
        #         element.setVisibleOnScreen(True)
        #         isFound = True
        #         break;

        if not isFound:
            indexObjectNumber = detectedObjects[detectedObjects.__len__()-1].detectionID + 1
            detectedObject = dO.detection(indexObjectNumber, secondFrame, topLeft, bottomRight, indexes)
            detectedObjects.append(detectedObject)

        #Print Bounding boxes in image. Based on the indexObjectNumber detected or created
        isInfracting = False
        img2 = drawCurrentBoundingBox(img2, indexObjectNumber, topLeft, bottomRight, isInfracting)

    # Find detectedObjects that are not showing in image since 10 frames ago.
    # And remove them
    for element in detectedObjects:
        if not element.getVisibleOnScreen() and frameID > (element.getCurrentFrame() + 10):
            detectedObjects.remove(element)

    return detectedObjects, img1, img2


if __name__ == "__main__":

    import cv2

    frameID = 44
    secondFrame = frameID+1

    # Load the mask
    folderGT = conf.folders["HighwayGT"]
    folder = conf.folders["Highway"]
    framesFilesGT = sorted(glob.glob(folderGT + '*'))
    framesFiles   = sorted(glob.glob(folder + '*'))
    nFrames = len(framesFilesGT)
    img1 = cv2.imread(framesFiles[frameID])
    img2 = cv2.imread(framesFiles[frameID+1])
    imgMask1 = cv2.imread(framesFilesGT[frameID])
    imgMask2 = cv2.imread(framesFilesGT[frameID+1])

    # Load the mask
    detectedObjects = []
    # cv2.imwrite('testingBefore.png', img2)
    detectedObjects, img1, img2 = computeTrackingBetweenFrames(True, detectedObjects, frameID, img1, imgMask1, img2, imgMask2)

    cv2.imshow("img1", img1)
    cv2.waitKey(0)
    cv2.imshow("img2", img2)
    cv2.waitKey(0)
