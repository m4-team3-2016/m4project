#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  4 21:11:13 2017

@author: group 3
"""

import numpy as np
import configuration as conf
import glob
from skimage import morphology
import drawBoundingBox as dbb

def getConnectedComponents(img_mask):
    # Binarize the image
    img_mask = np.where(img_mask > 1, 1, 0)
    # Detect connected components and assign an ID to each one
    connected_components_img = morphology.label(img_mask, background=0) 
    return connected_components_img
    
    
def getLabelCoordinates(img, connected_components_img, idx):
    # Find where labels in connected_components_img are equal to idx
    return np.where(connected_components_img == idx)
    
    
def drawAllBoundingBoxes(img_color, coordinates, infractionIDList):
    # Draw all bounding boxes in scene
    topLeft = (min(coordinates[1]),min(coordinates[0]))
    bottomRight = (max(coordinates[1]),max(coordinates[0]))
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
    
    # Load the mask
    folderGT = conf.folders["TrafficGT"]
    framesFiles = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)
    img_mask = cv2.imread(framesFiles[47])
    
    # Compute the connected components
    cc = getConnectedComponents(img_mask)
    
    # Get the pixel coordinates of the first connected component
    position = getLabelCoordinates(img_mask, cc, 1)
    
    # Load the color image
    folder = conf.folders["Traffic"]
    framesFiles = sorted(glob.glob(folder + '*'))
    nFrames = len(framesFiles)
    img_color = cv2.imread(framesFiles[47])
    
    # Draw all bounding boxes
    # Cars 1 and 5 are infringing the law!
    infractionIDList = [2, 5]
    oimg = drawAllBoundingBoxes(img_color, position, infractionIDList)
    
    # Save the output image
    cv2.imwrite('oimg.png',oimg)