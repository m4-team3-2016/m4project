# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 17:42:53 2017

@author: Sergi
"""

import cv2
import os
import numpy as np
from scipy import stats
import configuration as conf
import glob

# Original idea from:
# http://www.ee.cuhk.edu.hk/~jshao/papers_jshao/jshao_MovObjectDet_2012.pdf
# 2005-A RE-EVALUATION OF MIXTURE-OF-GAUSSIAN BACKGROUND MODELING.pdf

def bgr_to_rgs(img):
    # This is an implementation of the rgs method described by Elgammal et al. 
    # in "Non-parametric model for background subtraction." Computer Vision 751-767.
    # It expects to receive a BGR image.

    # First, extract the BGR components of the given image
    B=np.array(img[:,:,0], dtype=np.float32)
    G=np.array(img[:,:,1], dtype=np.float32)
    R=np.array(img[:,:,2], dtype=np.float32)
    
    # Compute the luminance as the sum of the previous channels
    sum_channels = np.zeros(img.shape, dtype=np.float32)
    sum_channels = B+G+R
    
    # Compute the normalized chromacity coordinates
    gp = np.divide(G, sum_channels, dtype=np.float32)
    rp = np.divide(R, sum_channels, dtype=np.float32)
    
    # Compute the luminance
    I = np.divide(sum_channels, 3, dtype=np.float32)
    
    # Map from [0,1] values to [0,255]
    rint = rp * 255.0
    gint = gp * 255.0

    # Generate the image in uint8 format
    rgs = np.array([rint,gint,I], dtype=np.uint8)
    RGS = np.array([R,G,I], dtype=np.uint8)
    
    # Convert the output image shape to (width,height,channels)
    rgs = rgs.transpose(1,2,0)
    RGS = RGS.transpose(1,2,0)
    
    # Turn to black the bottom part of the image
    rgs[(int(rgs.shape[0]/1.5)):,:,:] = 0
    
    return rgs, RGS
    
    
def rgs_thresholding(rgs, th=np.array([0,27])): #0,27 50,76
    rows = len(rgs)
    cols = len(rgs[0])
    channels = 3
    filtered_image = np.zeros([rows,cols,channels])
    
    # Extract the shadow map by thesholding the S component of the rgs image 
    for x in range(rows): # for each row
        for y in range(cols): # for each column
            for c in range(channels): # for each color channel
                if rgs[x,y,2] >= th[0] and rgs[x,y,2] < th[1]:
                    filtered_image[x,y,c] = rgs[x,y,c]
    
    return filtered_image

   
def generate_mask_and_inpaint_shadows(original_img, filtered_img):
    rows = len(filtered_img)
    cols = len(filtered_img[0])
    channels = 3
    
    # Generate the mask to inpaint given the shadow mask
    mask_to_inpaint = np.array(np.zeros([rows,cols]), dtype=np.uint8)
    for x in range(rows): # for each row
        for y in range(cols): # for each column
            for c in range(channels): # for each color channel
                if filtered_img[x,y,c] > 0:
                    mask_to_inpaint[x,y] = 255
                else:
                    mask_to_inpaint[x,y] = 0
    
    # Apply a dilation to increase the area to inpaint
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
    mask_to_inpaint = cv2.dilate(mask_to_inpaint,kernel,iterations = 3)
    
    # Perform an image inpainting
    inpaint = cv2.inpaint(original_img,mask_to_inpaint,3,cv2.INPAINT_TELEA)
    
    return inpaint

  
# Remove the shadows on an image given the input image
def shadow_removal(frame):
    # Convert from BGR to RGS/rgS color space
    rgs, RGS = bgr_to_rgs(frame)
    
    # Compute the shadow mask given the rgS image and a threshold
    filtered_img = rgs_thresholding(rgs)
    cv2.imwrite('test/filtered_img.png', filtered_img) 
    
    # Inpaint the regions corresponding to the shadows
    inpainted_img = generate_mask_and_inpaint_shadows(frame, filtered_img)
    
    return inpainted_img


# Remove the shadows on an image given the input image [0,255] and a mask [0,1]
def inmask_shadow_removal(frame, mask):
    
    # Apply the mask to the original image
    # filter_img = frame[:,:,:] * np.divide(mask, 255.0, dtype=np.float32)
    
    
    # If the mask is 2d
    filter_img = np.zeros([frame.shape[0],frame.shape[1],frame.shape[2]])
    filter_img[:,:,0] = frame[:,:,0] * mask # np.divide(mask, 255.0, dtype=np.float32)
    filter_img[:,:,1] = frame[:,:,1] * mask # np.divide(mask, 255.0, dtype=np.float32)
    filter_img[:,:,2] = frame[:,:,2] * mask # np.divide(mask, 255.0, dtype=np.float32)
    
    # Convert from BGR to RGS/rgS color space
    filter_img, RGS = bgr_to_rgs(filter_img)
    
    # Threshold the S channel corresponding to the luminance in order to remove the shadows
    filter_img = rgs_thresholding(filter_img, np.array([0,25]))
    
    # Generate the output mask
    # if 3d
    # output_mask =  (filter_img[:,:,:] == 0) * mask[:,:,:]
    # if 2d
    output_mask = np.zeros([frame.shape[0],frame.shape[1]])
    output_mask[:,:] =  (filter_img[:,:,0] == 0) * mask[:,:]
    #output_mask[:,:,1] =  (filter_img[:,:,1] == 0) * mask[:,:]
    #output_mask[:,:,2] =  (filter_img[:,:,2] == 0) * mask[:,:]
    
    return output_mask


if __name__ == "__main__":
    dataset    = "Highway"
    datasetGT  = "HighwayGT"
    ID = dataset
    IDGT = datasetGT
    folder = conf.folders[ID]
    folderGT = conf.folders[IDGT]
    framesFiles   = sorted(glob.glob(folder + '*'))
    framesFilesGT = sorted(glob.glob(folderGT + '*'))
    nFrames = len(framesFiles)
    
    background = cv2.imread(framesFiles[0])
    original = cv2.imread(framesFiles[1200])
    mask = cv2.imread(framesFilesGT[0])
    mask = mask[:,:,0]
    mask = mask / 255.0
    

    # Look for the best threshold
    # inpaint = shadow_removal(original)
    output_mask = inmask_shadow_removal(original,mask)
    
 
    # cv2.imwrite('test/inpaint.png', inpaint)
    # cv2.imwrite('test/filter_img.png', filter_img) 
    # cv2.imwrite('test/background.png', background)  
    # cv2.imwrite('test/original.png', original)  
    cv2.imwrite('test/input_mask_2d.png', mask)  
    cv2.imwrite('test/output_mask.png', output_mask) 
    # cv2.imwrite('test/mask_to_inpaint.png', mask_to_inpaint)
    # cv2.imwrite('test/rgs.png', rgs)
    # cv2.imwrite('test/RGSM.png', RGS)
    # cv2.imwrite('test/'+str(th[0])+'_'+str(th[1])+'.png', filtered_img)