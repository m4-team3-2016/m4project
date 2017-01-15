import numpy as np
import cv2

def filterImage(im,P):

    if isinstance(im,str):
        im = cv2.imread(im,0)

    if im.shape[2] = 3:
        im = im[:,:,0]

    out,contours,hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    for c in contours:
        area = cv2.contourArea(c)
        if area <= P:
            unPaintThePixels = True
        else

    return im
