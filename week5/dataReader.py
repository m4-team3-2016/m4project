import cv2
import week5configuration as finalConf
import numpy as np

def getSingleFrame(dataFile,frameNumber):
    if isinstance(dataFile,list):
        frame =  cv2.imread(list[frameNumber])
    else:
        dataFile.set(1,frameNumber)
        ret,frame = dataFile.read()
        cv2.imshow('tests',frame)
        cv2.waitKey(0)
        print "data read from video"
    print frame.shape
    if finalConf.colorSpace != "BGR":
        print "Changing colorspace"
        print type(frame)
        frame = cv2.cvtColor(frame,6) #finalConf.colorSpaceConversion[finalConf.colorSpace])
    return frame

def getFrameAndPrevious(dataFile,frameNumber):
    return getSingleFrame(dataFile,frameNumber),getSingleFrame(dataFile,frameNumber-1)

def getFrameAndNext(dataFile,frameNumber):
    return getSingleFrame(dataFile,frameNumber),getSingleFrame(dataFile,frameNumber+1)
