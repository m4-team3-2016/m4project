import cv2
import numpy as np
import week5configuration as finalConf
import glob
import sys
import dataReader
import detectionPipeline as pipeline
import trackingObjects as tracking


sys.path.append('../')

import sys
sys.path.append('../')
sys.path.append('../tools/')


if __name__ == "__main__":
    # Get mode from configuration file
    mode = finalConf.mode

    # Read the video/files and count images
    if mode == "video":
        data = cv2.VideoCapture(finalConf.inputData)
        nFrames = int(data.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = data.get(cv2.CAP_PROP_FPS)
    else:
        if finalConf.inputData[-1] == '/':
            data = sorted(glog.glob(finalConf.inputData + "*.png"))
        else:
            data = sorted(glog.glob(finalConf.inputData + "/*.png"))
        nFrames = len(data)
        fps = 30.0 # Educated guess

    # First stage: Training
    trainingRange = range(int(finalConf.trainingPercentage * nFrames))
    testingRange = range(int(finalConf.trainingPercentage * nFrames),nFrames)

    mu,sigma = pipeline.getMuSigma(data,trainingRange)
    res = np.concatenate((mu,sigma),1)
    cv2.imwrite('musigma.png',res)
    # Second stage: testing
    testingRange = range(50,60)
    startingFrame = testingRange[0]
    for idx in testingRange:
        print "Reading frames " + str(idx-1) + " and " + str(idx)
        if idx == startingFrame:
            frame1,frame2 = dataReader.getFrameAndPrevious(data,idx)
            frame1Color,frame2Color = dataReader.getFrameAndPrevious(data,idx,False)
        else:
            print "we here lol"
            frame1 = frame2
            frame1Color = frame2Color
            frame2 = dataReader.getSingleFrame(data,idx)
            frame2Color = dataReader.getSingleFrame(data,idx,False)
        print frame1.shape, frame2.shape
        out1 = pipeline.getObjectsFromFrame(frame1,mu,sigma,3)
        out2 = pipeline.getObjectsFromFrame(frame2,mu,sigma,3)

        '''
        cv2.imshow('test',np.concatenate((frame1Color[:,:,0],frame1*255),1))
        cv2.waitKey(0)
        '''
        if idx == startingFrame:
            objectList, bbox1,bbox2 = tracking.computeTrackingBetweenFrames(True,[],idx,frame1Color,out1,frame2Color,out2)
        else:
            objectList, bbox1,bbox2 = tracking.computeTrackingBetweenFrames(False,objectList,idx,frame1Color,out1,frame2,out2)


        cv2.imshow("bbox2",bbox2)
        cv2.waitKey(0)
