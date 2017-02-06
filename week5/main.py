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
    ID = finalConf.ID
    path = finalConf.folders[ID]
    alpha = finalConf.OptimalAlphaParameter[ID]
    rho = finalConf.OptimalRhoParameter[ID]

    # Read the video/files and count images
    if ID == "Video":
        data = cv2.VideoCapture(path)
        nFrames = int(data.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = data.get(cv2.CAP_PROP_FPS)
    else:
        if path[-1] == '/':
            data = sorted(glob.glob(path + "*.jpg"))
        else:
            data = sorted(glob.glob(path + "/*.jpg"))
        nFrames = len(data)
        fps = 30.0 # Educated guess

    # First stage: Training
    trainingRange = range(int(finalConf.trainingPercentage[ID] * nFrames))
    testingRange = range(int(finalConf.trainingPercentage[ID] * nFrames),nFrames)

    mu,sigma = pipeline.getMuSigma(data,trainingRange)
    # res = np.concatenate((mu,sigma),1)
    # cv2.imwrite('musigma.png',res)
    
    # Second stage: testing
    startingFrame = testingRange[0]
    for idx in testingRange:
        print "Reading frames " + str(idx-1) + " and " + str(idx)
        if idx == startingFrame:
            frame2,frame1 = dataReader.getFrameAndPrevious(data,idx)
            originalFrame2,originalFrame1 = dataReader.getFrameAndPrevious(data,idx,False)
        else:
            frame1 = frame2
            originalFrame1 = originalFrame2
            frame2 = dataReader.getSingleFrame(data,idx)
            originalFrame2 = dataReader.getSingleFrame(data,idx,False)
        print frame1.shape, frame2.shape
        out1, mu, sigma = pipeline.getObjectsFromFrame(frame1,mu,sigma,alpha, rho)
        out2, mu, sigma = pipeline.getObjectsFromFrame(frame2,mu,sigma,alpha, rho)
        # Third stage: Stabilizate

        '''
        cv2.imshow('test',np.concatenate((frame1Color[:,:,0],frame1*255),1))
        cv2.waitKey(0)
        '''
        if idx == startingFrame:
            objectList, bbox1, bbox2 = tracking.computeTrackingBetweenFrames(True,[],idx,originalFrame1,out1,originalFrame2,out2)
            cv2.imshow("bbox1", bbox1)
            cv2.waitKey(0)
        else:
            objectList, bbox1, bbox2 = tracking.computeTrackingBetweenFrames(False,objectList,idx,originalFrame1,out1,originalFrame2,out2)


        cv2.imshow("bbox2",bbox2)
        cv2.waitKey(0)
