import cv2
import numpy as np
import week5configuration as finalConf
import glob
import sys
import dataReader
import detectionPipeline as pipeline

sys.path.append('../')

import sys
sys.path.append('../')
sys.path.append('../tools/')


if __name__ == "__main__":
    # Get mode from configuration file
    mode = finalConf.mode

    # Read the video/files and count images
    if mode == "video":
        print finalConf.inputData
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
    # Second stage: testingRange
    for idx in range(50,60):
        print "Reading frames " + str(idx-1) + " and " + str(idx)
        frame1,frame2 = dataReader.getFrameAndPrevious(data,idx)
        frame1Color,frame2Color = dataReader.getFrameAndPrevious(data,idx,False)
        #frame1 = pipeline.getObjectsFromFrame(frame1,mu,sigma,3)
        #frame2 = pipeline.getObjectsFromFrame(frame2,mu,sigma,3)

        res = np.concatenate((frame1,frame1Color),1)
        cv2.imshow("test",res)
        cv2.waitKey(0)
