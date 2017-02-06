import cv2
import numpy as np
import week5configuration as finalConf
import glob
import sys
import dataReader
import detectionPipeline as pipeline
import trackingObjects as tracking
import stabizateFrames as stFrame


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
    testingRange = range(int(finalConf.trainingPercentage[ID] * nFrames)+1,nFrames)

    print "Starting training.. This takes too long. For faster computing goes to stabizateFrames.py line 160 and discomment"
    mu,sigma, lastStabilizedFrame = pipeline.getMuSigma(data,trainingRange)
    # res = np.concatenate((mu,sigma),1)
    # cv2.imwrite('musigma.png',res)

    # Second stage: testing
    startingFrame = testingRange[0]
    for idx in testingRange:
        print "Reading frames " + str(idx) + " and " + str(idx+1)
        if idx == startingFrame:
            originalFrame2,originalFrame1 = dataReader.getFrameAndPrevious(data,idx,False)
            # If we are doing Traffic or Highway we take the last stabilizated sample.
            if ID is not 'Own':
                originalFrame1S = stFrame.stabilizatePairOfImages(lastStabilizedFrame, originalFrame1)
                originalFrame2S = stFrame.stabilizatePairOfImages(originalFrame1, originalFrame2)
            else:
                originalFrame1S = originalFrame1
                originalFrame2S = stFrame.stabilizatePairOfImages(originalFrame1, originalFrame2)
            # Convert images to INT
            frame1 = cv2.cvtColor(originalFrame1S.astype(np.uint8), finalConf.colorSpaceConversion[finalConf.colorSpace])
            frame2 = cv2.cvtColor(originalFrame2S.astype(np.uint8), finalConf.colorSpaceConversion[finalConf.colorSpace])
            frame1 = frame1[finalConf.area_size:frame1.shape[0]-finalConf.area_size,finalConf.area_size:frame1.shape[1]-finalConf.area_size]
            frame2 = frame2[finalConf.area_size:frame2.shape[0] - finalConf.area_size, finalConf.area_size:frame2.shape[1] - finalConf.area_size]
        else:
            frame1 = frame2
            originalFrame1S = originalFrame2S
            originalFrame2 = dataReader.getSingleFrame(data,idx,False)
            originalFrame2S = stFrame.stabilizatePairOfImages(originalFrame1S, originalFrame2)
            frame2 = cv2.cvtColor(originalFrame2S.astype(np.uint8),finalConf.colorSpaceConversion[finalConf.colorSpace])
            frame2 = frame2[finalConf.area_size:frame2.shape[0] - finalConf.area_size, finalConf.area_size:frame2.shape[1] - finalConf.area_size]
        # print frame1.shape, frame2.shape
        out1, mu, sigma = pipeline.getObjectsFromFrame(frame1,mu,sigma,alpha, rho)
        out2, mu, sigma = pipeline.getObjectsFromFrame(frame2,mu,sigma,alpha, rho)
        originalFrame1SZoom = originalFrame1S[finalConf.area_size:originalFrame1S.shape[0] - finalConf.area_size, finalConf.area_size:originalFrame1S.shape[1] - finalConf.area_size]
        originalFrame2SZoom = originalFrame2S[finalConf.area_size:originalFrame2S.shape[0] - finalConf.area_size, finalConf.area_size:originalFrame2S.shape[1] - finalConf.area_size]

        '''
        cv2.imshow('test',np.concatenate((frame1Color[:,:,0],frame1*255),1))
        cv2.waitKey(0)
        '''
        if idx == startingFrame:
            objectList, bbox1, bbox2 = tracking.computeTrackingBetweenFrames(True,[],idx,originalFrame1SZoom,out1,originalFrame2SZoom,out2)
            res = np.concatenate((bbox1,np.stack([out1*255, out1*255, out1*255], axis=-1)),1)
            cv2.imwrite("./results/Image_" + str(idx) + '.png', res)
            # cv2.imshow("Image " + str(idx), res)
            # cv2.waitKey(0)
        else:
            objectList, bbox1, bbox2 = tracking.computeTrackingBetweenFrames(False,objectList,idx,originalFrame1SZoom,out1,originalFrame2SZoom,out2)

        res = np.concatenate((bbox2, np.stack([out2 * 255, out2 * 255, out2 * 255], axis=-1)), 1)
        cv2.imwrite("./results/Image_" + str(idx+1)+'.png', res)
        # cv2.imshow("Image " + str(idx), res)
        # cv2.waitKey(0)
