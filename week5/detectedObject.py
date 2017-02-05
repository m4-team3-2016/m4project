import numpy as np
import sys
sys.path.append('../')
import configuration as conf
import KalmanFilterClass as kf


class detection:

    def checkWidthHeight(self,tl,br):
        try:
            w = br[0] - tl[0]
            assert w > 0, "Object with id " + str(self.id) + " has impossible height: " + str(w)
        except AssertionError, e:
            raise Exception(e.args)
        try:
            h = br[1] - tl[1]
            assert h > 0,"Object with id " + str(self.id) + " has impossible height: " + str(h)
        except AssertionError, e:
            raise Exception(e.args)
        self.width = w
        self.height = h


    def __init__(self,detectionID,startFrame,topLeft,bottomRight,indexes):
        self.detectionID = detectionID
        self.startFrame = startFrame
        self.currentFrame = startFrame
        self.frames = [self.currentFrame]
        self.checkWidthHeight(topLeft,bottomRight)
        self.onScreen = True
        self.topLeft = topLeft
        self.bottomRight = bottomRight
        self.centroid = sum(indexes[0])/len(indexes[0]),sum(indexes[1])/len(indexes[1])
        self.centroids = [sum(indexes[0])/len(indexes[0]),sum(indexes[1])/len(indexes[1])]

    def update(self,currentFrame,topLeft,bottomRight,indexes):
        self.currentFrame = currentFrame
        self.frames.append(currentFrame)
        self.centroid = (sum(indexes[0])/len(indexes[0]),sum(indexes[1])/len(indexes[1]))
        self.centroids.append(self.centroid)

        self.checkWidthHeight(topLeft,bottomRight)
        # The white pixels position
        self.topLeft = topLeft
        self.bottomRight = bottomRight
    def comet(self,image,color,cometTail):
        comet = np.zeros_like(image)

        for iTail in range(0,cometTail):
            if self.currentFrame + iTail in self.frames:
                cv2.line(comet, centroids[-1-iTail],centroids[-2-iTail], color, 2)
            else:
                break

        return comet


        self.kalmanFilter =  kf.KalmanFilterClass(id,startFrame,self.centroid)

    def isInLine(self,line):
        distanceToLine = np.abs(self.indexes[0] * line[0] + self.indexes[1] * line[1] + line[2])
        if min(distanceToLine) < 2:
            return True
        else:
            return False

    def isCentroidInLine(self,line):
        if np.abs(self.centroid[0] * line[0] + self.centroid[1] * line[1] + line[2]) < 2:
            return True
        else:
            return False
