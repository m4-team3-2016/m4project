import numpy as np
import sys
sys.path.append('../')
import configuration as conf

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


    def __init__(self,id,startFrame,topLeft,bottomRight,indexes):
        self.id = id
        self.startFrame = startFrame
        self.currentFrame = startFrame
        self.frames = [self.currentFrame]
        self.checkWidthHeight(topLeft,bottomRight)
        # The white pixels position
        self.indexes = [indexes]
        self.topLeft = topLeft
        self.bottomRight = bottomRight

        self.centroid = [sum(indexes[0])/len(indexes[0]),sum(indexes[1])/len(indexes[1])]

    def isInLine(self,line):
        distanceToLine = np.abs(self.indexes[0] * line[0] + self.indexes[1] * line[1] + line[2])
        if min(distance) < 2:
            return True
        else:
            return False

    def isCentroidInLine(self,line):
        if np.abs(self.centroid[0] * line[0] + self.centroid[1] * line[1] + line[2]) < 2:
            return True
        else:
            return False
