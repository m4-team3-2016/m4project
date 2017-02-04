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


    def isInLine(self,line)
