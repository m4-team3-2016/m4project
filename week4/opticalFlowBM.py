from multiprocessing import Pool
import multiprocessing
import numpy as np
import sys
sys.path.append('../')
import configuration as conf


def findCorrespondentBlock(x):

    originBlock = x[0]
    searchArea = x[1]
    searchIndexes = x[2]
    yRange = searchIndexes[0]
    xRange = searchIndexes[1]

    imHeight = x[3][0]
    imWidth = x[3][1]


    assert originBlock.shape[0] == originBlock.shape[1]
    N = conf.OFblockSize
    P = conf.OFsearchArea

    adjY = -(P + min(0,yRange[0]))
    adjX = -(P + min(0,xRange[0]))

    topY = 1 + 2 * P + min(0,yRange[0]) - max(0,yRange[1] - imHeight)
    topX = 1 + 2 * P + min(0,xRange[0]) - max(0,xRange[1] - imWidth)
    score = 10000.0

    for x in range(0,topX):
        for y in range(0,topY):
            #print str(x) + "/" + str(topX-1) + ", " + str(y) + "/" + str(topY-1)
            temp = compareRegions(originBlock,searchArea[y:y+originBlock.shape[0],x:x+originBlock.shape[1]])
            if temp < score:
                score = temp
                dx = x
                dy = y

    #Coordinates correction
    return [dx + adjX ,dy + adjY]


def compareRegions(block1,block2):
    return np.sqrt(sum(sum((block1-block2)**2)))

def obtainIndexesImage(shape,blockSize= conf.OFblockSize,searchArea= conf.OFsearchArea):
    blockIndexes = []
    searchAreaIndexes = []

    for y in range(shape[0]/blockSize):
        for x in range(shape[1]/blockSize):
            yBlockRange = [blockSize*y,blockSize*(y+1)]
            xBlockRange = [blockSize*x,blockSize*(x+1)]
            blockIndexes.append([yBlockRange,xBlockRange])

            xSearchRange = [blockSize*x-searchArea,blockSize*(x+1) + searchArea]
            ySearchRange = [blockSize*y-searchArea,blockSize*(y+1) + searchArea]
            searchAreaIndexes.append([ySearchRange,xSearchRange])
    return blockIndexes,searchAreaIndexes

def opticalFlowBW(frame1,frame2):
    width = frame1.shape[1]
    height = frame1.shape[0]

    newWidth = width/conf.OFblockSize
    newHeight = height/conf.OFblockSize

    p = Pool(multiprocessing.cpu_count())
    blockIndexes,searchAreaIndexes = obtainIndexesImage(frame2.shape)
    xSearchIndexes = [el[0] for el in searchAreaIndexes]
    ySearchIndexes = [el[1] for el in searchAreaIndexes]

    x = [[frame1[b[0][0]:b[0][1],b[1][0]:b[1][1]], \
          frame2[max(0,sX[0]):min(frame2.shape[0],sX[1]),max(0,sY[0]):min(frame2.shape[1],sY[1])], \
          c, \
          frame2.shape] \
          for b,sY,sX,c in zip(blockIndexes,ySearchIndexes,xSearchIndexes,searchAreaIndexes)]
    OF = p.map(findCorrespondentBlock,x)

    OFx = np.reshape(np.asarray([el[0] for el in OF]),(newHeight,newWidth))
    OFy = np.reshape(np.asarray([el[1] for el in OF]),(newHeight,newWidth))

    return np.dstack((OFx,OFy))

if __name__ == '__main__':
    im = np.random.randint(0,5,(4,32))
    out = opticalFlowBW(im,np.roll(im,1,axis=1))
    print out
