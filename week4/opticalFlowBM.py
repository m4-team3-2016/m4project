from multiprocessing import Pool
import numpy as np
import sys
sys.path.append('../')
import configuration as conf


def findCorrespondentBlock(x):
    originBlock = x[0]
    searchArea = x[1]
    searchIndexes = x[2]
    imageShape = x[3]
    area = 0
    dx = 0
    dy = 0
    for x in range(2*conf.OFsearchArea-max(0,searchIndexes[0][1]-imageShape[1])):
        for y in range(2*conf.OFsearchArea-max(0,searchIndexes[1][1]-imageShape[0])):
            temp = compareRegions(originBlock,searchArea[y:y+originBlock.shape[0],x:x+originBlock.shape[1]])
            if temp > area:
                area = temp
                dx = x
                dy = y

    #Coordinates correction
    dx += conf.OFsearchArea + min(0,searchIndexes[0][0])
    dy += conf.OFsearchArea + min(0,searchIndexes[1][0])
    return [dx,dy]

def compareRegions(block1,block2):
    return np.random.randint(0,10,(5,5))[3,3]

def obtainIndexesImage(shape,blockSize= conf.OFblockSize,searchArea= conf.OFsearchArea):
    blockIndexes = []
    searchAreaIndexes = []

    for y in range(shape[0]/blockSize):
        for x in range(shape[1]/blockSize):
            xBlockRange = [blockSize*x,blockSize*(x+1)]
            yBlockRange = [blockSize*y,blockSize*(y+1)]
            blockIndexes.append([xBlockRange,yBlockRange])

            xSearchRange = [blockSize*x-searchArea,blockSize*(x+1) + searchArea]
            ySearchRange = [blockSize*y-searchArea,blockSize*(y+1) + searchArea]
            searchAreaIndexes.append([xSearchRange,ySearchRange])
    return blockIndexes,searchAreaIndexes

if __name__ == '__main__':
    p = Pool(5)

    im = np.random.randint(0,5,(16,16))
    blockIndexes,searchAreaIndexes = obtainIndexesImage(im.shape)

    x = [[im[b[0][0]:b[0][1],b[1][0]:b[1][1]],im[max(0,s[0][0]):min(im.shape[0],s[0][1]),max(0,s[1][0]):min(im.shape[1],s[1][1])],c,im.shape] for b,s,c in zip(blockIndexes,searchAreaIndexes,searchAreaIndexes)]
    print im
    dumb = []
    for i in x:
        dumb.append(findCorrespondentBlock(i))

    #p.map(findCorrespondentBlock,x)
    print dumb
