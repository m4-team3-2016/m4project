from multiprocessing import Pool
import numpy as np
import sys
sys.path.append('../')
import configuration as conf


def f(x):
    print x

def obtainIndexesImage(shape,blockSize= conf.OFblockSize,searchArea= conf.OFsearchArea):
    blockIndexes = []
    searchAreaIndexes = []

    for x in range(shape[1]/blockSize):
        for y in range(shape[0]/blockSize):
            xBlockRange = [blockSize*x,blockSize*(x+1)]
            yBlockRange = [blockSize*y,blockSize*(y+1)]
            blockIndexes.append([xBlockRange,yBlockRange])

            xSearchRange = [max(0,blockSize*x-searchArea),min(shape[1] - 1,blockSize*(x+1) + searchArea)]
            ySearchRange = [max(0,blockSize*y-searchArea),min(shape[0] - 1,blockSize*(y+1) + searchArea)]
            searchAreaIndexes.append([xSearchRange,ySearchRange])
    return blockIndexes,searchAreaIndexes

if __name__ == '__main__':
    p = Pool(5)

    im = np.random.randint(0,5,(64,64))
    a,b = obtainIndexesImage(im.shape)

    for el in a:
        print im[el[0][0]:el[0][1],el[1][0]:el[1][1]].shape



    #print(p.map(f,x))
