
import math
import numpy as np
import cv2
import configuration as conf
import block_matching as match
from skimage.measure import block_reduce
import opticalFlowMethods as opticalflow


def msen(resultOF, gtOF):
    errorVector = []
    correctPrediction = []

    uResult = []
    vResult = []
    uGT = []
    vGT = []
    imageToReconstruct = []

    validGroundTruth = []

    # flow_u(u, v) = ((float)I(u, v, 1) - 2 ^ 15) / 64.0;
    # flow_v(u, v) = ((float) I(u, v, 2) - 2 ^ 15) / 64.0;
    # valid(u, v) = (bool)I(u, v, 3);
    for pixel in range(0,resultOF[:,:,0].size):
        uResult.append( ((float)(resultOF[:,:,1].flat[pixel]) - math.pow(2, 15) ) / 64.0 )
        vResult.append(((float)(resultOF[:,:,2].flat[pixel])-math.pow(2, 15))/64.0)
        uGT.append(((float)(gtOF[:,:,1].flat[pixel])-math.pow(2, 15))/64.0)
        vGT.append(((float)(gtOF[:,:,2].flat[pixel])-math.pow(2, 15))/64.0)
        validGroundTruth.append( gtOF[:,:,0].flat[pixel] )

    for idx in range(len(uResult)):
        if validGroundTruth[idx] == 0:
            imageToReconstruct.append(0)
            continue
        else:
            squareError = math.sqrt(math.pow((uGT[idx] - uResult[idx]), 2) + math.pow((vGT[idx] - vResult[idx]), 2))

        errorVector.append(squareError)
        imageToReconstruct.append(squareError)

        if (squareError > 3):
            correctPrediction.append(0)
        else:
            correctPrediction.append(1)

    error = (1 - sum(correctPrediction)/(float)(sum(validGroundTruth))) * 100;

    errorArray = np.asarray(errorVector)

    return errorArray, error, imageToReconstruct


def transformGTOF(OFimage):
    r, c, d = OFimage.shape;
    OFimage = block_reduce(OFimage, block_size=(2,2,1), func=np.mean)
    r, c, d = OFimage.shape;
    uResult = []
    vResult = []
    validGroundTruth = []

    for pixel in range(0, OFimage[:, :, 0].size):
        isOF = OFimage[:, :, 0].flat[pixel]
        if isOF == 1:
            uResult.append((((float)(OFimage[:, :, 1].flat[pixel]) - math.pow(2, 15)) / 64.0)/200.0)
            vResult.append((((float)(OFimage[:, :, 2].flat[pixel]) - math.pow(2, 15)) / 64.0)/200.0)
        else:
            uResult.append(0)
            vResult.append(0)
        validGroundTruth.append(isOF)

    uResult = np.reshape(uResult, (r, c))
    vResult = np.reshape(vResult, (r, c))
    x, y = np.meshgrid(np.arange(0, c, 1), np.arange(0, r, 1))

    print 'Finished x, y'

    return np.dstack((x,y))



if __name__ == "__main__":
    print 'MSEN and PEPN values '

    '''
    Insert this paths into your configuration.py file

    folders["imgPrevOF"] = "../../../datasetDeliver_2/colored_0/000045_10.png"
    folders["imgCurrOF"] = "../../../datasetDeliver_2/colored_0/000045_11.png"
    folders["imgGTOF"] = "../../../datasetDeliver_2/colored_0/000045_10.png"

    '''

    prev_img = cv2.imread(conf.folders["imgPrevOF"], -1)
    curr_img = cv2.imread(conf.folders["imgCurrOF"], -1)
    gt_img = cv2.imread(conf.folders["imgGTOF"], -1)

    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

    # Transform the GT
    gt_matrix = transformGTOF(gt_img)
    # Reduce the shape from (188, 621, 2) to 188x620x2
    gt_matrix = gt_matrix[:,1:,:]

    # Apply block matching. The output shape is (188, 620, 2)
    # motion_matrix = match.compute_block_matching(prev_img, curr_img)
    motion_matrix = opticalflow.opticalFlowBW(prev_img, curr_img)

    print(motion_matrix.shape)
    print(gt_matrix.shape)

    # Compute metrics
    msenValues, error, image = msen(motion_matrix, gt_matrix)

    print (msenValues)
    print (error)
