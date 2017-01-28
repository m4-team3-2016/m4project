
import math
import numpy as np


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


if __name__ == "__main__":
    print 'MSEN and PEPN values '
