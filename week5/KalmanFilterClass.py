import numpy as np
import cv2.cv as cv


import sys
sys.path.append('../')
import configuration as conf


class KalmanFilterClass:

    def __init__(self,id,startFrame,initialPosition):
        self.id = id
        self.startFrame = startFrame
        self.currentFrame = startFrame
        self.frames = [self.currentFrame]
        self.kalman = cv.CreateKalman(4, 2, 0)
        self.previousValues = cv.CreateMat(2, 1, cv.CV_32FC1)
        self.initValues = cv.CreateMat(2, 1, cv.CV_32FC1)
        # This happens only one time to initialize the kalman Filter with the first (x,y) point
        self.kalman.state_pre[0, 0] = initialPosition[0]
        self.kalman.state_pre[1, 0] = initialPosition[1]
        self.kalman.state_pre[2, 0] = 0
        self.kalman.state_pre[3, 0] = 0

        # set kalman transition matrix
        self.kalman.transition_matrix[0, 0] = 1
        self.kalman.transition_matrix[1, 1] = 1
        self.kalman.transition_matrix[2, 2] = 1
        self.kalman.transition_matrix[3, 3] = 1

        # set Kalman Filter
        cv.SetIdentity(self.kalman.measurement_matrix, cv.RealScalar(1))
        cv.SetIdentity(self.kalman.process_noise_cov, cv.RealScalar(1e-5))  ## 1e-5
        cv.SetIdentity(self.kalman.measurement_noise_cov, cv.RealScalar(1e-1))
        cv.SetIdentity(self.kalman.error_cov_post, cv.RealScalar(0.1))


    def computePredictionKalmanFilter(self, currentPosition):
        kalman_prediction = cv.KalmanPredict(self.kalman)
        kalman_prediction = np.array([kalman_prediction[0, 0], kalman_prediction[1, 0]])
        return kalman_prediction

    def correctKalmanFilter(self, currentPosition):
        rightPoints = cv.CreateMat(2, 1, cv.CV_32FC1)
        rightPoints[0, 0] = currentPosition[0]
        rightPoints[1, 0] = currentPosition[1]

        self.kalman.state_pre[0, 0] = currentPosition[0]
        self.kalman.state_pre[1, 0] = currentPosition[1]
        self.kalman.state_pre[2, 0] = 0
        self.kalman.state_pre[3, 0] = 0

        estimated = cv.KalmanCorrect(self.kalman, rightPoints)