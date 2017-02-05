import cv2.cv as cv

kalman = cv.CreateKalman(4, 2, 0)

i = 0
# I read the point from an .txt file
with open('example.txt') as f:
    array = []
    for line in f: # read rest of lines
        array.append([float(x) for x in line.split()])
        vec=array.pop()
        x=vec[0]
        y=vec[1]
        # I obtain the (x,y) points

        if i== 0:
        # This happens only one time to initialize the kalman Filter with the first (x,y) point
            kalman.state_pre[0,0]  = 103
            kalman.state_pre[1,0]  = 225
            kalman.state_pre[2,0]  = 0
            kalman.state_pre[3,0]  = 0

        # set kalman transition matrix
            kalman.transition_matrix[0,0] = 1
            kalman.transition_matrix[1,1] = 1
            kalman.transition_matrix[2,2] = 1
            kalman.transition_matrix[3,3] = 1

            # set Kalman Filter
            cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
            cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(1e-5))## 1e-5
            cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(1e-1))
            cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(0.1))
        else:
        # Kalman prediction with Kalman Correction with the points I have in trajectory_0000.txt
            kalman_prediction = cv.KalmanPredict(kalman)
            rightPoints = cv.CreateMat(2, 1, cv.CV_32FC1)
            rightPoints[0,0]=x
            rightPoints[1,0]=y

            kalman.state_pre[0,0]  = x
            kalman.state_pre[1,0]  = y
            kalman.state_pre[2,0]  = 0
            kalman.state_pre[3,0]  = 0

            estimated = cv.KalmanCorrect(kalman, rightPoints)

            print "Sample:" + str(i)
            print "Real values:      " + str( x ) +  " - " + str( y )
            print "Kalman prediction " + str(i) + ": " + str(kalman_prediction[0,0]) + " - " + str(kalman_prediction[1,0])
            print ""

        i = i + 1