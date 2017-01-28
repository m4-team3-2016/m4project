import cv2
import configuration as conf
import numpy as np
import glob
import sys
import os
import block_matching as match
import opticalFlowMethods as opticalflow
sys.path.append('../')
operativeSystem = os.name
(CVmajor, CVminor, _) = cv2.__version__.split(".")

resultsPath = "./resultsStabilization/"
if not os.path.exists(resultsPath):
    os.makedirs(resultsPath)

if not os.path.exists("./videos/"):
    os.makedirs("./videos/")

################### task 2 ######################
# Task 2.1: Video Stabilization with Block Matching
ID = "Traffic"
folder = conf.folders[ID]
block_size = conf.block_size
area_size = conf.area_size
compensation = conf.compensation

folder = conf.folders[ID]
framesFiles = sorted(glob.glob(folder + '*'))
nFrames = len(framesFiles)

referenceImageName = framesFiles[0]
referenceImage = cv2.imread(referenceImageName)
referenceImageBW = cv2.cvtColor(referenceImage, cv2.COLOR_BGR2GRAY)

# OS dependant writing
if operativeSystem == 'posix':
    # posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
    if conf.isMac:
        cv2.imwrite(resultsPath + referenceImageName.split('/')[-1][0:-4] + '.png', referenceImage)
    else:
        cv2.imwrite(resultsPath + referenceImageName.split('/')[-1] + '.png', referenceImage)
else:
    # say hello to propietary software
    cv2.imwrite(resultsPath + referenceImageName.split('\\')[-1].split('.')[0] + '.png', referenceImage)

if CVmajor == '3':
    # openCV 3
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
else:
    # openCV 2
    fourcc = cv2.cv.CV_FOURCC(*'MJPG')

videoOutputOriginal = cv2.VideoWriter("videos/originalVideo.avi", fourcc, 20.0,(referenceImage.shape[1], referenceImage.shape[0]))
videoOutputPost = cv2.VideoWriter("videos/stabilizedVideo.avi", fourcc, 20.0,(referenceImage.shape[1], referenceImage.shape[0]))
videoOutputPost.write(referenceImage)
videoOutputOriginal.write(referenceImage)

for idx in range(1, nFrames):
    file_name = framesFiles[idx]
    if operativeSystem == 'posix':
        # posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
        if conf.isMac:
            print "Computing image " + str(idx) + " " + file_name.split('/')[-1][0:-4]
        else:
            print "Computing image " + str(idx) + " " + file_name.split('/')[-1]
    else:
        print "Computing image " + str(idx) + " " + file_name.split('\\')[-1].split('.')[0]

    currentImage = cv2.imread(framesFiles[idx])
    currentImageBW = cv2.cvtColor(currentImage, cv2.COLOR_BGR2GRAY)

    # Apply block matching
    motion_matrix = match.compute_block_matching(referenceImageBW, currentImageBW)
    # motion_matrix = opticalflow.opticalFlowBW(referenceImageBW, currentImageBW)

    # bincount cannot be negative
    motion_matrix[:, :, :] = motion_matrix[:, :, :] + area_size
    x_motion = np.intp(motion_matrix[:, :, 0].ravel())
    y_motion = np.intp(motion_matrix[:, :, 1].ravel())
    real_x = np.argmax(np.bincount(x_motion)) - area_size
    real_y = np.argmax(np.bincount(y_motion)) - area_size

    out = match.camera_motion(real_x, real_y, currentImage)
    # OS dependant writing
    if operativeSystem == 'posix':
        # posix systems go here: ubuntu, debian, linux mint, red hat, etc, even osX (iew)
        if conf.isMac:
            cv2.imwrite(resultsPath + file_name.split('/')[-1][0:-4] + '.png',out)
        else:
            cv2.imwrite(resultsPath + file_name.split('/')[-1] + '.png', out)
    else:
        # say hello to propietary software
        cv2.imwrite(resultsPath + file_name.split('\\')[-1].split('.')[0] + '.png', out)

    videoOutputPost.write(out)
    videoOutputOriginal.write(currentImage)

    # Create prediction image
    # comp_img = create_compensated_image(prev_img, motion_matrix, block_size, x_blocks, y_blocks)
