import cv2

# Mode regarding video or folders:
mode = 'video' # video | folder
# A .mp4 file if mode is video, a folder with all the frames if mode is folder
inputData = 'video.mp4'

#ColorSpace to work
colorSpaceConversion = {}
colorSpaceConversion['YCrCb'] = cv2.COLOR_BGR2YCR_CB
colorSpaceConversion['HSV']   = cv2.COLOR_BGR2HSV
colorSpaceConversion['HLS']   = cv2.COLOR_BGR2HLS
colorSpaceConversion['gray']  = cv2.COLOR_BGR2GRAY
colorSpaceConversion['LUV']   = cv2.COLOR_BGR2LUV
colorSpaceConversion['LAB']   = cv2.COLOR_BGR2LAB

colorSpace = 'HLS'


# Background Substraction
trainingPercentage = 0.01
