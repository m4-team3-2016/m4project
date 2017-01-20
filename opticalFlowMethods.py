
import numpy as np
import cv2
import configuration as conf
from matplotlib import pyplot as plt
import imutils

RoI = conf.RoI
temporalRoI = conf.temporalRoI

def readImagesFromVideo(videoFile,frameID1,frameID2):

    cap = cv2.VideoCapture(videoFile)
    ret = True

    cap.set(1,frameID1)
    ret,frame1 = cap.read()

    cap.set(1,frameID2)
    ret,frame2 = cap.read()

    cap.release()
    return frame1,frame2


def LukasKanade(videoFile):
    frame1,frame2 = readImagesFromVideo(videoFile,280,281)
    height = frame1.shape[0]
    width = frame1.shape[1]

    cap = cv2.VideoCapture('../movingTrain.mp4')

    feature_params = dict( maxCorners = 100,qualityLevel = 0.1,minDistance = 5,blockSize = 7 )
    lk_params = dict( winSize  = (15,15), maxLevel = 2, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    mask = np.zeros_like(frame1)
    color = np.random.randint(0,255,(100,3))
    p0 = cv2.goodFeaturesToTrack(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), mask = None, **feature_params)
    cap.set(1,280)
    while(1 and cap.get(1) < 400):

        ret,frame2 = cap.read()
        p1, st, err = cv2.calcOpticalFlowPyrLK(cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY), p0, None, **lk_params)

        good_new = p1[st==1]
        good_old = p0[st==1]
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame1 = cv2.circle(frame1,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame1,mask)

        cv2.imshow('frame',img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        frame1 = frame2.copy()
        p0 = good_new.reshape(-1,1,2)

    cv2.destroyAllWindows()
    cap.release()

def denseOpticalFlow(videoFile):

    cap = cv2.VideoCapture(videoFile)
    ret, frame1 = cap.read()
    frame1 = frame1[:300,250:650,:]
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[:,:,1] = 255

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoOutput = cv2.VideoWriter('OpticalFlowTrain.avi',fourcc, 10.0, (frame1.shape[1],frame1.shape[0]),True)


    idx = 0
    flow = None
    ret = True
    while(ret):
        ret, frame2 = cap.read()
        if ret == False:
            break
        frame2 = frame2[:300,250:650,:]
        nextImg = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        '''if np.mod(idx,2) != 0:
            prvs = nextImg
            idx = idx + 1
            continue'''
        if idx < 200 or np.abs((prvs-nextImg).mean()) < 5:
            prvs = nextImg
            idx = idx + 1
            continue
        else:
            flow = cv2.calcOpticalFlowFarneback(prvs,nextImg, flow, 0.5, 5, 15, 9, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            hsv[...,0] = ang*180/np.pi/2
            hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
            #hsv[...,2] = 255
            bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR_FULL)
            mask = np.zeros_like(frame2)
            for h in range(0,frame2.shape[1],5):
                for w in range(0,frame2.shape[0],5):
                    cv2.arrowedLine(mask,(h,w),(h+int(flow[w,h,0]),w+int(flow[w,h,1])),(int(bgr[w,h,0]),int(bgr[w,h,1]),int(bgr[w,h,2])),1)
            outputImage = cv2.add(frame2,mask)
            cv2.arrowedLine(outputImage,(int(outputImage.shape[1]/2),int(outputImage.shape[0]/2)),(int(outputImage.shape[1]/2 + max(0,mag.mean()) * np.cos(ang.mean())),int(outputImage.shape[0]/2 + max(0,mag.mean()) * np.sin(ang.mean()))),(0,0,255),1)
            cv2.namedWindow('Next image', cv2.WINDOW_NORMAL)
            cv2.namedWindow('OF', cv2.WINDOW_NORMAL)
            #cv2.setWindowProperty('Next image',cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
            cv2.putText(outputImage,"Frame:" + str(idx),(50,50),1,1,(255,255,255))
            cv2.imshow('Next image',outputImage)
            cv2.putText(bgr,"Mean diff:" + str(np.abs((prvs-nextImg).mean())),(50,100),1,1,(255,255,255))
            cv2.putText(bgr,"Mean movement:" + str(mag.mean()),(50,50),1,1,(255,255,255))
            cv2.putText(bgr,"Mean orientation:" + str(hsv[...,0].mean()),(50,150),1,1,(255,255,255))
            cv2.imshow('OF',bgr)
            videoOutput.write(outputImage)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            elif k == ord('s'):
                cv2.imwrite('opticalfb.png',frame2)
                cv2.imwrite('opticalhsv.png',bgr)
            idx = idx + 1
            prvs = nextImg
    cap.release()
    videoOutput.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    #LukasKanade('../movingTrain.mp4')
    denseOpticalFlow('../movingTrain.mp4')
