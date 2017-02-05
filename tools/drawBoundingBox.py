import cv2
import numpy as np

def drawBBoxWithText(image,text,topLeft,bottomRight,color,alpha):
    textDisplacementX = 10
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.4
    textSize = cv2.getTextSize(text, font, fontScale,1)

    frame = np.zeros_like(image)
    cv2.rectangle(frame,topLeft,bottomRight,color)
    cv2.rectangle(frame,(topLeft[0],bottomRight[1]),(topLeft[0] + textDisplacementX + textSize[0][0] ,bottomRight[1] + 15),color,-1)

    # cv2.addWeighted(frame, alpha, image, 1 - alpha, 0, image)

    cv2.putText(frame,text,(topLeft[0]+5,bottomRight[1]+textDisplacementX),font,fontScale,(1,1,1))
    return frame


def drawCentroidWithText(image,text,centroid,color,alpha):
    textDisplacementX = 10
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.4
    textSize = cv2.getTextSize(text, font, fontScale,1)
    frame = np.ones_like(image)
    cv2.circle(frame,centroid,3,color,-1)

    cv2.rectangle(frame,(centroid[0]+15,centroid[1]),(centroid[0] + 15 + textDisplacementX + textSize[0][0] ,centroid[1] + 15),color,-1)
    cv2.putText(frame,text,(centroid[0]+20,centroid[1]+textDisplacementX),font,fontScale,(255,255,255))

    return frame