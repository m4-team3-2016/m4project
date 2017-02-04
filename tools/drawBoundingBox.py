import cv2
import numpy as np

def drawBBoxWithText(image,text,topLeft,bottomRight,color,alpha):
    textDisplacementX = 10
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 0.4
    textSize = cv2.getTextSize(text, font, fontScale,1)

    frame = np.ones_like(image)
    cv2.rectangle(frame,topLeft,bottomRight,color)
    cv2.rectangle(frame,(topLeft[0],bottomRight[1]),(topLeft[0] + textDisplacementX + textSize[0][0] ,bottomRight[1] + 15),color,-1)

    cv2.addWeighted(frame, alpha, image, 1 - alpha, 0, image)

    cv2.putText(image,text,(topLeft[0]+5,bottomRight[1]+textDisplacementX),font,fontScale,(255,255,255))
    return image
if __name__ == "__main__":
    im = np.ones((600,480,3))
    text = "d"
    text = "id:0 - 72kmh"
    a = drawBBoxWithText(im,text,(200,200),(350,350),(0,255,255),0.5)

    cv2.imwrite('./test.png',a)
