#-*- coding: utf-8 -*-
try:
    from PIL import Image
except ImportError:
    import Image
import numpy as np
import cv2
import PIL
import pytesseract
import os
import sys



def find_points(pts):
    #x,y 좌표 4개
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect

def scan_image():
    
    image = cv2.imread('D:\\eclipse-workspace\\java_Server/document.jpg')
    original = image.copy() #비교
    r = 600.0 / image.shape[0]
    dim = (int(image.shape[1] * r), 600)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)

    #gray-scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    #Canny-edge-detection (minval,maxval)
    edged = cv2.Canny(gray, 50, 175)

    #cv2.CHAIN_APPROX_SIMPLE는 4개의 point 저장
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            imgcnt = approx
            break

    cv2.drawContours(image, [imgcnt], -1, (0, 255, 0), 2)
    
    rect = find_points(imgcnt.reshape(4, 2) / r)
    (topLeft, topRight, bottomRight, bottomLeft) = rect
    
    #너비높이계산
    w1 = abs(bottomRight[0] - bottomLeft[0])
    w2 = abs(topRight[0] - topLeft[0])
    h1 = abs(topRight[1] - bottomRight[1])
    h2 = abs(topLeft[1] - bottomLeft[1])
    maxWidth = max([w1, w2])
    maxHeight = max([h1, h2])
    
    dst = np.float32([[0,0], [maxWidth-1,0], 
                      [maxWidth-1,maxHeight-1], [0,maxHeight-1]])
    
    #투영변환
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(original, M, (maxWidth, maxHeight))

    #그레이스케일
    warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    #영상이진화
    warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 10)

    print ("Scan completed!!")
    cv2.imshow("Original", original)
    cv2.imshow("Scanned", warped)
    cv2.imwrite('scanned.png', warped)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)


    print(pytesseract.image_to_string(Image.open('scanned.png'), lang='kor'))
    
    
    os.system("pause")

if __name__ == '__main__':
    scan_image()