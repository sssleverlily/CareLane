#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv
import numpy as np
import time
import json


# In[ ]:


from periphery import Serial


# In[ ]:


def displayImg(name, ori, imgType = 'rgb'):
    displayImg = ori
    if(imgType == 'rgb'):
        displayImg = cv.cvtColor(ori, cv.COLOR_RGB2BGR)
    #cv.imshow(name, displayImg)
    #cv.waitKey(1)


# In[ ]:


def detectLane(rgbImg, roiPoints = None):
    # Global variable
    maxHeight = rgbImg.shape[0]
    maxWidth = rgbImg.shape[1]
    kernel = np.ones((5, 5), np.uint8)
    
    displayImg('rgbImg', rgbImg)

    # 11 55
    lowerYellow = np.array([11, threshold1, threshold2])
    upperYellow = np.array([35, 255, 255])

    lowerWhite = np.array([0, 0, 255-threshold3])
    upperWhite = np.array([255, threshold3, 255])
    
    # Init
    hsvImg = cv.cvtColor(rgbImg, cv.COLOR_RGB2HSV)
    
    # Color spilt
    maskYellow = cv.inRange(hsvImg, lowerYellow, upperYellow)
    maskYellow = cv.morphologyEx(maskYellow, cv.MORPH_OPEN, kernel)
    
    displayImg('maskYellow', maskYellow, 'gray')

    maskWhite = cv.inRange(hsvImg, lowerWhite, upperWhite)
    maskWhite = cv.morphologyEx(maskWhite, cv.MORPH_OPEN, kernel)
    
    displayImg('maskWhite', maskWhite, 'gray')
    
    # ROI area
    if(type(roiPoints) == type(None)):
        roiPoints = defaultRoiPoints
    
    roiMask = np.zeros((maxHeight, maxWidth), np.uint8)
    cv.fillPoly(roiMask, [roiPoints], (255, 0, 0))
    
    displayImg('roiMask', roiMask, 'gray')
    
    needQuickly = False
    
    # Yellow line
    maskYellow = cv.bitwise_and(maskYellow, roiMask)
    yellowLineEdges = cv.Canny(maskYellow, 20, 60)
    
    # displayImg('yellowLineEdges', yellowLineEdges, 'gray')
    
    yellowLines = cv.HoughLinesP(yellowLineEdges, 4, np.pi/180, 30,minLineLength=40, maxLineGap=180)
    
    yellowBottomX = -1
    if type(yellowLines) != type(None):
        yellowKAvg = 0

        cnt = 0
        for line in yellowLines:
            x1, y1, x2, y2 = line[0]
            if(x2 != x1):
                yellowKAvg += 1.0 * (y2-y1) / (x2-x1)
                cnt += 1
        if(cnt != 0):
            yellowKAvg /= cnt
            
            if(abs(yellowKAvg) < 0.55):
                needQuickly = True

            _, yellowLineContours, _ = cv.findContours(yellowLineEdges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            yellowMID = 0
            yellowMA = 0
            for i in range(len(yellowLineContours)):
                tmp = cv.contourArea(yellowLineContours[i])
                if(tmp > yellowMA):
                    yellowMA = tmp
                    yellowMID = i

            yellowM = cv.moments(yellowLineContours[i])
            yellowBottomX = 1
            if(yellowM['m00']!=0):
                yellowCX = int(yellowM['m10']/yellowM['m00'])
                yellowCY = int(yellowM['m01']/yellowM['m00'])
                yellowB = yellowCY - yellowKAvg * yellowCX
                
                if((maxHeight - yellowB) / yellowKAvg != float("inf")):
                    yellowBottomX = (int)((maxHeight - yellowB) / yellowKAvg)
    
    # White line
    maskWhite = cv.bitwise_and(maskWhite, roiMask)
    whiteLineEdges = cv.Canny(maskWhite, 20, 60)
    
    # displayImg('whiteLineEdges', whiteLineEdges, 'gray')
    
    whiteLines = cv.HoughLinesP(whiteLineEdges, 4, np.pi/180, 30,minLineLength=40, maxLineGap=180)
    
    whiteBottomX = -1
    if type(whiteLines) != type(None):
        
        whiteKAvg = 0

        cnt = 0
        for line in whiteLines:
            x1, y1, x2, y2 = line[0]
            if(x2 != x1):
                whiteKAvg += 1.0 * (y2-y1) / (x2-x1)
                cnt += 1
        if(cnt != 0):
            whiteKAvg /= cnt
            
            if(abs(whiteKAvg) < 0.55):
                needQuickly = True

            _, whiteLineContours, _ = cv.findContours(whiteLineEdges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            whiteMID = 0
            whiteMA = 0
            for i in range(len(whiteLineContours)):
                tmp = cv.contourArea(whiteLineContours[i])
                if(tmp > whiteMA):
                    whiteMA = tmp
                    whiteMID = i

            whiteM = cv.moments(whiteLineContours[i])
            whiteBottomX = 1
            if(whiteM['m00']!=0):
                whiteCX = int(whiteM['m10']/whiteM['m00'])
                whiteCY = int(whiteM['m01']/whiteM['m00'])
                whiteB = whiteCY - whiteKAvg * whiteCX
                
                if((maxHeight - whiteB) / whiteKAvg != float("inf")):
                    whiteBottomX = (int)((maxHeight - whiteB) / whiteKAvg)

    debugImg = cv.bitwise_or(maskYellow, maskWhite)
    displayImg('debugImg', debugImg, 'gray')
        
    # turn left
    if(yellowBottomX == -1 and whiteBottomX != -1):
        if(needQuickly):
            return 'leftQuickly'
        else:
            return 'left'
    
    # turn right
    if(yellowBottomX != -1 and whiteBottomX == -1):
        if(needQuickly):
            return 'rightQuickly'
        else:
            return 'right'
    
    # go straight
    return 'straight'


# In[ ]:


delayTIme = 0.1

# Image
maxHeight = 480
maxWidth = 640

# light
blight = 1.2
bweight = 25

# color
threshold1 = 80
threshold2 = 80
threshold3 = 70

# default ROI
defaultRoiPoints = np.array([[maxWidth / 2 - maxWidth * 3 / 8, maxHeight],
                              [maxWidth / 2  + maxWidth * 3 / 8, maxHeight],
                              [maxWidth / 2 + maxWidth / 8, maxHeight / 2 - maxHeight / 7],
                              [maxWidth / 2 - maxWidth / 8, maxHeight / 2 - maxHeight / 7]], np.int32)


# In[ ]:


# Serial
currentTime = time.time()
serial = Serial('/dev/ttyTHS1', 115200)
commandJson = {'o':1, 'v': 0, 'c':0, 'd':0, 'r':0, 'a':0}


# In[ ]:


cap = cv.VideoCapture(0)
cap.set(3,maxWidth)
cap.set(4,maxHeight)


# In[ ]:


def controlTricar(command):
    global currentTime
    global commandJson

    if(command == 'left'):
        commandJson['o'] = 0
        commandJson['v'] = 29
        commandJson['c'] = 0
        commandJson['d'] = 0
        commandJson['r'] = 10000
        commandJson['a'] = 0
    elif(command == 'leftQuickly'):
        commandJson['o'] = 0
        commandJson['v'] = 10
        commandJson['c'] = 0
        commandJson['d'] = 0
        commandJson['r'] = 7000
        commandJson['a'] = 0
    elif(command == 'right'):
        commandJson['o'] = 0
        commandJson['v'] = 29
        commandJson['c'] = 0
        commandJson['d'] = 1
        commandJson['r'] = 10000
        commandJson['a'] = 0
    elif(command == 'rightQuickly'):
        commandJson['o'] = 0
        commandJson['v'] = 10
        commandJson['c'] = 0
        commandJson['d'] = 1
        commandJson['r'] = 7000
        commandJson['a'] = 0
    elif(command == 'straight'):
        commandJson['o'] = 0
        commandJson['v'] = 35
        commandJson['c'] = 0
        commandJson['d'] = 0
        commandJson['r'] = 0
        commandJson['a'] = 0
    elif(command == 'stop'):
        commandJson['o'] = 0
        commandJson['v'] = 0
        commandJson['c'] = 0
        commandJson['d'] = 0
        commandJson['r'] = 0
        commandJson['a'] = 0
    elif(command == 'turnaround'):
        commandJson['o'] = 0
        commandJson['v'] = 0
        commandJson['c'] = 0
        commandJson['d'] = 1
        commandJson['r'] = 1000
        commandJson['a'] = 180


# In[ ]:


def sendCommand():
    serial.write( bytes(json.dumps(commandJson), encoding = "utf8"))


# In[ ]:


def gogogo():
    global currentTime
    
    while 1:
        # Auto send commdand with none stop
        if(time.time() - currentTime > delayTIme):
            sendCommand()
            currentTime = time.time()
            
        ret,laneFrame=cap.read()

        if(ret):
            laneFrame = (laneFrame.astype(np.float32) * blight + bweight)
            laneFrame[laneFrame > 255] = 255
            laneFrame = laneFrame.astype(np.uint8)

            laneImage = cv.cvtColor(laneFrame, cv.COLOR_BGR2RGB)
            lineCommand = detectLane(laneImage)
            controlTricar(lineCommand)
        else:
            print("False")


# In[ ]:


print('准备开车')
gogogo()


# In[ ]:


controlCar('stop')
sendCommand()

