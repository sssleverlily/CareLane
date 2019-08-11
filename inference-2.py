#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time


# In[ ]:


import os
import sys
import cv2 as cv
import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import uff
import tensorrt as trt
import graphsurgeon as gs
import json

from GenerateCarLicensePlate import generate_car_license_plate
from GenerateCarLicensePlate import chinese
from GenerateCarLicensePlate import number
from GenerateCarLicensePlate import ALPHABET


# In[ ]:


plt.rcParams['figure.dpi'] = 120 #分辨率


# In[ ]:


cap = cv.VideoCapture(0)


# In[ ]:


maxWidth = 640
maxHeight = 480
cap.set(3,maxWidth)
cap.set(4,maxHeight)


# In[ ]:


text, image = generate_car_license_plate()
print("车牌图像channel:", image.shape)
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 400
MAX_CAPTCHA = len(text)
print("车牌文本最长字符数", MAX_CAPTCHA)

# 把彩色图像转为灰度图像（色彩对识别车牌没有什么用）
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        return gray
    else:
        return img

char_set = number + ALPHABET + chinese 
print(char_set)
CHAR_SET_LEN = len(char_set)
number_len = len(number)

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 400
MAX_CAPTCHA = 7
print("车牌字符数", MAX_CAPTCHA)


# In[ ]:


# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i, c in enumerate(char_pos):
        char_at_pos = i 
        char_idx = c % CHAR_SET_LEN
        if char_idx < number_len:
            char_code = chr(char_idx + ord('0'))
        elif char_idx == 10:
            char_code = 'A'
        elif char_idx == 11:
            char_code = 'B'
        elif char_idx == 12:
            char_code = 'C'
        elif char_idx == 13:
            char_code = 'K'
        elif char_idx == 14:
            char_code = 'P'
        elif char_idx == 15:
            char_code = 'S'
        elif char_idx == 16:
            char_code = 'T'
        elif char_idx == 17:
            char_code = 'X'
        elif char_idx == 18:
            char_code = 'Y'
        elif char_idx == 19:
            char_code = '浙'
        elif char_idx == 20:
            char_code = '苏'
        elif char_idx == 21:
            char_code = '沪'
        elif char_idx == 22:
            char_code = '京'
        elif char_idx == 23:
            char_code = '辽'
        elif char_idx == 24:
            char_code = '鲁'
        elif char_idx == 25:
            char_code = '闽'
        elif char_idx == 26:
            char_code = '陕'
        elif char_idx == 27:
            char_code = '渝'
        elif char_idx == 28:
            char_code = '川'
        text.append(char_code)
    return "".join(text)


# In[ ]:


TRT_LOGGER = trt.Logger(trt.Logger.INFO)
trt.init_libnvinfer_plugins(TRT_LOGGER, '')
runtime = trt.Runtime(TRT_LOGGER)


# In[ ]:

print('asdf')
with open('model.bin', 'rb') as f:
    print('dd')
    buf = f.read()
    print('ss')
    engine = runtime.deserialize_cuda_engine(buf)


# In[ ]:


host_inputs  = []
cuda_inputs  = []
host_outputs = []
cuda_outputs = []
bindings = []
stream = cuda.Stream()

for binding in engine:
    size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
    host_mem = cuda.pagelocked_empty(size, np.float32)
    cuda_mem = cuda.mem_alloc(host_mem.nbytes)

    bindings.append(int(cuda_mem))
    if engine.binding_is_input(binding):
        host_inputs.append(host_mem)
        cuda_inputs.append(cuda_mem)
    else:
        host_outputs.append(host_mem)
        cuda_outputs.append(cuda_mem)
context = engine.create_execution_context()


# In[ ]:


while 1:
    ret,frame=cap.read()
    
    if(ret):
        rgbImg = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        # show
#         plt.imshow(rgbImg)
#         plt.show()
        
        rgbImg = rgbImg.astype(np.float32) * 1.2 + 30
        rgbImg[rgbImg > 255] = 255
        rgbImg = rgbImg.astype(np.uint8)
        
        hsvImg = cv.cvtColor(rgbImg, cv.COLOR_RGB2HSV)
        
        lowerBlue = np.array([78, 80, 80])
        upperBlue = np.array([124, 255, 255])
        
        maskBlue = cv.inRange(hsvImg, lowerBlue, upperBlue)
        kernel = np.ones((5, 5), np.uint8)
        
        maskBlue = cv.morphologyEx(maskBlue, cv.MORPH_OPEN, kernel)
        _, contours, hierarchy = cv.findContours(maskBlue, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        
        maxContour = 0
        maxArea = 0

        if len(contours) != 0:
            for contour in contours:
                area = cv.contourArea(contour)
                if(area > maxArea):
                    maxArea = area
                    maxContour = contour
                    
                    rect = cv.minAreaRect(maxContour)
                    box = cv.boxPoints(rect)
                    pts2 = np.float32([[0, 100], [0, 0], [400, 0], [400, 100]])

                    if(abs(rect[2]) > 45):
                        pts2 = np.float32([[400, 100], [0, 100], [0, 0], [400, 0]])
        
            rect = cv.minAreaRect(maxContour)
            box = cv.boxPoints(rect)
            pts2 = np.float32([[0, 100], [0, 0], [400, 0], [400, 100]])

            if(abs(rect[2]) > 45):
                pts2 = np.float32([[400, 100], [0, 100], [0, 0], [400, 0]])
                
                
            M = cv.getPerspectiveTransform(box, pts2)
            carlicenseImg = cv.warpPerspective(rgbImg, M, (400, 100))
            
            # show
#             plt.imshow(carlicenseImg)
#             plt.show()
            
            newCarlicenseImg = carlicenseImg[3:97,20:380]
            newCarlicenseImg = cv.resize(newCarlicenseImg, (400, 100))
            
            hsvImg = cv.cvtColor(newCarlicenseImg, cv.COLOR_RGB2HSV)

            lowerWhite = np.array([0, 0, 255-70])
            upperWhite = np.array([255, 70, 255])

            maskWhite = cv.inRange(hsvImg, lowerWhite, upperWhite)
            maskWhite = cv.morphologyEx(maskWhite, cv.MORPH_OPEN, kernel)
            maskWhite = cv.cvtColor(maskWhite, cv.COLOR_GRAY2RGB)
            
            backgroundImg = cv.imread('background.jpg')
            processedImg = cv.cvtColor(backgroundImg, cv.COLOR_BGR2RGB)
            processedImg = cv.add(processedImg, maskWhite)
            
            # show
#             plt.imshow(processedImg)
#             plt.show()

            image = convert2gray(processedImg)
            image = image / 255.0
            image = image.reshape((100, 400, 1))
            image = image.transpose(2, 0, 1)
            
            np.copyto(host_inputs[0], image.ravel())
            
            cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
            context.execute_async(bindings=bindings, stream_handle=stream.handle)
            cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
            stream.synchronize()
            
            output = host_outputs[0]

            predict = np.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
            text_list = np.argmax(predict, 2)

            text = text_list[0].tolist()
            vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
            i = 0
            for n in text:
                vector[i * CHAR_SET_LEN + n] = 1
                i += 1

            predict_text = vec2text(vector)

            print('预测： {}'.format(predict_text))
