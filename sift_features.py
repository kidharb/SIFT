#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:54:26 2019

@author: kidharb
"""

import cv2
import numpy as np
from sklearn.preprocessing import normalize


def DoG():
    fn = '/Users/kidharb/Documents/MSc/ComputerVision/project/lena_gray.png'
    fn_no_ext = fn.split('.')[0]
    #read the input file
    img = cv2.imread(str(fn),cv2.IMREAD_GRAYSCALE)

    # normalize image
    img = cv2.normalize(img, None, norm_type=cv2.NORM_INF,dtype=cv2.CV_32F)

    sigma = 0.707
    k = 2**(1/2)
    #run a 5x5 gaussian blur then a 3x3 gaussian blr
    blur3 = cv2.GaussianBlur(img,(3,3),sigma,sigma)
    sigma = 1
    blur5 = cv2.GaussianBlur(blur3,(3,3),sigma,sigma)
    sigma = 1.41
    blur7 = cv2.GaussianBlur(blur5,(3,3),sigma,sigma)
    sigma = 2
    blur9 = cv2.GaussianBlur(blur7,(3,3),sigma,sigma)


    #write the results of the previous step to new files
    cv2.imwrite('lena_gray3x3.jpg', blur3 * 255)
    cv2.imwrite('lena_gray5x5.jpg', blur5 * 255)
    cv2.imwrite('lena_gray7x7.jpg', blur7 * 255)
    cv2.imwrite('lena_gray9x9.jpg', blur9 * 255)

    DoGim53 = blur5 - blur3
    DoGim75 = blur7 - blur5
    DoGim97 = blur9 - blur7

    combined = np.dstack((DoGim53, DoGim75, DoGim97))

    minimum = np.amin(combined, axis=2)
    maximum = np.amax(combined, axis=2)
    outputFile = fn_no_ext+'DoG53.jpg'
    cv2.imwrite(outputFile, cv2.normalize(DoGim53, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U))
    outputFile = fn_no_ext+'DoG75.jpg'
    cv2.imwrite(outputFile, cv2.normalize(DoGim75, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U))
    outputFile = fn_no_ext+'DoG97.jpg'
    cv2.imwrite(outputFile, cv2.normalize(DoGim97, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U))
    return

def compareImages():
    input1 = 'lena_gray3x3.jpg'
    input2 = 'lena_gray5x5.jpg'
    outFile = 'lena_dog.jpg'
 
    in1 = cv2.imread(input1)
    in2 = cv2.imread(input2)
 
    output1 = in2 * -1*in1
 
    cv2.imwrite(outFile, output1)
    return

print("Welcome to the Difference of Gaussian Image creation utility.")

DoG()
#compareImages()
img = cv2.imread('/Users/kidharb/Documents/MSc/ComputerVision/project/lena_gray.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,outImage=np.array([]))

cv2.imwrite('sift_keypoints.jpg',img)