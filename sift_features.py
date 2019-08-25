#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:54:26 2019

@author: kidharb
"""

import cv2
import numpy as np
from scipy import ndimage
import sys
from sklearn.preprocessing import normalize

# Takes in 3 matrices and checks if the point in the middle matrix (m2) is a keypoint
# Do this by first checking if we have a maximum
def keypoints(m1,m2,m3):
    keypoint_candidate = m2.item(4)
    if keypoint_candidate == ndimage.maximum(m2):
        if keypoint_candidate > ndimage.maximum(m1):
            if keypoint_candidate > ndimage.maximum(m3):
                return 1 #max keypoint_candidate
            else:
                return -1
        else:
            return -1
    elif keypoint_candidate == ndimage.minimum(m2):
        if keypoint_candidate < ndimage.minimum(m1):
            if keypoint_candidate < ndimage.minimum(m3):
                return 2 #min keypoint_candidate
            else:
                return -1
        else:
            return -1
    return -1

def DoG():
    fn = '/Users/kidharb/Documents/MSc/ComputerVision/project/lena_gray.png'
    fn_no_ext = fn.split('.')[0]
    #read the input file
    img = cv2.imread(str(fn),cv2.IMREAD_GRAYSCALE)
    image_width = len(img[0])
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
    DoGim75_norm = cv2.normalize(DoGim75, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U)
    DoGim97 = blur9 - blur7

    outputFile = fn_no_ext+'DoG53.jpg'
    cv2.imwrite(outputFile, cv2.normalize(DoGim53, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U))
    outputFile = fn_no_ext+'DoG75.jpg'
    cv2.imwrite(outputFile, cv2.normalize(DoGim75, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U))
    outputFile = fn_no_ext+'DoG97.jpg'
    cv2.imwrite(outputFile, cv2.normalize(DoGim97, None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U))

    keypoint_count = 0
    for row in range(0, image_width-2):
        for col in range(0, image_width-2):
            indices = np.array(
                [(row * image_width) + col, (row * image_width) + col + 1, (row * image_width) + col + 2,
                 (row * image_width) + col + 9, (row * image_width) + col + 10, (row * image_width) + col + 11,
                 (row * image_width) + col + 18, (row * image_width) + col + 19, (row * image_width) + col + 20])
            mat1 = np.reshape(np.take(DoGim53, indices), (3, 3))
            mat2 = np.reshape(np.take(DoGim75, indices), (3, 3))
            mat3 = np.reshape(np.take(DoGim97, indices), (3, 3))
            keypoint_value = keypoints(mat1, mat2, mat3)

            if (keypoint_value >= 1):
                keypoint_count += 1
                cv2.circle(DoGim75_norm, (row, col), 3, (50, 25, 10), thickness=1)

    outputFile = fn_no_ext+'DoG75_keypoints.jpg'
    cv2.imwrite(outputFile, DoGim75_norm)
    print("Detected " + str(keypoint_count) + " keypoints")
    return

print("Welcome to the Difference of Gaussian Image creation utility.")
image_width = 512
DoG()
img = cv2.imread('/Users/kidharb/Documents/MSc/ComputerVision/project/lena_gray.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,outImage=np.array([]))

cv2.imwrite('sift_keypoints.jpg',img)