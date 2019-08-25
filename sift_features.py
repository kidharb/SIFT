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
def get_keypoints(m1,m2,m3):
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
    keypoint_img = img

    # normalize image
    img = cv2.normalize(img, None, norm_type=cv2.NORM_INF,dtype=cv2.CV_32F)

    #produce s + 3 images at each octave
    s = 2
    sigma = 0.707
    k = 2**(1/s)

    scale = []
    for i in range(s+3):
        scale.append(cv2.GaussianBlur(img,(3,3),sigma,sigma))
        img = scale[i]
        sigma = k * sigma
        cv2.imwrite("lena_gray_gaussian_{}.jpg".format(i), scale[i] * 255)

    DoG = []
    for i in range(s+3-1):
        DoG.append(scale[i+1] - scale[i])
        cv2.imwrite("lena_gray_DoG_{}.jpg".format(i), cv2.normalize(DoG[i], None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U))

    keypoint_count = 0
    key_points = []
    for i in range(s):
        for row in range(0, image_width-2):
            for col in range(0, image_width-2):
                indices = np.array(
                    [(row * image_width) + col, (row * image_width) + col + 1, (row * image_width) + col + 2,
                    (row * image_width) + col + 9, (row * image_width) + col + 10, (row * image_width) + col + 11,
                    (row * image_width) + col + 18, (row * image_width) + col + 19, (row * image_width) + col + 20])
                mat1 = np.reshape(np.take(DoG[i], indices), (3, 3))
                mat2 = np.reshape(np.take(DoG[i+1], indices), (3, 3))
                mat3 = np.reshape(np.take(DoG[i+2], indices), (3, 3))
                keypoint_value = get_keypoints(mat1, mat2, mat3)

                if (keypoint_value >= 1):
                    keypoint_count += 1
                    key_points.append([row,col,s])
                    cv2.circle(keypoint_img, (row, col), 3, (255, 255, 255), thickness=1)

    outputFile = fn_no_ext+'_my_keypoints.jpg'
    cv2.imwrite(outputFile, keypoint_img)
    print("Detected " + str(keypoint_count) + " keypoints")
    return
"""
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
"""

print("Welcome to the Difference of Gaussian Image creation utility.")
image_width = 512
DoG()
img = cv2.imread('/Users/kidharb/Documents/MSc/ComputerVision/project/lena_gray.png')
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)

img=cv2.drawKeypoints(gray,kp,outImage=np.array([]))

cv2.imwrite('sift_keypoints.jpg',img)