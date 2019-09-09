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


def generate_octave(img, s, sigma, oct):
    k = 2 ** (1 / s)
    octave = []
    for i in range(s + 3):
        filter_size = int(2*np.floor(3 * sigma) - 1)
        octave.append(cv2.GaussianBlur(img, (filter_size, filter_size), sigma, sigma))
        blurred_image = octave[i]
        sigma = k * sigma
        denorm_image = np.round(((blurred_image - blurred_image.min()) * (1 / (blurred_image.max() - blurred_image.min()) * 255)))
        cv2.imwrite("lena_gray_gaussian_{}_{}.jpg".format(i, oct), denorm_image)

    return octave


# Takes in root image, s, and sigma and outputs a pyramid of image blur
def generate_guassian_pyramid(img, num_octaves, s, sigma):
    gauss_pyr = []
    # normalize image
    img = ((img - img.min()) * (1 / (img.max() - img.min())).astype('float32'))


    denorm_image = np.round(((img - img.min()) * (1 / (img.max() - img.min()) * 255)))
    cv2.imwrite("lena_gray_denorm.jpg", denorm_image)

    for oct in range(num_octaves):
        octave = generate_octave(img, s, sigma, oct)
        gauss_pyr.append(octave)
        img = octave[-3][::2, ::2]

    return gauss_pyr


def generate_DoG_pyramid(gauss_pyr):
    DoG_pyr = []
    size = 512
    for gauss_octave in gauss_pyr:
        DoG_pyr.append(DoG_Octave(gauss_octave, size))
        size //= 2
    return DoG_pyr


def DoG_Octave(gauss_octave, size):
    DoG_octave = []

    for i in range(0, len(gauss_octave)-1):
        DoG = gauss_octave[i+1] - gauss_octave[i]
        DoG_octave.append(DoG)
        norm_image = np.round(((DoG - DoG.min()) * (1 / (DoG.max() - DoG.min())))* 255)
        cv2.imwrite("lena_gray_dog_octave_{}_{}.jpg".format(i, size), norm_image)

    return DoG_octave


def find_keypoint_extrema(m1, m2, m3):
    extrema = m2.item(4)
    if extrema == ndimage.maximum(m2):
        if extrema > ndimage.maximum(m1):
            if extrema > ndimage.maximum(m3):
                return 1  # max keypoint_candidate
            else:
                return -1
        else:
            return -1
    elif extrema == ndimage.minimum(m2):
        if extrema < ndimage.minimum(m1):
            if extrema < ndimage.minimum(m3):
                return 2  # min keypoint_candidate
            else:
                return -1
        else:
            return -1
    return -1


def get_candidate_keypoints_for_DoG_Octave(DoG_octave, s, image_width):
    key_point_candidates = []

    for i in range(s):
        keypoint_count = 0
        for row in range(0, image_width - 2):
            for col in range(0, image_width - 2):
                indices = np.array(
                    [ (row * image_width) + col, (row * image_width) + (col + 1), (row * image_width) + (col + 2),
                     ((row + 1) * image_width) + col, ((row + 1) * image_width) + (col + 1), ((row + 1) * image_width) + (col + 2),
                     ((row + 2) * image_width) + col, ((row + 2) * image_width) + (col + 1), ((row + 2) * image_width) + (col + 2)])

                mat1 = np.reshape(np.take(DoG_octave[i], indices), (3, 3))
                mat2 = np.reshape(np.take(DoG_octave[i + 1], indices), (3, 3))
                mat3 = np.reshape(np.take(DoG_octave[i + 2], indices), (3, 3))
                keypoint_value = find_keypoint_extrema(mat1, mat2, mat3)

                if (keypoint_value >= 1):
                    keypoint_count += 1
                    key_point_candidates.append([row , col, i])

        print("Detected " + str(keypoint_count) + " keypoints")
    return key_point_candidates


def get_candidate_keypoints(DoG_pyr, s, image_width):
    candidate_kps = []

    for DoG in DoG_pyr:
        candidate_kps.append(get_candidate_keypoints_for_DoG_Octave(DoG_pyr[0], s, image_width))
        image_width //= 2
    return candidate_kps


################################# MAIN #################################################################################
print("Welcome to the Difference of Gaussian Image creation utility.")

s = 4
num_octaves = 4
sigma = 1.6
th = 512
r_th = 10
t_c = 0.03
w = 16
image_width = 512

source_image = '/Users/kidharb/Documents/MSc/ComputerVision/project/lena_gray.png'
# read the input file
print("Reading Source Image file")
img = cv2.imread(str(source_image), cv2.IMREAD_GRAYSCALE)
print("Generating Gaussian Blur pyramids")
gauss_pyr = generate_guassian_pyramid(img, num_octaves, s, sigma)
print("Generating Difference of Gaussian")
DoG_pyr = generate_DoG_pyramid(gauss_pyr)
print("Generating Keypoint candidates")
keypoint_img=img
cadidate_keypoints = get_candidate_keypoints(DoG_pyr, s, image_width)
for kpi in cadidate_keypoints[0]:
    cv2.circle(keypoint_img, tuple(kpi), 3, (255, 255, 255), thickness=1)
cv2.imwrite("/Users/kidharb/Documents/MSc/ComputerVision/project/lena_gray_keypoints.jpg", keypoint_img)

img = cv2.imread('/Users/kidharb/Documents/MSc/ComputerVision/project/lena_gray.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, outImage=np.array([]))
cv2.imwrite('sift_keypoints.jpg', img)
