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


def localize_keypoint(D, x, y, s):
    dx = (D[y, x + 1, s] - D[y, x - 1, s]) / 2.
    dy = (D[y + 1, x, s] - D[y - 1, x, s]) / 2.
    ds = (D[y, x, s + 1] - D[y, x, s - 1]) / 2.
    dxx = D[y, x + 1, s] - 2 * D[y, x, s] + D[y, x - 1, s]
    dxy = ((D[y + 1, x + 1, s] - D[y + 1, x - 1, s]) - (D[y - 1, x + 1, s] - D[y - 1, x - 1, s])) / 4.
    dxs = ((D[y, x + 1, s + 1] - D[y, x - 1, s + 1]) - (D[y, x + 1, s - 1] - D[y, x - 1, s - 1])) / 4.
    dyy = D[y + 1, x, s] - 2 * D[y, x, s] + D[y - 1, x, s]
    dys = ((D[y + 1, x, s + 1] - D[y - 1, x, s + 1]) - (D[y + 1, x, s - 1] - D[y - 1, x, s - 1])) / 4.
    dss = D[y, x, s + 1] - 2 * D[y, x, s] + D[y, x, s - 1]
    J = np.array([dx, dy, ds])
    HD = np.array([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
    offset = -LA.inv(HD).dot(J)
    return offset, J, HD[:2, :2], x, y, s


def find_keypoints_for_DoG(D, R_th, t_c, s=2):
    candidates = get_candidate_keypoints(D)
    keypoints = []
    for i, cand in enumerate(candidates):
        y, x, s = cand[0], cand[1], cand[2]
        offset, J, H, x, y, s = localize_keypoint(D, x, y, s)
        contrast = D[y, x, s] + .5 * J.dot(offset)
        if abs(contrast) < t_c: continue
        w, v = LA.eig(H)
        r = w[1] / w[0]
        R = (r + 1) ** 2 / r
        if R > R_th: continue
        kp = np.array([x, y, s]) + offset
        keypoints.append(kp)
    return np.array(keypoints)


def generate_octave(img, s, sigma, oct):
    k = 2 ** (1 / s)
    octave = []
    for i in range(s + 3):
        octave.append(cv2.GaussianBlur(img, (3, 3), sigma, sigma))
        img = octave[i]
        sigma = k * sigma
        cv2.imwrite("lena_gray_gaussian_{}_{}.jpg".format(i, oct), octave[i] * 255)

    return octave


# Takes in root image, s, and sigma and outputs a pyramid of image blur
def generate_guassian_pyramid(img, num_octaves, s, sigma):
    gauss_pyr = []
    # normalize image
    img = cv2.normalize(img, None, norm_type=cv2.NORM_INF, dtype=cv2.CV_32F)

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
    #       cv2.imwrite("lena_gray_DoG_{}.jpg".format(i), cv2.normalize(DoG_pyr[gauss_octave], None, alpha=0, beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U))
    return DoG_pyr


def DoG_Octave(gauss_octave, size):
    DoG_octave = []

    for i in range(0, len(gauss_octave)-1):
        DoG_octave.append(gauss_octave[i+1] - gauss_octave[i])
        cv2.imwrite("lena_gray_dog_octave_{}_{}.jpg".format(i, size), DoG_octave[i - 1] * 255)

    return DoG_octave
    # return np.concatenate([oct[:, :, np.newaxis] for oct in DoG_octave], axis=2)


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
    keypoint_count = 0
    key_point_candidates = []

    for i in range(s):
        for row in range(0, image_width - 2):
            for col in range(0, image_width - 2):
                indices = np.array(
                    [(row * image_width) + col, (row * image_width) + col + 1, (row * image_width) + col + 2,
                     ((row + 1) * image_width) + col, ((row + 1) * image_width) + col + 1,
                     ((row + 1) * image_width) + col + 2,
                     ((row + 2) * image_width) + col, ((row + 2) * image_width) + col + 1,
                     ((row + 2) * image_width) + col + 2])
                mat1 = np.reshape(np.take(DoG_octave[i], indices), (3, 3))
                mat2 = np.reshape(np.take(DoG_octave[i + 1], indices), (3, 3))
                mat3 = np.reshape(np.take(DoG_octave[i + 2], indices), (3, 3))
                keypoint_value = find_keypoint_extrema(mat1, mat2, mat3)

                if (keypoint_value >= 1):
                    keypoint_count += 1
                    key_point_candidates.append([row, col, i])
                    cv2.circle(DoG_octave[i + 1], (row, col), 3, (255, 255, 255), thickness=1)

        # cv2.imwrite("lena_gray_keypoints_{}_{}.jpg".format(image_width, i), DoG_octave[i+1] * 255)

    print("Detected " + str(keypoint_count) + " keypoints")
    return key_point_candidates


def get_candidate_keypoints(DoG_pyr, s, image_width):
    candidate_kps = []
    for DoG in DoG_pyr:
        candidate_kps.append(get_candidate_keypoints_for_DoG_Octave(DoG, s, image_width))
        image_width //= 2
    return candidate_kps


################################# MAIN #################################################################################
print("Welcome to the Difference of Gaussian Image creation utility.")

s = 3
num_octaves = 4
s0 = 1.3
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
cadidate_keypoints = get_candidate_keypoints(DoG_pyr, s, image_width)

img = cv2.imread('/Users/kidharb/Documents/MSc/ComputerVision/project/lena_gray.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray, None)
img = cv2.drawKeypoints(gray, kp, outImage=np.array([]))
cv2.imwrite('sift_keypoints.jpg', img)
