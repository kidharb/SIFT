import cv2
import numpy as np
from scipy import ndimage

def keypoint(m1,m2,m3):
    keypoint_candidate = m2.item(4)
    if keypoint_candidate >= ndimage.maximum(m2):
        if keypoint_candidate > ndimage.maximum(m1):
            if keypoint_candidate > ndimage.maximum(m3):
                return keypoint_candidate
    return -1

mat1 = np.array([[22, 217, 64],
          [134, 35, 65],
          [127, 84, 12]])

mat2 = np.array([[25, 24, 68],
          [16, 222, 6],
          [18, 83, 14]])

mat3 = np.array([[212, 222, 62],
          [156, 36, 61],
          [143, 34, 122]])


big_array = np.random.randint(255, size=(512, 512))

for row in range(0, 510):
    for col in range(0, 510):
        indices = np.array([(row*9)+col, (row*9)+col+1, (row*9)+col+2, (row*9)+col+9, (row*9)+col+10, (row*9)+col+11, (row*9)+col+18, (row*9)+col+19, (row*9)+col+20])
        small_arr = np.reshape(np.take(big_array, indices),(3,3))


keypoint_value = keypoint(mat1,mat2,mat3)
if (keypoint_value == -1):
    print("No keypoint detected")
else:
    print("Keypoint of " + str(keypoint_value) + " at position" + str(ndimage.maximum_position(mat2)))
