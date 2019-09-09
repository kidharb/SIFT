import cv2
import numpy as np
from scipy.spatial.distance import cdist
from scipy import ndimage

distance_metric = "euclidean"

left_img = cv2.imread('/Users/kidharb/Documents/MSc/ComputerVision/project/left.png')
left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
left_sift = cv2.xfeatures2d.SIFT_create()
left_kp, left_desc = left_sift.detectAndCompute(left_gray, None)
left_img_keypoints = cv2.drawKeypoints(left_gray, left_kp, outImage=np.array([]))
cv2.imwrite('left_keypoints.jpg', left_img_keypoints)


right_img = cv2.imread('/Users/kidharb/Documents/MSc/ComputerVision/project/right.png')
right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
right_sift = cv2.xfeatures2d.SIFT_create()
right_kp, right_desc = right_sift.detectAndCompute(right_gray, None)
right_img_keypoints = cv2.drawKeypoints(right_gray, right_kp, outImage=np.array([]))
cv2.imwrite('right_keypoints.jpg', right_img_keypoints)

dist = cdist( left_desc, right_desc, metric=distance_metric )

bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# Match descriptors.
matches = bf.match(left_desc,right_desc)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 50 matches.
img3 = cv2.drawMatches(left_img,left_kp,right_img,right_kp,matches[:50], None, flags=2)

cv2.imwrite('keypoint_matches.jpg', img3)

dst_pts = np.float32([left_kp[m.queryIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)
src_pts = np.float32([right_kp[m.trainIdx].pt for m in matches[:50]]).reshape(-1, 1, 2)

Homography, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

result = cv2.warpPerspective(right_img, Homography, (left_img.shape[1] + right_img.shape[1], left_img.shape[0]))
result[0:left_img.shape[0], 0:left_img.shape[1]] = left_img

cv2.imwrite('stitched_images.jpg', result)
