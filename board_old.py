import cv2
import numpy as np
from utils import convertTo8U
from math import *

img = cv2.imread('img1.jpg',0)
im = cv2.resize(img, (int(0.2*img.shape[1]), int(0.2*img.shape[0])))

# Otsu's thresholding - doesn't work
# ret,thresh = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow('otsu', thresh)

''' Edge Detection '''
# TODO: tune the minVal and maxVal parameters
edges = cv2.Canny(im, 100, 300)
# cv2.imshow('edges', edges)

''' Corner Detection '''
# Harris corner detection
dst = cv2.cornerHarris(im,5,5,0.01)
# np.savetxt('test.txt', dst)
f = np.vectorize(convertTo8U)
dst = f(dst)
# np.savetxt('dst.txt', dst)
# img_grey = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

# dst = cv2.convertScaleAbs(dst)
cv2.imshow('scaleabs', dst)

ret,thresh = cv2.threshold(dst,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('corners', thresh)

# lines = cv2.HoughLines(thresh, 1, np.pi/180, 10)
# top_lines = lines[:18]
# print(top_lines.shape)

# for i in xrange(len(top_lines)):
# 	line = top_lines[i][0]
# 	rho = line[0]
# 	theta = line[1]
# 	print(theta)
# 	a, b = cos(theta), sin(theta)
# 	x0, y0 = a*rho, b*rho
# 	x1 = int(x0 + 1000*(-b))
# 	y1 = int(y0 + 1000*(a))
# 	x2 = int(x0 - 1000*(-b))
# 	y2 = int(y0 - 1000*(a))
# 	pt1 = (x1, y1)
# 	pt2 = (x2, y2)
# 	cv2.line(thresh, pt1, pt2, (255, 0, 0))

# cv2.imshow('lines', thresh)

''' Hough Transform '''

# Identify chessboard lines with Hough transform
# TODO: tune the threshold parameter (fourth parameter)
# lines = cv2.HoughLines(edges, 1, np.pi/180, 250)
# top_lines = lines[:18]
# print(top_lines.shape)

# for i in xrange(len(top_lines)):
# 	line = top_lines[i][0]
# 	rho = line[0]
# 	theta = line[1]
# 	print(theta)
# 	a, b = cos(theta), sin(theta)
# 	x0, y0 = a*rho, b*rho
# 	x1 = int(x0 + 1000*(-b))
# 	y1 = int(y0 + 1000*(a))
# 	x2 = int(x0 - 1000*(-b))
# 	y2 = int(y0 - 1000*(a))
# 	pt1 = (x1, y1)
# 	pt2 = (x2, y2)
# 	cv2.line(im, pt1, pt2, (255, 0, 0))

# cv2.imshow('lines', im)
# cv2.waitKey(0)

# Use histogram of line orientations to identify two sets that represent
# edges going along ranks and files


# Remove extraneous lines from the sets

# Find one square

# Calculate homography

# Extropolate to other squares and update homography accordingly

cv2.waitKey(0)