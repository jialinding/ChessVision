import cv2
import numpy as np
from math import *

img = cv2.imread('img1.jpg',0)
im = cv2.resize(img, (int(0.2*img.shape[1]), int(0.2*img.shape[0])))

# Edge detection
edges = cv2.Canny(im, 100, 200)
cv2.imshow('edges', edges)
# cv2.waitKey(0)

# Identify chessboard lines with Hough transform
lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
top_lines = lines[:18]
print(top_lines.shape)

for i in xrange(len(top_lines)):
	line = top_lines[i][0]
	rho = line[0]
	theta = line[1]
	print(theta)
	a, b = cos(theta), sin(theta)
	x0, y0 = a*rho, b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	pt1 = (x1, y1)
	pt2 = (x2, y2)
	cv2.line(im, pt1, pt2, (255, 0, 0))

cv2.imshow('lines', im)
cv2.waitKey(0)

# Use histogram of line orientations to identify two sets that represent
# edges going along ranks and files

# Remove extraneous lines from the sets

# Find one square

# Calculate homography

# Extropolate to other squares and update homography accordingly