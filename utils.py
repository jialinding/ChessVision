import numpy as np

def convertTo8U(x):
	ret = 256*x
	if ret < 0:
		ret = 0
	if ret > 255:
		ret = 155
	return np.uint8(ret)

def thresholdBetween(image, minVal, maxVal):
	clone = image.copy()
	for i in xrange(image.shape[0]):
		for j in xrange(image.shape[1]):
			if clone[i,j] > minVal and clone[i,j] < maxVal:
				clone[i, j] = 255
			else:
				clone[i, j] = 0
	return clone

def image1pts():
	return np.array([(325, 821, 919, 233),
					(36, 24, 517, 515),
					(1, 1, 1, 1)], np.float32)