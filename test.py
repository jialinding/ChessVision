import os
import glob
import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn import svm
import preprocessing
from defs import *

########################################################
####    											####
####       SIFT										####
####												####
########################################################

def test_sift():
	clf = joblib.load("classifiers/classifier_sift.pkl")
	centers = np.load("feature_data/SIFT/centers.npy")
	detector = cv2.FeatureDetector_create("SIFT")
	extractor = cv2.DescriptorExtractor_create("SIFT")

	for filename in glob.glob(os.path.join("test_images", "wp", "*.jpg")):
		image = cv2.imread(filename)
		features = preprocessing.generateBOWFeatures(image, centers, detector, extractor)
		print clf.predict(features)

########################################################
####    											####
####       Dense SIFT								####
####												####
########################################################

def test_dsift():
	clf = joblib.load("classifiers/classifier_dsift.pkl")
	centers = np.load("feature_data/DSIFT/centers.npy")
	detector = cv2.FeatureDetector_create("Dense")
	extractor = cv2.DescriptorExtractor_create("SIFT")

	for filename in glob.glob(os.path.join("test_images", "wp", "*.jpg")):
		image = cv2.imread(filename)
		features = preprocessing.generateBOWFeatures(image, centers, detector, extractor)
		print clf.predict(features)

########################################################
####    											####
####       HOG										####
####												####
########################################################

def test_hog():

	# Aspect ratio 1:1 - pawns, rooks
	clf = joblib.load("classifiers/classifier_hog_1.pkl")
	winSize = (64, 64)
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
		histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

	for filename in glob.glob(os.path.join("test_images", "wp", "*.jpg")):
		image = cv2.imread(filename)
		image = cv2.resize(image, winSize)
		features = hog.compute(image)
		print clf.predict(features.transpose())


if __name__ == "__main__":
	test_sift()
	test_dsift()
	test_hog()