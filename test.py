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
	print("SIFT: predict - actual")
	clf = joblib.load("classifiers/classifier_sift.pkl")
	centers = np.load("feature_data/SIFT/centers.npy")
	detector = cv2.FeatureDetector_create("SIFT")
	extractor = cv2.DescriptorExtractor_create("SIFT")

	for piece_dir in pieces:
		num_correct = float(0)
		num_images = 0
		for filename in glob.glob(os.path.join("test_images", piece_dir, "*.jpg")):
			num_images = num_images + 1
			image = cv2.imread(filename)
			features = preprocessing.generateBOWFeatures(image, centers, detector, extractor)
			prediction = clf.predict(features)
			if prediction[0] == piece_classes[piece_dir]:
				num_correct = num_correct + 1
			print(str(prediction) + " - " + str(piece_classes[piece_dir]))
		if num_images > 0:
			print("Accuracy for " + piece_dir + ": " + str(num_correct/num_images))

########################################################
####    											####
####       Dense SIFT								####
####												####
########################################################

def test_dsift():
	print("DSIFT: predict - actual")
	clf = joblib.load("classifiers/classifier_dsift.pkl")
	centers = np.load("feature_data/DSIFT/centers.npy")
	detector = cv2.FeatureDetector_create("Dense")
	extractor = cv2.DescriptorExtractor_create("SIFT")

	for piece_dir in pieces:
		num_correct = float(0)
		num_images = 0
		for filename in glob.glob(os.path.join("test_images", piece_dir, "*.jpg")):
			num_images = num_images + 1
			image = cv2.imread(filename)
			features = preprocessing.generateBOWFeatures(image, centers, detector, extractor)
			prediction = clf.predict(features)
			if prediction[0] == piece_classes[piece_dir]:
				num_correct = num_correct + 1
			print(str(prediction) + " - " + str(piece_classes[piece_dir]))
		if num_images > 0:
			print("Accuracy for " + piece_dir + ": " + str(num_correct/num_images))

########################################################
####    											####
####       HOG										####
####												####
########################################################

def test_hog():
	print("HOG: predict - actual")

	# Aspect ratio 1:1 - pawns, rooks
	clf = joblib.load("classifiers/classifier_hog_1.pkl")
	winSize = (64, 64)
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
		histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

	for piece_dir in pieces_aspect_ratio_1:
		num_correct = float(0)
		num_images = 0
		for filename in glob.glob(os.path.join("training_images", piece_dir, "*.jpg")):
			num_images = num_images + 1
			image = cv2.imread(filename)
			image = cv2.resize(image, winSize)
			features = hog.compute(image)
			prediction = clf.predict(features.transpose())
			if prediction[0] == piece_classes[piece_dir]:
				num_correct = num_correct + 1
			# print(str(prediction) + " - " + str(piece_classes[piece_dir]))
		if num_images > 0:
			print("Train accuracy for " + piece_dir + ": " + str(num_correct/num_images))

		num_correct = float(0)
		num_images = 0
		for filename in glob.glob(os.path.join("test_images", piece_dir, "*.jpg")):
			num_images = num_images + 1
			image = cv2.imread(filename)
			image = cv2.resize(image, winSize)
			features = hog.compute(image)
			prediction = clf.predict(features.transpose())
			if prediction[0] == piece_classes[piece_dir]:
				num_correct = num_correct + 1
			print(str(prediction) + " - " + str(piece_classes[piece_dir]))
		if num_images > 0:
			print("Test accuracy for " + piece_dir + ": " + str(num_correct/num_images))


if __name__ == "__main__":
	# test_sift()
	# test_dsift()
	test_hog()