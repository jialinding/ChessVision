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
	print("SIFT:")

	detector = cv2.FeatureDetector_create("SIFT")
	extractor = cv2.DescriptorExtractor_create("SIFT")
	centers = np.load("feature_data/SIFT/centers.npy")
	for piece in pieces:
		clf = joblib.load("classifiers/classifier_sift_" + piece + ".pkl")
		ratio = piece_to_ratio[piece]
		winSize = (64, int(64*ratio))

		print(piece)

		# Training set
		num_correct = float(0)
		num_images = 0
		num_true_pos = 0
		num_true_neg = 0
		for piece_dir in pieces:
			num_correct_in_piece = 0
			for filename in glob.glob(os.path.join("training_images", piece_dir, "*.jpg")):
				num_images = num_images + 1
				image = cv2.imread(filename)
				image = cv2.resize(image, (64, 128))
				image = image[int(128-64*ratio):,:]
				features = preprocessing.generateBOWFeatures(image, centers, detector, extractor)
				prediction = clf.predict(features)
				if piece == piece_dir and prediction[0] == 1:
					num_true_pos = num_true_pos + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				elif piece != piece_dir and prediction[0] == 0:
					num_true_neg = num_true_neg + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				# print(str(prediction) + " - " + str(piece_classes[piece_dir]))
			print(num_correct_in_piece)
		if num_images > 0:
			print(str(num_true_pos), str(num_true_neg))
			print(str(num_correct), str(num_images))
			print("Train accuracy: " + str(num_correct/num_images))

		# Test set
		num_correct = float(0)
		num_images = 0
		num_true_pos = 0
		num_true_neg = 0
		for piece_dir in pieces:
			num_correct_in_piece = 0
			for filename in glob.glob(os.path.join("test_images", piece_dir, "*.jpg")):
				num_images = num_images + 1
				image = cv2.imread(filename)
				image = cv2.resize(image, (64, 128))
				image = image[int(128-64*ratio):,:]
				features = preprocessing.generateBOWFeatures(image, centers, detector, extractor)
				prediction = clf.predict(features)
				if piece == piece_dir and prediction[0] == 1:
					num_true_pos = num_true_pos + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				elif piece != piece_dir and prediction[0] == 0:
					num_true_neg = num_true_neg + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				# print(str(prediction) + " - " + str(piece_classes[piece_dir]))
			print(num_correct_in_piece)
		if num_images > 0:
			print(str(num_true_pos), str(num_true_neg))
			print(str(num_correct), str(num_images))
			print("Test accuracy: " + str(num_correct/num_images))# 

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
	print("HOG:")

	for piece in pieces:
		clf = joblib.load("classifiers/classifier_hog_" + piece + ".pkl")
		ratio = piece_to_ratio[piece]
		winSize = (64, int(64*ratio))
		hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,
			winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

		print(piece)

		# Training set
		num_correct = float(0)
		num_images = 0
		num_true_pos = 0
		num_true_neg = 0
		for piece_dir in pieces:
			num_correct_in_piece = 0
			for filename in glob.glob(os.path.join("training_images", piece_dir, "*.jpg")):
				num_images = num_images + 1
				image = cv2.imread(filename)
				image = cv2.resize(image, (64, 128))
				image = image[int(128-64*ratio):,:]
				features = hog.compute(image)
				prediction = clf.predict(features.transpose())
				if piece == piece_dir and prediction[0] == 1:
					num_true_pos = num_true_pos + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				elif piece != piece_dir and prediction[0] == 0:
					num_true_neg = num_true_neg + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				# print(str(prediction) + " - " + str(piece_classes[piece_dir]))
			print(num_correct_in_piece)
		if num_images > 0:
			print(str(num_true_pos), str(num_true_neg))
			print(str(num_correct), str(num_images))
			print("Train accuracy: " + str(num_correct/num_images))

		# Test set
		num_correct = float(0)
		num_images = 0
		num_true_pos = 0
		num_true_neg = 0
		for piece_dir in pieces:
			num_correct_in_piece = 0
			for filename in glob.glob(os.path.join("test_images", piece_dir, "*.jpg")):
				num_images = num_images + 1
				image = cv2.imread(filename)
				image = cv2.resize(image, (64, 128))
				image = image[int(128-64*ratio):,:]
				features = hog.compute(image)
				prediction = clf.predict(features.transpose())
				if piece == piece_dir and prediction[0] == 1:
					num_true_pos = num_true_pos + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				elif piece != piece_dir and prediction[0] == 0:
					num_true_neg = num_true_neg + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				# print(str(prediction) + " - " + str(piece_classes[piece_dir]))
			print(num_correct_in_piece)
		if num_images > 0:
			print(str(num_true_pos), str(num_true_neg))
			print(str(num_correct), str(num_images))
			print("Test accuracy: " + str(num_correct/num_images))


########################################################
####    											####
####       Neural Network							####
####												####
########################################################

def test_nn():
	print("Neural Network:")

	for piece in pieces:
		clf = joblib.load("classifiers/classifier_nn_" + piece + ".pkl")
		ratio = piece_to_ratio[piece]
		winSize = (64, int(64*ratio))

		print(piece)

		# Training set
		num_correct = float(0)
		num_images = 0
		num_true_pos = 0
		num_true_neg = 0
		for piece_dir in pieces:
			num_correct_in_piece = 0
			for filename in glob.glob(os.path.join("training_images", piece_dir, "*.jpg")):
				num_images = num_images + 1
				image = cv2.imread(filename)
				image = cv2.resize(image, (64, 128))
				image = image[int(128-64*ratio):,:]
				features = np.reshape(image, (1, np.product(image.shape)))
				prediction = clf.predict(features)
				if piece == piece_dir and prediction[0] == 1:
					num_true_pos = num_true_pos + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				elif piece != piece_dir and prediction[0] == 0:
					num_true_neg = num_true_neg + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				# print(str(prediction) + " - " + str(piece_classes[piece_dir]))
			print(num_correct_in_piece)
		if num_images > 0:
			print(str(num_true_pos), str(num_true_neg))
			print(str(num_correct), str(num_images))
			print("Train accuracy: " + str(num_correct/num_images))

		# Test set
		num_correct = float(0)
		num_images = 0
		num_true_pos = 0
		num_true_neg = 0
		for piece_dir in pieces:
			num_correct_in_piece = 0
			for filename in glob.glob(os.path.join("test_images", piece_dir, "*.jpg")):
				num_images = num_images + 1
				image = cv2.imread(filename)
				image = cv2.resize(image, (64, 128))
				image = image[int(128-64*ratio):,:]
				features = np.reshape(image, (1, np.product(image.shape)))
				prediction = clf.predict(features)
				if piece == piece_dir and prediction[0] == 1:
					num_true_pos = num_true_pos + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				elif piece != piece_dir and prediction[0] == 0:
					num_true_neg = num_true_neg + 1
					num_correct = num_correct + 1
					num_correct_in_piece = num_correct_in_piece + 1
				# print(str(prediction) + " - " + str(piece_classes[piece_dir]))
			print(num_correct_in_piece)
		if num_images > 0:
			print(str(num_true_pos), str(num_true_neg))
			print(str(num_correct), str(num_images))
			print("Test accuracy: " + str(num_correct/num_images))


if __name__ == "__main__":
	test_sift()
	# test_dsift()
	# test_hog()
	# test_nn()