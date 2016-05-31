import os
import glob
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import cv2
import preprocessing
from defs import *


def load_data(desc, piece):
	X = None
	Y = None
	ratio = piece_to_ratio[piece]

	for piece_dir in pieces:
		piece_class = 0
		if piece == piece_dir:
			piece_class = 1

		for filename in glob.glob(os.path.join("feature_data", desc, str(ratio),
			piece_dir, "*.npy")):
			data = np.load(filename)
			if X is None:
				X = np.array(data)
				Y = np.array([piece_class])
			else:
				X = np.vstack( (X, data) )
				Y = np.hstack( (Y, [piece_class]) )
	return (X, Y)


########################################################
####    											####
####       SIFT										####
####												####
########################################################

def train_sift():
	for piece in pieces:
		X, Y = load_data("SIFT", piece)
		clf = svm.SVC(class_weight={0: 1, 1: 2})
		clf.fit(X, Y)
		joblib.dump(clf, "classifiers/classifier_sift" + piece + ".pkl")


########################################################
####    											####
####       Dense SIFT								####
####												####
########################################################

def train_dsift():
	for piece in pieces:
		X, Y = load_data("DSIFT", piece)
		clf = svm.SVC(class_weight={0: 1, 1: 2})
		clf.fit(X, Y)
		joblib.dump(clf, "classifiers/classifier_dsift" + piece + ".pkl")


########################################################
####    											####
####       HOG										####
####												####
########################################################

def train_hog():
	for piece in pieces:
		X, Y = load_data_hog(piece)
		clf = svm.SVC(class_weight=piece_weights, probability=True)
		clf.fit(X, Y)
		joblib.dump(clf, "classifiers/classifier_hog_" + piece + ".pkl")

def load_data_hog(piece):
	X = None
	Y = None

	ratio = piece_to_ratio[piece]
	for piece_dir in pieces:
		piece_class = 0
		if piece == piece_dir:
			piece_class = 1

		for filename in glob.glob(os.path.join("feature_data", "HOG", str(ratio),
			piece_dir, "*.npy")):
			data = np.load(filename)
			if X is None:
				X = np.array(data.transpose())
				Y = np.array([piece_class])
			else:
				X = np.vstack( (X, data.transpose()) )
				Y = np.hstack( (Y, [piece_class]) )
	return (X, Y)


########################################################
####    											####
####       Neural Network							####
####												####
########################################################

def train_nn():
	for piece in pieces:
		X, Y = load_data_hog(piece)
		clf = svm.SVC()
		clf.fit(X, Y)
		joblib.dump(clf, "classifiers/classifier_nn_" + piece + ".pkl")


if __name__ == "__main__":
	train_sift()
	train_dsift()
	train_hog()