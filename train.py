import os
import glob
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import cv2
import preprocessing
from defs import *


def load_data(desc):
	X = None
	Y = None
	for piece_dir in pieces:
		piece_class = piece_classes[piece_dir]

		for filename in glob.glob(os.path.join("feature_data", desc, piece_dir, "*.npy")):
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
	X, Y = load_data("SIFT")
	weights = {0: 10, 1: 2}
	clf = svm.SVC(class_weight=weights)
	clf.fit(X, Y)
	joblib.dump(clf, "classifiers/classifier_sift.pkl")


########################################################
####    											####
####       Dense SIFT								####
####												####
########################################################

def train_dsift():
	X, Y = load_data("DSIFT")
	clf = svm.SVC(class_weight=piece_weights)
	clf.fit(X, Y)
	joblib.dump(clf, "classifiers/classifier_dsift.pkl")


########################################################
####    											####
####       HOG										####
####												####
########################################################

def train_hog():
	aspect_ratios = ["1"]

	for ratio in aspect_ratios:
		X, Y = load_data_hog(ratio)
		clf = svm.SVC(class_weight=piece_weights)
		clf.fit(X, Y)
		joblib.dump(clf, "classifiers/classifier_hog_" + ratio + ".pkl")

def load_data_hog(ratio):
	ratio_to_piece = {"1": pieces_aspect_ratio_1}

	X = None
	Y = None
	for piece_dir in ratio_to_piece[ratio]:
		piece_class = piece_classes[piece_dir]

		for filename in glob.glob(os.path.join("feature_data", "HOG", ratio,
			piece_dir, "*.npy")):
			data = np.load(filename)
			if X is None:
				X = np.array(data.transpose())
				Y = np.array([piece_class])
			else:
				X = np.vstack( (X, data.transpose()) )
				Y = np.hstack( (Y, [piece_class]) )
	return (X, Y)


if __name__ == "__main__":
	train_sift()
	train_dsift()
	train_hog()