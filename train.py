import os
import glob
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import cv2
import preprocessing

def train():
	X, Y = load_data()
	clf = svm.SVC()
	clf.fit(X, Y)
	joblib.dump(clf, "classifiers/classifier.pkl")


def load_data():
	X = None
	Y = None
	piece_class = 0
	for piece_dir in ["empty", "wp"]:
		if piece_dir == "empty":
			piece_class = 13
		elif piece_dir == "wp":
			piece_class = 1
		for filename in glob.glob(os.path.join("feature_data", piece_dir, "*.npy")):
			data = np.load(filename)
			if X is None:
				X = np.array(data)
				Y = np.array([piece_class])
			else:
				X = np.vstack( (X, data) )
				Y = np.hstack( (Y, [piece_class]) )
	return (X, Y)


if __name__ == "__main__":
	train()