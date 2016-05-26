import os
import glob
import numpy as np
import cv2
from sklearn.externals import joblib
from sklearn import svm
import preprocessing

def test():
	clf = joblib.load("classifiers/classifier.pkl")
	centers = np.load("feature_data/centers.npy")
	for filename in glob.glob(os.path.join("test_images", "wp", "*.jpg")):
		image = cv2.imread(filename)
		features = preprocessing.generateBOWFeatures(image, centers)
		print clf.predict(features)

if __name__ == "__main__":
	test()