import os
import glob
import cv2
import numpy as np
from sklearn import cluster
from sklearn import metrics
from defs import *

def generateClusterCenters(n_clusters, detector, extractor):
	features = None
	for piece_dir in ["emtpy", "wp"]:
		for filename in glob.glob(os.path.join("raw_piece_images", piece_dir, "*.jpg")):
			image = cv2.imread(filename)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			kp = detector.detect(gray)
			kp, des = extractor.compute(gray, kp)
			if features is None:
				features = np.array(des)
			else:
				features = np.vstack((features, des))
	k_means = cluster.KMeans(n_clusters)
	k_means.fit(features)
	return k_means.cluster_centers_


def generateBOWFeatures(image, centers, detector, extractor):
	num_centers = centers.shape[0]
	histogram = np.zeros( (1, num_centers) )

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kp = detector.detect(gray)
	best_centers = []

	if not kp:
		return histogram

	kp, des = extractor.compute(gray, kp)
	distances = metrics.pairwise.pairwise_distances(des, centers)
	best_centers = np.argmin(distances, axis=1)
	
	for i in best_centers:
		histogram[0,i] = histogram[0,i] + 1
	histogram = histogram/np.sum(histogram)

	return histogram

########################################################
####    											####
####       SIFT										####
####												####
########################################################

# TODO:
# adjust n_clusters
# class weights
# normalize training examples per class

def preprocessing_sift():
	sift_detector = cv2.FeatureDetector_create("SIFT")
	sift_extractor = cv2.DescriptorExtractor_create("SIFT")

	n_clusters = 8
	centers = generateClusterCenters(n_clusters, sift_detector, sift_extractor)
	np.save("feature_data/SIFT/centers", centers)

	for piece_dir in ["empty", "wp"]:
		for filename in glob.glob(os.path.join("training_images", piece_dir, "*.jpg")):
			image = cv2.imread(filename)
			features = generateBOWFeatures(image, centers, sift_detector, sift_extractor)
			np.save("feature_data/SIFT/" + piece_dir + "/" + os.path.basename(filename), features)


########################################################
####    											####
####       Dense SIFT								####
####												####
########################################################

# add pyramid depth

def preprocessing_dsift():
	dsift_detector = cv2.FeatureDetector_create("Dense")
	dsift_extractor = cv2.DescriptorExtractor_create("SIFT")

	n_clusters = 8
	centers = generateClusterCenters(n_clusters, dsift_detector, dsift_extractor)
	np.save("feature_data/DSIFT/centers", centers)

	for piece_dir in ["empty", "wp"]:
		for filename in glob.glob(os.path.join("training_images", piece_dir, "*.jpg")):
			image = cv2.imread(filename)
			features = generateBOWFeatures(image, centers, dsift_detector, dsift_extractor)
			np.save("feature_data/DSIFT/" + piece_dir + "/" + os.path.basename(filename), features)


########################################################
####    											####
####       HOG										####
####												####
########################################################

# TODO:
# Adjust HOG parameters

def preprocessing_hog():

	# Aspect ratio 1:1 - pawns, rooks
	winSize = (64, 64)
	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
		histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

	for piece_dir in ["empty", "wp"]:
		for filename in glob.glob(os.path.join("training_images", piece_dir, "*.jpg")):
			image = cv2.imread(filename)
			image = cv2.resize(image, winSize)
			features = hog.compute(image)
			np.save("feature_data/HOG/1/" + piece_dir + "/" + os.path.basename(filename),
				features)


if __name__ == "__main__":
	# preprocessing_sift()
	# preprocessing_dsift()
	preprocessing_hog()