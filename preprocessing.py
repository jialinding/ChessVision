import os
import glob
import cv2
import numpy as np
from sklearn import cluster
from sklearn import metrics

def preprocessing():
	n_clusters = 8
	centers = generateClusterCenters(n_clusters)
	np.save("feature_data/centers", centers)

	for piece_dir in ["empty", "wp"]:
		for filename in glob.glob(os.path.join("training_images", piece_dir, "*.jpg")):
			image = cv2.imread(filename)
			features = generateBOWFeatures(image, centers)
			np.save("feature_data/" + piece_dir + "/" + os.path.basename(filename), features)


def generateClusterCenters(n_clusters):
	sift_detector = cv2.FeatureDetector_create("SIFT")
	sift_extractor = cv2.DescriptorExtractor_create("SIFT")
	features = None
	for piece_dir in ["emtpy", "wp"]:
		for filename in glob.glob(os.path.join("raw_piece_images", piece_dir, "*.jpg")):
			image = cv2.imread(filename)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			kp = sift_detector.detect(gray)
			kp, des = sift_extractor.compute(gray, kp)
			if features is None:
				features = np.array(des)
			else:
				features = np.vstack((features, des))
	k_means = cluster.KMeans(n_clusters)
	k_means.fit(features)
	return k_means.cluster_centers_


def generateBOWFeatures(image, centers):
	sift_detector = cv2.FeatureDetector_create("SIFT")
	sift_extractor = cv2.DescriptorExtractor_create("SIFT")

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kp = sift_detector.detect(gray)
	kp, des = sift_extractor.compute(gray, kp)
	distances = metrics.pairwise.pairwise_distances(des, centers)
	best_centers = np.argmin(distances, axis=1)

	num_centers = centers.shape[0]
	histogram = np.zeros( (1, num_centers) )
	for i in best_centers:
		histogram[0,i] = histogram[0,i] + 1
	histogram = histogram/np.sum(histogram)

	return histogram


if __name__ == "__main__":
	preprocessing()