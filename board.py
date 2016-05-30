import cv2
import numpy as np
import utils
from sklearn import svm
from sklearn.externals import joblib
import preprocessing
from defs import *

aspect_ratios = [1]

boardPts = np.zeros( (3, 4), np.float32)
numPts = 0

def selectPoints(event, x, y, flags, param):
	global boardPts, numPts

	if numPts > 3:
		return
	if event == cv2.EVENT_LBUTTONDOWN:
		print('Point selected: (' + str(x) + ', ' + str(y) + ')')
		boardPts[0, numPts] = x
		boardPts[1, numPts] = y
		boardPts[2, numPts] = 1
		numPts = numPts + 1

class Board:
	def __init__(self, image):
		self.image = image
		self.board = np.zeros( (8, 8) )
		self.homography = None
		self.homography_inv = None
		self.transformedBoard = None

		# Classification
		self.classifier_sift = joblib.load("classifiers/classifier_sift.pkl")
		self.classifier_dsift = joblib.load("classifiers/classifier_dsift.pkl")
		self.classifier_hog_1 = joblib.load("classifiers/classifier_hog_1.pkl")
		
		self.centers_sift = np.load("feature_data/SIFT/centers.npy")
		self.centers_dsift = np.load("feature_data/DSIFT/centers.npy")

	########################################################
	####    											####
	####       BOARD RECOGNITION						####
	####												####
	########################################################
	
	def constructFromImage(self):
		clone = self.image.copy()

		cv2.namedWindow("image")
		cv2.setMouseCallback("image", selectPoints)

		cv2.imshow("image", self.image)
		while True:
			key = cv2.waitKey(0)
			
			if key == ord("r"):
				self.image = clone.copy()

			if key == ord("c"):
				break

		imagePts = np.array([(0, 640, 640, 0),
					(0, 0, 640, 640)],
					np.float32)
		
		# self.computeHomography(boardPts, imagePts)
		self.computeHomography2(np.transpose(boardPts), np.transpose(imagePts))


	def constructFromImageTest(self):
		global boardPts
		boardPts = utils.image1pts()

		imagePts = np.array([(0, 640, 640, 0),
					(0, 0, 640, 640)],
					np.float32)
		
		# self.computeHomography(boardPts, imagePts)
		self.computeHomography2(np.transpose(boardPts), np.transpose(imagePts))


	def computeHomography(self, boardPts, imagePts):
		''' Using SVD technique from lecture 3 '''
		numPts = boardPts.shape[1]
		P = np.zeros( (2*numPts, 9) )
		for i in xrange(numPts):
			P[2*i, 0] = boardPts[0, i]
			P[2*i, 1] = boardPts[1, i]
			P[2*i, 2] = 1
			P[2*i, 6] = -imagePts[0, i]*boardPts[0, i]
			P[2*i, 7] = -imagePts[0, i]*boardPts[1, i]
			P[2*i, 8] = -imagePts[0, i]

			P[2*i+1, 3] = boardPts[0, i]
			P[2*i+1, 4] = boardPts[1, i]
			P[2*i+1, 5] = 1
			P[2*i+1, 6] = -imagePts[1, i]*boardPts[0, i]
			P[2*i+1, 7] = -imagePts[1, i]*boardPts[1, i]
			P[2*i+1, 8] = -imagePts[1, i]

		U, S, V = np.linalg.svd(P)
		m = V[:,-1]

		H = np.zeros( (3, 3) )
		H[:,0] = m[0:3]
		H[:,1] = m[3:6]
		H[:,2] = m[6:9]
		self.homography = H
		self.homography_inv = np.linalg.inv(H)
		self.transformedBoard = cv2.warpPerspective(self.image, self.homography, (640, 640))


	def computeHomography2(self, boardPts, imagePts):
		self.homography = cv2.getPerspectiveTransform(boardPts[:,0:2], imagePts)
		self.homography_inv = np.linalg.inv(self.homography)
		self.transformedBoard = cv2.warpPerspective(self.image, self.homography, (640, 640))


	def displayTransformedImage(self, displayCorners=True):
		# cv2.imshow("Original Image", self.image)
		clone = self.transformedBoard.copy()
		if displayCorners:
			for i in xrange(7):
				for j in xrange(7):
					cv2.circle(clone, ((i+1)*80, (j+1)*80), 5, (255, 0, 0), -1)
		cv2.imshow("Transformed Image", clone)


	def displayHomographyMatrix(self):
		print(self.homography)
		point = np.transpose(boardPts[:,2])
		print(self.homography.dot(point))


	def binarizeTransformedBoard(self):
		gray = cv2.cvtColor(self.transformedBoard, cv2.COLOR_BGR2GRAY)
		# ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		ret, thresh = cv2.threshold(gray, 0, 100, cv2.THRESH_BINARY)
		cv2.imshow("Binarized", thresh)

	########################################################
	####    											####
	####       PIECE RECOGNITION						####
	####												####
	########################################################

	def detectPieces(self, descriptor):
		self.board = np.zeros( (8, 8) )
		for r in xrange(8):
			for f in xrange(8):
				possible_bb = self.getPossibleBoundingBoxes(r, f)
				for i in xrange(possible_bb.shape[0]):
					bb = possible_bb[i,:]
					# if r == 3 and f == 4:
					# 	print(bb)
					# 	subim = self.image[bb[2]:bb[3],bb[0]:bb[1]]
					# 	cv2.imwrite('testimage.jpg', subim)
					piece_class = self.identifyPiece(bb, descriptor)
					if piece_class != 0:
						self.board[7-r, f] = piece_class
						break


	def getPossibleBoundingBoxes(self, r, f):
		corner_pts = np.ones( (3, 4) )
		corner_pts[1, 0] = 640 - (r+1)*80
		corner_pts[0, 0] = f*80
		corner_pts[1, 1] = 640 - (r+1)*80
		corner_pts[0, 1] = (f+1)*80
		corner_pts[1, 2] = 640 - r*80
		corner_pts[0, 2] = (f+1)*80
		corner_pts[1, 3] = 640 - r*80
		corner_pts[0, 3] = f*80

		pts = self.homography_inv.dot(corner_pts)
		tlc = pts[0:2,0]/pts[2,0]
		trc = pts[0:2,1]/pts[2,1]
		brc = pts[0:2,2]/pts[2,2]
		blc = pts[0:2,3]/pts[2,3]

		sq_bottom = min(brc[1], blc[1])
		sq_top = max(trc[1], tlc[1])
		height = sq_bottom - sq_top

		height_scaling_factors = np.array([1.5])
		possible_bb = np.zeros( (height_scaling_factors.shape[0], 4), np.int )
		for index, scale in np.ndenumerate(height_scaling_factors):
			possible_bb[index[0], 0] = np.array([ int(blc[0]) ]) #x1
			possible_bb[index[0], 1] = np.array([ int(brc[0]) ]) #x2
			possible_bb[index[0], 2] = np.array([ max(int(sq_bottom-scale*height), 0) ]) #y1
			possible_bb[index[0], 3] = np.array([ int(sq_bottom) ]) #y2

		return possible_bb


	def identifyPiece(self, bounding_box, descriptor):
		x1 = bounding_box[0]
		x2 = bounding_box[1]
		y1 = bounding_box[2]
		y2 = bounding_box[3]
		subimage = self.image[y1:y2, x1:x2]

		sift_detector = cv2.FeatureDetector_create("SIFT")
		dsift_detector = cv2.FeatureDetector_create("Dense")
		sift_extractor = cv2.DescriptorExtractor_create("SIFT")

		if descriptor == "SIFT":
			features = preprocessing.generateBOWFeatures(subimage, self.centers_sift,
				sift_detector, sift_extractor)
			return self.classifier_sift.predict(features)
		elif descriptor == "DSIFT":
			features = preprocessing.generateBOWFeatures(subimage, self.centers_dsift,
				dsift_detector, sift_extractor)
			return self.classifier_dsift.predict(features)
		else:
			# Update when more aspect ratios appear
			winSize = (64, 64)
			hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,
				winSigma,histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
			subimage = cv2.resize(subimage, winSize)
			features = hog.compute(subimage)
			return self.classifier_hog_1.predict(features.transpose())


	def displayBoard(self):
		print(self.board)


	def findPiecesCanny(self):
		edges = cv2.Canny(self.transformedBoard, 50, 200)
		cv2.imshow("Canny", edges)