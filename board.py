import cv2
import numpy as np
import utils

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
		self.board = None
		self.homography = None
		self.transformedBoard = None


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
		self.transformedBoard = cv2.warpPerspective(self.image, self.homography, (640, 640))


	def computeHomography2(self, boardPts, imagePts):
		self.homography = cv2.getPerspectiveTransform(boardPts[:,0:2], imagePts)
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


	def findPieces(self):
		edges = cv2.Canny(self.transformedBoard, 50, 200)
		cv2.imshow("Canny", edges)