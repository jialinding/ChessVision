import os
import glob
import cv2

def preprocess():
	for piece_dir in ["wp"]:
		for filename in glob.glob(os.path.join("raw_piece_images", piece_dir, "*.jpg")):
			image = cv2.imread(filename)
			# thresh = createPieceMask(image, "white")
			# cv2.imshow(filename, thresh)
			edges = createPieceMaskEdge(image, "white")
			cv2.imshow(filename, edges)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


def createPieceMaskThreshold(image, color):
	clone = image.copy()
	if color == "white":
		gray = cv2.cvtColor(clone, cv2.COLOR_BGR2GRAY)
		ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	return thresh


def createPieceMaskEdge(image, color):
	clone = image.copy()
	if color == "white":
		edges = cv2.Canny(clone, 50, 100)
	return edges


if __name__ == "__main__":
	preprocess()