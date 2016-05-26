import cv2
import argparse
from board import Board

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread("images/" + args["image"])
image = cv2.resize(image, (int(0.2*image.shape[1]), int(0.2*image.shape[0])))

board = Board(image)
board.constructFromImageTest()
board.displayHomographyMatrix()
board.displayTransformedImage()
board.findPieces()

cv2.waitKey(0)
cv2.destroyAllWindows()