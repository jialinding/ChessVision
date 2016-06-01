import cv2
import argparse
from board import Board
import plots
from defs import *

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())

image = cv2.imread("images/" + args["image"])
image = cv2.resize(image, (int(0.2*image.shape[1]), int(0.2*image.shape[0])))

board = Board(image)
board.constructFromImage()
board.displayHomographyMatrix()
board.displayTransformedImage()

# board.detectPiecesSIFT()
board.detectPiecesHOG()
# board.detectPiecesNN()
board.displayBoard()

# plots.heatmap(board.probabilities)
# for piece in pieces:
# 	plots.heatmap(board.probabilities, piece)

cv2.waitKey(0)
cv2.destroyAllWindows()