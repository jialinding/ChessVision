import cv2
from board import Board
import plots
from defs import *
import numpy as np

test_results = np.zeros( (6, 6) )
confusion_matrix = np.zeros( (7, 7) )

for i, boards in enumerate(sets):
	num_pieces = (i+1)*5

	averages = np.zeros ( (6, 1) )

	for j, pair in enumerate(boards):
		print(i,j)
		set_num = j + 1
		pts = pair[0]
		correct_board = pair[1]

		image = cv2.imread("images/" + str(num_pieces) + "_" + str(set_num) + ".jpg")
		image = cv2.resize(image, (int(0.2*image.shape[1]), int(0.2*image.shape[0])))

		board = Board(image)
		board.constructFromImageWithPoints(pts)

		(ce_sift, da_sift, ca_sift) = board.detectPiecesSIFT(correct_board)
		averages[0] += ce_sift
		averages[1] += ca_sift
		averages[2] += da_sift

		(ce_hog, da_hog, ca_hog, confusion) = board.detectPiecesHOG(correct_board)
		averages[3] += ce_hog
		averages[4] += ca_hog
		averages[5] += da_hog
		confusion_matrix = np.add(confusion_matrix, confusion)

	averages = averages/5

	for b in xrange(6):
		test_results[b,i] = averages[b]

np.savetxt('test_results.csv', test_results, delimiter=",")
total_averages = np.mean(test_results, axis=1)
np.savetxt('total_averages.csv', total_averages, delimiter=",")
np.savetxt('confusion_matrix.csv', confusion_matrix, delimiter=",")


# plots.heatmap(board.probabilities)
# for piece in pieces:
# 	plots.heatmap(board.probabilities, piece)

# cv2.waitKey(0)
# cv2.destroyAllWindows()