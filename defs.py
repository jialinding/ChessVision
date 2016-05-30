pieces = ["empty", "wp", "wr", "bp", "br"]
pieces_aspect_ratio_1 = ["empty", "wp", "wr", "bp", "br"]

piece_classes = {"empty": 0,
				 "wp": 1,
				 "wr": 2,
				 "wn": 3,
				 "wb": 4,
				 "wq": 5,
				 "wk": 6,
				 "bp": 7,
				 "br": 8,
				 "bn": 9,
				 "bb": 10,
				 "bq": 11,
				 "bk": 12}

piece_weights = {0: 10,
				 1: 5.5,
				 2: 6,
				 7: 6,
				 8: 7}

# HOG
blockSize = (16,16)
blockStride = (8,8)
cellSize = (8,8)
nbins = 9
derivAperture = 1
winSigma = 4.
histogramNormType = 0
L2HysThreshold = 2.0000000000000001e-01
gammaCorrection = 0
nlevels = 64