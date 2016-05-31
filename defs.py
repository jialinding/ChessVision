pieces = ["empty", "wp"] # change weights
piece_weights = {0: 1, 1: 1}

aspect_ratios = [1, 1.5, 2]

piece_to_ratio = {"empty": 1,
				  "wp": 1,
				  "bp": 1}

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