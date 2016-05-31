pieces = ["empty", "pawn", "knight", "queen"] # change weights
piece_weights = {"empty": {0: 64, 1: 192},
				 "pawn": {0: 128, 1: 128},
				 "knight": {0: 32, 1: 224},
				 "queen": {0: 32, 1: 224}}

# piece_weights = {"empty": {0: 64, 1: 32},
# 				 "knight": {0: 32, 1: 64}}

# piece_weights = {"empty": {0: 64, 1: 128},
# 				 "pawn": {0: 128, 1: 64}}

aspect_ratios = [1, 1.25, 1.5, 1.75, 2]

piece_to_ratio = {"empty": 1,
				  "pawn": 1,
				  "knight": 1.25,
				  "queen": 1.75}

piece_classes = {"empty": 0,
				 "pawn": 1,
				 "knight": 2,
				 "bishop": 3,
				 "rook": 4,
				 "queen": 5,
				 "king": 6}

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