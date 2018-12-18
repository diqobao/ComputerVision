import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
	"""
	To start with, you might want to simply use normalized patches as your
	local feature. This is very simple to code and works OK. However, to get
	full credit you will need to implement the more effective SIFT descriptor
	(See Szeliski 4.1.2 or the original publications at
	http://www.cs.ubc.ca/~lowe/keypoints/)

	Your implementation does not need to exactly match the SIFT reference.
	Here are the key properties your (baseline) descriptor should have:
	(1) a 4x4 grid of cells, each feature_width/4. It is simply the
		terminology used in the feature literature to describe the spatial
		bins where gradient distributions will be described.
	(2) each cell should have a histogram of the local distribution of
		gradients in 8 orientations. Appending these histograms together will
		give you 4x4 x 8 = 128 dimensions.
	(3) Each feature should be normalized to unit length.

	You do not need to perform the interpolation in which each gradient
	measurement contributes to multiple orientation bins in multiple cells
	As described in Szeliski, a single gradient measurement creates a
	weighted contribution to the 4 nearest cells and the 2 nearest
	orientation bins within each cell, for 8 total contributions. This type
	of interpolation probably will help, though.

	You do not have to explicitly compute the gradient orientation at each
	pixel (although you are free to do so). You can instead filter with
	oriented filters (e.g. a filter that responds to edges with a specific
	orientation). All of your SIFT-like feature can be constructed entirely
	from filtering fairly quickly in this way.

	You do not need to do the normalize -> threshold -> normalize again
	operation as detailed in Szeliski and the SIFT paper. It can help, though.

	Another simple trick which can help is to raise each element of the final
	feature vector to some power that is less than one.

	Args:
	-   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
	-   x: A numpy array of shape (k,), the x-coordinates of interest points
	-   y: A numpy array of shape (k,), the y-coordinates of interest points
	-   feature_width: integer representing the local feature width in pixels.
			You can assume that feature_width will be a multiple of 4 (i.e. every
				cell of your local SIFT-like feature will have an integer width
				and height). This is the initial window size we examine around
				each keypoint.
	-   scales: Python list or tuple if you want to detect and describe features
			at multiple scales

	You may also detect and describe features at particular orientations.

	Returns:
	-   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
			"feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
			These are the computed features.
	"""
	assert image.ndim == 2, 'Image must be grayscale'
	#############################################################################
	# TODO: YOUR CODE HERE                                                      #
	# If you choose to implement rotation invariance, enabling it should not    #
	# decrease your matching accuracy.                                          #
	#############################################################################
	x = np.int64(x)
	y = np.int64(y)

	num = len(x)
	cell_num = int(feature_width/4)
	half_width = int(feature_width/2)
	n = image.shape[0]
	m = image.shape[1]

	gaussian_filter = cv2.getGaussianKernel(feature_width, 1)
	image = cv2.filter2D(image, ddepth=-1, kernel=gaussian_filter)
	Ix = cv2.Sobel(image,cv2.CV_64F,0,1)
	Iy = cv2.Sobel(image,cv2.CV_64F,1,0)
	strength = np.sqrt(np.multiply(Ix, Ix) + np.multiply(Iy, Iy))
	angles = np.ceil(4*np.arctan2(Ix, Iy)/np.pi)+3
	
	fv = np.zeros((num, 8*cell_num*cell_num))

	for i in range(num):
		patch_deg = angles[y[i][0]-half_width: y[i][0]+half_width, x[i][0]-half_width: x[i][0]+half_width]
		patch_str = strength[y[i][0]-half_width: y[i][0]+half_width, x[i][0]-half_width: x[i][0]+half_width]      
		patch_str = cv2.filter2D(patch_str, ddepth=-1, kernel=gaussian_filter)
		# features = np.zeros(8*cell_num*cell_num)
		count = 0
		for j in range(cell_num):
			for k in range(cell_num):
				cell_deg = patch_deg[j*4:(j+1)*4, k*4:(k+1)*4]
				cell_str = patch_str[j*4:(j+1)*4, k*4:(k+1)*4]
				for c in range(8):
					fv[i, count + c] = np.sum(cell_str[np.where(cell_deg==c)])
				count += 8

	fv = fv/fv.max()

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return fv
