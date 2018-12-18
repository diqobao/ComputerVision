import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_interest_points(image, feature_width):
	"""
	Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
	You can create additional interest point detector functions (e.g. MSER)
	for extra credit.

	If you're finding spurious interest point detections near the boundaries,
	it is safe to simply suppress the gradients / corners near the edges of
	the image.

	Useful in this function in order to (a) suppress boundary interest
	points (where a feature wouldn't fit entirely in the image, anyway)
	or (b) scale the image filters being used. Or you can ignore it.

	By default you do not need to make scale and orientation invariant
	local features.

	The lecture slides and textbook are a bit vague on how to do the
	non-maximum suppression once you've thresholded the cornerness score.
	You are free to experiment. For example, you could compute connected
	components and take the maximum value within each component.
	Alternatively, you could run a max() operator on each sliding window. You
	could use this to ensure that every interest point is at a local maximum
	of cornerness.

	Args:
	-   image: A numpy array of shape (m,n,c),
		image may be grayscale of color (your choice)
	-   feature_width: integer representing the local feature width in pixels.

	Returns:
	-   x: A numpy array of shape (N,) containing x-coordinates of interest points
	-   y: A numpy array of shape (N,) containing y-coordinates of interest points
	-   confidences (optional): numpy nd-array of dim (N,) containing the strength
		of each interest point
	-   scales (optional): A numpy array of shape (N,) containing the scale at each
		interest point
	-   orientations (optional): A numpy array of shape (N,) containing the orientation
		at each interest point
	"""
	confidences, scales, orientations = None, None, None
	#############################################################################
	# TODO: YOUR HARRIS CORNER DETECTOR CODE HERE                               #
	########ß####################################################################
	
	m = image.shape[0]
	n = image.shape[1]
	alpha = 0.05
	half_width = int(feature_width/2)

	# kernel = cv2.getGaussianKernel(9, 7)
	# image = cv2.filter2D(image, ddepth=-1, kernel=kernel)

	# Iy, Ix = np.gradient(image)
	Ix = cv2.Sobel(image,cv2.CV_64F,1,0)
	Iy = cv2.Sobel(image,cv2.CV_64F,0,1)
	Ixx = np.multiply(Ix, Ix)
	Iyy = np.multiply(Iy, Iy)
	Ixy = np.multiply(Iy, Ix)
	kernel = cv2.getGaussianKernel(3,1)
	Ixx = cv2.filter2D(Ixx, ddepth=-1, kernel=kernel)
	Ixy = cv2.filter2D(Ixy, ddepth=-1, kernel=kernel)
	Iyy = cv2.filter2D(Iyy, ddepth=-1, kernel=kernel)

	
	measures = np.multiply(Ixx, Iyy) - np.multiply(Ixy, Ixy) - alpha*np.square(Ixx + Iyy)

	mean = np.mean(measures)
	threshold = np.abs(3*mean);
	y, x = np.where(measures > threshold)
	ids = np.where(x-half_width >=0)
	y = y[ids]
	# radius = radius[ids]
	x = x[ids]
	ids = np.where(x+half_width<=n)
	y = y[ids]
	# radius = radius[ids]
	x = x[ids]
	ids = np.where(y-half_width>=0)
	y = y[ids]
	# radius = radius[ids]
	x = x[ids]
	ids = np.where(y+half_width<=m)
	y = y[ids]
	# radius = radius[ids]
	x = x[ids]
	strength = measures[y, x]

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	
	#############################################################################
	# TODO: YOUR ADAPTIVE NON-MAXIMAL SUPPRESSION CODE HERE                     #
	# While most feature detectors simply look for local maxima in              #
	# the interest function, this can lead to an uneven distribution            #
	# of feature points across the image, e.g., points will be denser           #
	# in regions of higher contrast. To mitigate this problem, Brown,           #
	# Szeliski, and Winder (2005) only detect features that are both            #
	# local maxima and whose response value is significantly (10%)              #
	# greater than that of all of its neighbors within a radius r. The          #
	# goal is to retain only those points that are a maximum in a               #
	# neighborhood of radius r pixels. One way to do so is to sort all          #
	# points by the response strength, from large to small response.            #
	# The first entry in the list is the global maximum, which is not           #
	# suppressed at any radius. Then, we can iterate through the list           #
	# and compute the distance to each interest point ahead of it in            #
	# the list (these are pixels with even greater response strength).          #
	# The minimum of distances to a keypoint's stronger neighbors               #
	# (multiplying these neighbors by >=1.1 to add robustness) is the           #
	# radius within which the current point is a local maximum. We              #
	# call this the suppression radius of this interest point, and we           #
	# save these suppression radii. Finally, we sort the suppression            #
	# radii from large to small, and return the n keypoints                     #
	# associated with the top n suppression radii, in this sorted               #
	# orderself. Feel free to experiment with n, we used n=1500.                #
	#                                                                           #
	# See:                                                                      #
	# https://www.microsoft.com/en-us/research/wp-content/uploads/2005/06/cvpr05.pdf
	# or                                                                        #
	# https://www.cs.ucsb.edu/~holl/pubs/Gauglitz-2011-ICIP.pdf                 #
	#############################################################################

	nn = 2000
	# sorting
	sortedId = np.flipud(strength.argsort())
	x = x[sortedId]
	y = y[sortedId]
	strength = strength[sortedId]

	radius = np.zeros_like(x)
	radius[0] = m*m + n*n + 1
	for i in range(len(x) - 1):
		strongneighbors = np.where(strength > 1.1*strength[i+1])[0]
		# strongneighbors = np.array(range(i+1))
		if(len(strongneighbors)>0):
			radius[i+1] = np.min(np.square(x[strongneighbors] - x[i+1]) + np.square(y[strongneighbors] - y[i+1]))
		else:
			radius[i+1] = m*m + n*n

	si = np.flipud(radius.argsort())
	x = np.reshape(x[si][:nn], (-1,1))
	y = np.reshape(y[si][:nn], (-1,1))

	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return x,y, confidences, scales, orientations

