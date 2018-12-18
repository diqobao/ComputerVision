import numpy as np
import scipy.linalg as linalg
import time


def match_features(features1, features2, x1, y1, x2, y2):
	"""
	This function does not need to be symmetric (e.g. it can produce
	different numbers of matches depending on the order of the arguments).

	To start with, simply implement the "ratio test", equation 4.18 in
	section 4.1.3 of Szeliski. There are a lot of repetitive features in
	these images, and all of their descriptors will look similar. The
	ratio test helps us resolve this issue (also see Figure 11 of David
	Lowe's IJCV paper).

	For extra credit you can implement various forms of spatial/geometric
	verification of matches, e.g. using the x and y locations of the features.

	Args:
	-   features1: A numpy array of shape (n,feat_dim) representing one set of
			features, where feat_dim denotes the feature dimensionality
	-   features2: A numpy array of shape (m,feat_dim) representing a second set
			features (m not necessarily equal to n)
	-   x1: A numpy array of shape (n,) containing the x-locations of features1
	-   y1: A numpy array of shape (n,) containing the y-locations of features1
	-   x2: A numpy array of shape (m,) containing the x-locations of features2
	-   y2: A numpy array of shape (m,) containing the y-locations of features2

	Returns:
	-   matches: A numpy array of shape (k,2), where k is the number of matches.
			The first column is an index in features1, and the second column is
			an index in features2
	-   confidences: A numpy array of shape (k,) with the real valued confidence for
			every match

	'matches' and 'confidences' can be empty e.g. (0x2) and (0x1)
	"""
	#############################################################################
	# TODO: YOUR CODE HERE                                                        #
	#############################################################################
	time1 = time.time()
	features1 -= np.mean(features1, axis=0)
	features1 /= np.std(features1, axis=0)
	features2 -= np.mean(features2, axis=0)
	features2 /= np.std(features2, axis=0)
	features = np.vstack((features1, features2))
	covariance = np.cov(features, rowvar=False)
	w, v = linalg.eigh(covariance)
	index = np.argsort(w)[::-1]
	w = w[index]/np.sum(w)
	w = np.cumsum(w)
	v = v[:, index]
	index = np.where(w >= 0.85)[0][0]
	features1 = np.dot(features1, v[:, :index+1])
	features2 = np.dot(features2, v[:, :index+1])

	matches = np.zeros((len(features1), 2))
	confidences = np.zeros(len(features1))
	count = 0
	threshold = 0.8

	for i in range(len(features1)):
		dists = np.sum(np.square(features2 - features1[i]),axis=1)
		sorting = dists.argsort()
		ratio = np.sqrt(dists[sorting[0]])/np.sqrt(dists[sorting[1]])
		if(ratio < threshold):
			matches[count] = [i, sorting[0]]
			confidences[count] = 1 / max([ratio,0.00000001])
			count += 1
		
	matches = np.int64(matches[0:count])
	confidences = confidences[0:count]
	sortedId = np.flipud(confidences.argsort())
	matches = matches[sortedId]
	confidences = confidences[sortedId]
	time2 = time.time()
	# print(time2-time1)
	#############################################################################
	#                             END OF YOUR CODE                              #
	#############################################################################
	return matches, confidences