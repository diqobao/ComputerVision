import numpy as np

def my_imfilter(image, filter):
  """
  Apply a filter to an image. Return the filtered image.

  Args
  - image: numpy nd-array of dim (m, n, c)
  - filter: numpy nd-array of dim (k, k)
  Returns
  - filtered_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You may not use any libraries that do the work for you. Using numpy to work
   with matrices is fine and encouraged. Using opencv or similar to do the
   filtering for you is not allowed.
  - I encourage you to try implementing this naively first, just be aware that
   it may take an absurdly long time to run. You will need to get a function
   that takes a reasonable amount of time to run so that the TAs can verify
   your code works.
  - Remember these are RGB images, accounting for the final image dimension.
  """

  assert filter.shape[0] % 2 == 1
  assert filter.shape[1] % 2 == 1

  ############################
  ### TODO: YOUR CODE HERE ###

  dim = [image.shape[0], image.shape[1], image.shape[2]]
  paddings = [int(filter.shape[0]/2), int(filter.shape[1]/2)]
  filtered_image = np.zeros_like(image)
  
  # Padding
  padded = [image[:,:,0],image[:,:,0],image[:,:,0]]

  for i in range(dim[2]):    
    # padded[i] = np.pad(image[:,:,i], pad_width = ((paddings[0],paddings[0]),(paddings[1],paddings[1])), mode= 'constant', constant_values=0)
    padded[i] = np.pad(image[:,:,i], pad_width = ((paddings[1],paddings[1]),(paddings[0],paddings[0])), mode= 'reflect')
 
  # convolution
  for i in range(dim[2]):
    for j in range(dim[0]):
      for k in range(dim[1]):
        filtered_image[j, k, i] = (filter * padded[i][j:j + filter.shape[0],k:k+filter.shape[1]]).sum()

  # raise NotImplementedError('`my_imfilter` function in `student_code.py` ' +
  #   'needs to be implemented')

  ### END OF STUDENT CODE ####
  ############################

  return filtered_image

def create_hybrid_image(image1, image2, filter):
  """
  Takes two images and creates a hybrid image. Returns the low
  frequency content of image1, the high frequency content of
  image 2, and the hybrid image.

  Args
  - image1: numpy nd-array of dim (m, n, c)
  - image2: numpy nd-array of dim (m, n, c)
  Returns
  - low_frequencies: numpy nd-array of dim (m, n, c)
  - high_frequencies: numpy nd-array of dim (m, n, c)
  - hybrid_image: numpy nd-array of dim (m, n, c)

  HINTS:
  - You will use your my_imfilter function in this function.
  - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
  - Don't forget to make sure the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
  - If you want to use images with different dimensions, you should resize them
    in the notebook code.
  """

  assert image1.shape[0] == image2.shape[0]
  assert image1.shape[1] == image2.shape[1]
  assert image1.shape[2] == image2.shape[2]

  ############################
  ### TODO: YOUR CODE HERE ###

  # using convolution
  low_frequencies = my_imfilter(image1, filter)
  high_frequencies = image2 - my_imfilter(image2, filter)

  # # using fft
  # cut_freq = 17
  # sigma = cut_freq * cut_freq
  # i0 = int(image2.shape[0] / 2) + image2.shape[0]%2
  # j0 = int(image2.shape[1] / 2) + image2.shape[1]%2
  # filter = np.zeros((image2.shape[0], image2.shape[1]))
  # for i in range(image2.shape[0]):
  #   for j in range(image2.shape[1]):
  #     filter[i, j] = np.exp(((i-i0)**2 + (j-j0)**2)/(-2*sigma))

  # low_frequencies = np.zeros_like(image1)
  # high_frequencies = np.zeros_like(image2)
  # for c in range(image2.shape[2]):
  #   low_frequencies[:,:,c] = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(image1[:,:,c])) * filter))
  #   high_frequencies[:,:,c] = np.fft.ifft2(np.fft.ifftshift(np.fft.fftshift(np.fft.fft2(image2[:,:,c])) * filter))
  # high_frequencies = image2 - high_frequencies


  hybrid_image = low_frequencies + high_frequencies
  hybrid_image = np.clip(hybrid_image, 0, 1)

  ### END OF STUDENT CODE ####
  ############################

  return low_frequencies, high_frequencies, hybrid_image
  