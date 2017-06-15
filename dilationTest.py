__author__ = 'Spencer'

import numpy as np
import skimage
from skimage import io
from skimage.color import rgb2gray
import skimage.data
import os
import matplotlib.pyplot as plt


from skimage.feature import canny
from scipy import ndimage

from skimage import morphology

from skimage.filters import sobel

from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage.morphology import reconstruction

dataDirectory = "/Users/Spencer/PycharmProjects/PieceRecognizer/data"

filename = os.path.join(dataDirectory, 'bnImages1/17_04_19_bn_52.JPG')

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from skimage import data
from skimage import img_as_float
from skimage import img_as_int
from skimage.morphology import reconstruction
from skimage import morphology
bn = io.imread(filename)
# print(np.shape(bn))

greybn = rgb2gray(bn)
# print(greybn[0])

# Convert to float: Important for subtraction later which won't work with uint8
image = greybn
image = gaussian_filter(image, 1)

seed = np.copy(image)
seed[1:-1, 1:-1] = image.min()
mask = image

dilated = reconstruction(seed, mask, method='dilation')
# Subtracting the dilated image leaves an image with just the coins and a flat, black background, as shown below.

final = image - dilated
final = img_as_int(final)

cleanFinal = morphology.remove_small_objects(final, 21)

plt.imshow(cleanFinal, cmap='gray')
plt.axis('off')

plt.show()

