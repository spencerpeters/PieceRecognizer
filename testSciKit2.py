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

dataDirectory = "/Users/Spencer/PycharmProjects/PieceRecognizer/data"

filename = os.path.join(dataDirectory, 'bnImages1/17_04_19_bn_52.JPG')

bn = io.imread(filename)
# print(np.shape(bn))

greybn = rgb2gray(bn)
# print(greybn[0])

elevationMap = sobel(greybn)

markers = np.zeros_like(greybn)
markers[greybn < 0.1] = 1
markers[greybn > 0.6] = 2

segmentation = morphology.watershed(elevationMap, markers)

fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)

axes[0].imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
axes[1].imshow(elevationMap, cmap=plt.cm.gray, interpolation='nearest')
axes[2].imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')

plt.axis('off')
plt.show()


