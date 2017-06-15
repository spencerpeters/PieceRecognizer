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

n = 3
markersList = []
segmentationList = []
for i in range(n):
    for j in range(n):
        low = 0.05 + 0.05 * i
        high = 0.5 + 0.1 * j
        markers = np.zeros_like(greybn)
        markers[greybn < low] = 1
        markers[greybn > high] = 2
        markersList.append(markers)
        segmentation = morphology.watershed(elevationMap, markers)
        segmentationList.append(segmentation)

fig, axes = plt.subplots(1, 2 * n*n, figsize=(9, 18), sharey=True)

#axes[1].imshow(elevationMap, cmap=plt.cm.gray, interpolation='nearest')

for markerAxis in range(0, 2*n*n, 2):
    axes[markerAxis].imshow(segmentationList[markerAxis], cmap=plt.cm.grey, interpolation='nearest')
    axes[markerAxis + 1].imshow(markersList[markerAxis], cmap=plt.cm.spectral, interpolation='nearest')

plt.axis('off')
plt.show()
