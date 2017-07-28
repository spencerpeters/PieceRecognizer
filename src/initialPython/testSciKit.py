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

dataDirectory = "/Users/Spencer/PycharmProjects/PieceRecognizer/originals"

filename = os.path.join(dataDirectory, 'bnImages1/17_04_19_bn_52.JPG')

bn = io.imread(filename)
# print(np.shape(bn))

greybn = rgb2gray(bn)
# print(greybn[0])
#
# coins = skimage.originals.coins()
# print(coins[0])

edges = canny(greybn)

fillBn = ndimage.binary_fill_holes(edges)

bnCleaned = morphology.remove_small_objects(fillBn, 64)

# histo = np.histogram(greybn, bins=np.arange(0, 256))

# print(np.shape(greybn))
# print(np.shape(edges))
print(np.shape(bnCleaned))

fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)

axes[0].imshow(bnCleaned, cmap=plt.cm.gray, interpolation='nearest')
axes[1].imshow(greybn, cmap=plt.cm.gray, interpolation='nearest')

plt.axis('off')
plt.show()


