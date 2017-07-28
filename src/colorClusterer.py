__author__ = 'Andy, Spencer'

# Color clustering code
# Partitions an image into potential candidate clusters

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

from skimage.measure import label, regionprops


import skimage
import skimage.color

# compression of image width
ZOOM_FACTOR = 25

def main():
    comp_img, labels = cluster("/Users/Spencer/PycharmProjects/PieceRecognizer/originals/bnImages1/17_04_19_bn_52.JPG")
    regionImages(labels, comp_img)
    # plt.imshow(labels)
    # plt.show()

# Shitty downsizing of image using ZOOM_FACTOR as stride length
def comp(img):
    compress = img[0:len(img):ZOOM_FACTOR,0:len(img[0]):ZOOM_FACTOR]
    return compress

# converts image to greyscale
def avg(img):
    return skimage.color.rgb2gray(img)

# takes in an image array and optional number of clusters desired
# returns array of size of image whose nonzero values are to which cluster each pixel belongs
def cluster(impath, n_clusters = 200):
    full_img = mpimg.imread(impath)
    comp_img = comp(full_img)

    grey = avg(comp_img)

    # initializes connectivity matrix
    connectivity = grid_to_graph(*grey.shape)

    X = np.reshape(comp_img, (-1, 3))

    # linkage basically chooses how "distance" between color values computed, ward only setting that worked for me
    ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                                   connectivity=connectivity, compute_full_tree = 'auto')

    ward.fit(X)
    labels = np.reshape(ward.labels_, grey.shape)

    mpimg.imsave("/Users/Spencer/PycharmProjects/PieceRecognizer/derived/labeled/myLabeled.jpg", labels)

    return grey, labels

def regionImages(labels, comp_img):
    regions = regionprops(labels, comp_img)

    i = 0
    for region in regions:
        if region.area > 100:
            mpimg.imsave("/Users/Spencer/PycharmProjects/PieceRecognizer/derived/clusters/region" + str(i) + ".jpg", region.intensity_image)
        i += 1

if __name__ == '__main__':
    main()
