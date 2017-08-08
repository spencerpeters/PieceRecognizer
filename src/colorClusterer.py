__author__ = 'Andy, Spencer'

# Color clustering code
# Partitions an image into potential candidate clusters

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

from skimage.measure import regionprops

from skimage.color import rgb2gray

import shutil
import os
from os.path import isfile, join

# compression of image width
ZOOM_FACTOR = 25

# number of clusters to segment the image into
NUM_CLUSTERS = 200

# file locations
ROOT_DIR = "/Users/Spencer/PycharmProjects/PieceRecognizer/"
DERIVED_DIR = ROOT_DIR + "derived/"
ORIGINALS_DIR = ROOT_DIR + "originals/"
BN_DIR = ORIGINALS_DIR + "bnImages1/"
GR_DIR = ORIGINALS_DIR + "grapheneImages1/"
CLUSTERS_DIR = DERIVED_DIR + "clusters/"
LABELS_DIR = DERIVED_DIR + "labeled/"
CACHED_TREE_COMPUTATIONS_DIR = DERIVED_DIR + "cachedTreeComputations/"

def main():
    paths = [join(BN_DIR, f) for f in os.listdir(BN_DIR) if isfile(join(BN_DIR, f))]
    originals = [mpimg.imread(path) for path in paths]

    # Clear the directory and make it again
    shutil.rmtree(CLUSTERS_DIR)
    os.mkdir(CLUSTERS_DIR)

    for i in range(len(originals)):
        original = originals[i]
        clusterer = ColorClusterer(original, ZOOM_FACTOR, NUM_CLUSTERS)
        clusterer.cluster()
        clusteredImage = clusterer.clusteredImage

        mpimg.imsave(LABELS_DIR + "myLabel" + str(i) + ".jpg", clusteredImage)

        clusterer.regionImages()
        currentImageClusterDir = CLUSTERS_DIR + "original" + str(i) + "/"
        os.mkdir(currentImageClusterDir)
        for j in range(len(clusterer.thumbnails)):
            mpimg.imsave(currentImageClusterDir + "region" + str(j) + ".jpg", clusterer.thumbnails[j])

class ColorClusterer:

    def __init__(self, originalImage, zoomFactor, numClusters):
        self.originalImage = originalImage
        assert(originalImage.shape)

        self.initializeCompressedImageAndRelatedData(zoomFactor)

        self.initializeClusteringInputOutput(numClusters)

        self.initializeClusterer()

    # ----- Setter methods -----

    def setZoomFactor(self, newZoomFactor):
        self.initializeCompressedImageAndRelatedData(newZoomFactor)

    def setNumClusters(self, newNumClusters):
        self.numClusters = newNumClusters
        self.agglomerativeClusterer.set_params(n_clusters=newNumClusters)

    # ----- Main methods -----
    # takes in an image array and optional number of clusters desired
    # returns array of size of image whose nonzero values are to which cluster each pixel belongs
    def cluster(self):
        # linkage basically chooses how "distance" between color values computed, ward only setting that worked for me
        self.agglomerativeClusterer.fit(self.features)

        labels = np.reshape(self.agglomerativeClusterer.labels_, self.gray.shape)
        self.clusteredImage = labels

    def regionImages(self):
        regions = regionprops(self.clusteredImage)
        self.thumbnails = []

        for region in regions:
            if 200 > region.area > 10:
                min_row, min_col, max_row, max_col = region.bbox
                # print(str(max_row - min_row) + ", " + str(max_col - min_col))
                # print(np.shape(region.intensity_image))
                thumbnail = self.compressedImage[min_row:max_row, min_col:max_col, :]
                self.thumbnails.append(thumbnail)

    # --------- Internal methods --------

    # Shitty downsizing of image using ZOOM_FACTOR as stride length
    def comp(self, img):
        compress = img[0:len(img):self.zoomFactor,0:len(img[0]):self.zoomFactor]
        return compress

    # converts image to greyscale
    @staticmethod
    def avg(img):
        return rgb2gray(img)

    def initializeClusteringInputOutput(self, numClusters):
        self.numClusters = numClusters
        self.clusteredImage = None
        self.thumbnails = []

    def initializeCompressedImageAndRelatedData(self, zoomFactor):
        self.zoomFactor = zoomFactor
        self.compressedImage = self.comp(self.originalImage)
        self.gray = self.avg(self.compressedImage)
        self.connectivity = grid_to_graph(*self.gray.shape)
        self.features = np.reshape(self.compressedImage, (-1, 3))

    def initializeClusterer(self):
        self.agglomerativeClusterer = self.agglomerativeClusterer = AgglomerativeClustering(n_clusters=self.numClusters,
                                                                                            linkage='ward',
                                                                                            connectivity=self.connectivity,
                                                                                            compute_full_tree='auto')

if __name__ == '__main__':
    main()
