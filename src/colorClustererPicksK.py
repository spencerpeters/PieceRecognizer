import math

from mpl_toolkits.mplot3d import Axes3D

from colorClusterer import ColorClusterer
from colorClusterer import CACHED_TREE_COMPUTATIONS_DIR, CLUSTERS_DIR, BN_DIR, ZOOM_FACTOR, NUM_CLUSTERS, LABELS_DIR
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabaz_score

import shutil
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
from os.path import isfile, join

MAX_CLUSTERS = 500
__author__ = 'Spencer'

TEST_POINTS = 30

def main():
    paths = [join(BN_DIR, f) for f in os.listdir(BN_DIR) if isfile(join(BN_DIR, f))][0:1]  #only want 1 element to start with.
    print("length of paths is" + str(len(paths)))
    originals = [mpimg.imread(path) for path in paths]

    # Clear the directory and make it again
    shutil.rmtree(CLUSTERS_DIR)
    os.mkdir(CLUSTERS_DIR)

    for i in range(len(originals)):
        original = originals[i]
        clusterer = ColorClustererPicksK(original, ZOOM_FACTOR, NUM_CLUSTERS, TEST_POINTS)
        clusterer.cluster()
        clusteredImage = clusterer.clusteredImage

        mpimg.imsave(LABELS_DIR + "compressed" + str(i) + ".jpg", clusterer.compressedImage)
        mpimg.imsave(LABELS_DIR + "myLabel" + str(i) + ".jpg", clusteredImage)

        clusterer.regionImages()
        currentImageClusterDir = CLUSTERS_DIR + "original" + str(i) + "/"
        os.mkdir(currentImageClusterDir)
        for j in range(len(clusterer.thumbnails)):
            mpimg.imsave(currentImageClusterDir + "region" + str(j) + ".jpg", clusterer.thumbnails[j])

        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(clusterer.features[:, 0], clusterer.features[:, 1], clusterer.features[:, 2])
        plt.show()

class ColorClustererPicksK(ColorClusterer):
    """This class is a clusterer that picks the optimal number of clusters. It does so by maximizing the silhouette
    score of the clustering.
    """

    def __init__(self, originalImage, zoomFactor, numClusters, numOfTestPoints):
        ColorClusterer.__init__(self, originalImage, zoomFactor, numClusters)
        self.numOfTestPoints = numOfTestPoints
        self.alphas = self.computeAlphaKs(MAX_CLUSTERS)
        print(self.alphas)

    def initializeClusterer(self):
        ColorClusterer.initializeClusterer(self)
        self.agglomerativeClusterer.set_params(memory=CACHED_TREE_COMPUTATIONS_DIR)
        self.agglomerativeClusterer.set_params(compute_full_tree=True)

    def cluster(self):
        # compute the entire tree. Everything gets cached.
        self.setNumClusters(2)
        self.agglomerativeClusterer.fit(self.features)

        f = []
        schedule = self.testSchedule()
        for testpoint in schedule:

            # get S_k-1
            self.setNumClusters(testpoint - 1)
            self.agglomerativeClusterer.fit(self.features)
            lastDistortion = self.summmedDistortion(self.features, self.agglomerativeClusterer.labels_, self.numClusters)

            self.setNumClusters(testpoint)
            self.agglomerativeClusterer.fit(self.features)
            currentDistortion = self.summmedDistortion(self.features, self.agglomerativeClusterer.labels_, self.numClusters)

            if lastDistortion == 0:
                f.append(1)
            else:
                fOfK = currentDistortion/(self.alphas[self.numClusters]*lastDistortion)
                f.append(fOfK)

            # print("calculating silhouette")
            # score = calinski_harabaz_score(self.features, self.agglomerativeClusterer.labels_)

        print(f)
        bestScoreIndex = np.argmin(f)
        print(bestScoreIndex)
        bestTestpoint = schedule[bestScoreIndex]
        self.setNumClusters(bestTestpoint)
        self.agglomerativeClusterer.fit(self.features)

        labels = np.reshape(self.agglomerativeClusterer.labels_, self.gray.shape)

        self.clusteredImage = labels
        # plt.imshow(labels)
        # plt.show()


    def testSchedule(self):
        schedule = np.geomspace(MAX_CLUSTERS, 2, self.numOfTestPoints).tolist()
        schedule = [math.floor(testpoint) for testpoint in schedule]
        print(schedule)
        return schedule

    def summmedDistortion(self, features, labels, numLabels):
        featureBuckets = [[] for i in range(self.numClusters)]
        for i in range(len(labels)):
            featureBuckets[labels[i]].append(features[i])

        featureBucketArrays = [np.array(featureBucket) for featureBucket in featureBuckets]

        sum = 0
        for bucket in featureBucketArrays:
            sum += self.distortion(bucket)

        return sum

    def distortion(self, bucket):
        centroid = np.mean(bucket)
        return np.linalg.norm(bucket - centroid)

    # to get alpha-j from the result, get alphas[j - 2]
    def computeAlphaKs(self, maxK):
        alphas = []
        # alpha_1 and alpha_0 don't make sense
        alphas.append(None)
        alphas.append(None)
        alpha2 = float(3)/4  # in general it is 1 - 3 / (4 * N_d) where N_d is the number of dimensions of the data
        alphas.append(alpha2)
        # look at https://datasciencelab.wordpress.com/2014/01/21/selection-of-k-in-k-means-clustering-reloaded/
        for i in range(maxK - 2):
            alphas.append(float(alphas[-1]) + float(1 - alphas[-1])/6)

        return alphas


if __name__ == '__main__':
    main()



