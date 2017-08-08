import pickle
from sklearn.cluster import AgglomerativeClustering
from colorClusterer import ColorClusterer
from utilities.sampleData import smallGraphene
import time
from scipy.stats import describe

__author__ = 'Spencer'

def main():
    clusterer = ColorClusterer(smallGraphene(), 25, 2)
    time0 = time.time()
    clusterer.cluster()
    agg = clusterer.agglomerativeClusterer
    print(agg.labels_)
    print(describe(agg.labels_))
    time1 = time.time()
    result1 = agg.labels_
    print(describe(agg.labels_))
    print(result1)
    agg.set_params(n_clusters=100)
    agg.fit(clusterer.features)
    time2 = time.time()
    result2 = agg.labels_
    print(result2)
    print(describe(agg.labels_))

    agg.set_params(n_clusters=50)
    agg.fit(clusterer.features)
    time3 = time.time()
    result3 = agg.labels_
    print(result3)
    print(describe(agg.labels_))

    time4 = time.time()
    print(time0)
    print(time1)
    print(time2)
    print(time3)
    print(time4)
    # tree = agg.children_
    # print(tree)
    for i in range(1, 100):
        start = time.time()
        agg.set_params(n_clusters=i)
        agg.fit(clusterer.features)
        end = time.time()
        print(i)
        print(end - start)


def iteration(tree):
   pass

if __name__ == '__main__':
    main()