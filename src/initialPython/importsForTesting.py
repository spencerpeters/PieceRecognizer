__author__ = 'Spencer'

import numpy as np
import matplotlib.pyplot as plt

import sampleData

from spencerUtilities import *
from skimage.feature import canny
from skimage.morphology import dilation, disk
from scipy import ndimage as ndi
from skimage.measure import label, regionprops
from skimage.filters import sobel
from skimage.morphology import watershed, h_maxima, h_minima
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters.rank import gradient
from scipy.signal import argrelmin, find_peaks_cwt
import matplotlib.patches as mpatches

bn = sampleData.bn()
sbn = sampleData.smallBn()
gbn = rgb2gray(bn)
gsbn = rgb2gray(sbn)

gr = sampleData.graphene()
sgr = sampleData.smallGraphene()
ggr = rgb2gray(gr)
gsgr = rgb2gray(sgr)



