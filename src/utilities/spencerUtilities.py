__author__ = 'Spencer'

from skimage.exposure import histogram

from utilities.importsForTesting import *


def plotHistogram(image, channel=None, log=False):
    if channel is None:
        channelImage = image
    else:
        channelImage = image[:, :, channel]
    histo, _ = histogram(channelImage, nbins=256)

    if log:
        histo = np.log(1 + histo)
    plt.plot(histo)

def normalizedIntegralHistogram(floatImage):
    histogram, _ = np.histogram(floatImage, range=(0, 1), bins=255)
    integral = np.cumsum(histogram)
    totalPixels = integral[-1]
    normalized = integral/totalPixels
    return normalized

def makeMarkersFromThresholds(floatImage, lowerWeight, upperWeight):
    normalized = normalizedIntegralHistogram(floatImage)
    upper, lower = findUpperAndLowerBounds(normalized, lowerWeight, upperWeight)
    return makeMarkers(floatImage, lower/255, upper/255)


def makeMarkers(image, lower, upper):
    markers = np.zeros_like(image)

    markers[image < lower] = 1
    markers[image > upper] = 2
    return markers

def doWatershed(floatImage, lowerWeight, upperWeight, markerMethod, markerMethodParams):
    markers = markerMethod(*markerMethodParams)
    elevationMap = sobel(floatImage)
    segmentation = watershed(elevationMap, markers)
    labelImage = label(segmentation)
    return labelImage, segmentation, elevationMap, markers, normalized

def findUpperAndLowerBounds(integralNormalizedHistogram, lowerWeight, upperWeight):
    upper = 0
    lower = 0
    for i in range(len(integralNormalizedHistogram)):
        if integralNormalizedHistogram[i] > lowerWeight:
            lower = i
            break

    for j in range(1, len(integralNormalizedHistogram)+1):
        if integralNormalizedHistogram[-j] < upperWeight:
            upper = 256 - j
            break

    return (lower, upper)