from math import sqrt, atan2
import numpy

__author__ = 'Spencer'

import skimage.io
import skimage.color
import skimage

bn = skimage.io.imread("/Users/Spencer/PycharmProjects/PieceRecognizer/originals/bnImages1/17_04_19_bn_54.JPG")

hsv = skimage.color.rgb2hsv(bn)

# def rgb2hsiUnvectorized(r, g, b):
#     M = max(r, g, b)
#     m = min(r, g, b)
#
#     alpha = 0.5*(2*r + g + b)
#     beta = sqrt(3)/2 * (g - b)
#
#     H = atan2(beta, alpha)
#     I = (1/3) * (r + g + b)
#
#     S = 0 if I == 0 else 1 - m/I
#
#     return (H, S, I)
#
# hsi = bn.copy()
#
# X, Y, Color = numpy.shape(hsi)
#
# for x in range(X):
#     for y in range(Y):
#         r = hsi[x, y, 0]
#         g = hsi[x, y, 1]
#         b = hsi[x, y, 2]
#         h, s, i = rgb2hsiUnvectorized(r, g, b)
#         hsi[x, y, 0] = h
#         hsi[x, y, 1] = s
#         hsi[x, y, 2] = i

grey = hsv[:, :, 0]

skimage.io.imsave("/Users/Spencer/PycharmProjects/PieceRecognizer/derived/hsiHsv derived/hue.jpg", grey)




