__author__ = 'Spencer'

import skimage.io as io
from skimage.color import rgb2gray


def bn():
    """20x color BN image containing target piece, full size"""
    return io.imread("/Users/Spencer/PycharmProjects/PieceRecognizer/data/bnImages1/17_04_19_bn_54.JPG")

def graphene():
    """20x color Graphene image containing target piece, full size"""
    return io.imread("/Users/Spencer/PycharmProjects/PieceRecognizer/data/grapheneImages1/17_04_11_graphene_chip1_13.JPG")

def smallBn():
     """20x color BN image containing target piece, small size"""
     return bn()[1250:2500, 1300:3900]

def smallGraphene():
     """20x color Graphene image containing target piece, small size"""
     return graphene()[1000:2500, 2200:3600]
