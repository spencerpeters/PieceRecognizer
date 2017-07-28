__author__ = 'Spencer'
import skimage.io as io

class SingleImageMetadataBase:

    def __init__(self, filename):
        self.filename = filename

    def save(self, data):
        io.imsave(data, self.filename)
