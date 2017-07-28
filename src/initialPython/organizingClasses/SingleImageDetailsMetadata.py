from src.initialPython.organizingClasses.SingleImageMetadataBase import SingleImageMetadataBase

__author__ = 'Spencer'

class SingleImageDetailsMetadata(SingleImageMetadataBase):

    def __init__(self, filename, extension, numChannels):
        super(SingleImageDetailsMetadata, self).__init__(filename)
        self.extension = extension
        self.numChannels = numChannels



