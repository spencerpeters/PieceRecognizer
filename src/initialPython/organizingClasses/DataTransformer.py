class DataTransformerBase:

    # main method of the class, called to perform the originals transformation and get back a metadata object
    # example calls:    output, meta = transformerInstance.transformInput()
    #                   _, meta = transformerInstance.transformInput()
    #                   transformedData, _ = transformerInstance.transformInput()
    def transformInput(self, inputMetaData):
        processedInputList = self.readAndValidateInput(inputFilePath)
        outputData, outputMetadata = self.transform(processedInput)
        outputMetadata.save(outputData)
        return outputData, outputMetadata

    # validates the input file extension, reads the input file, and validates the read object
    def readAndValidateInput(self, inputFilePath):
        raise NotImplementedError("base class")

    # transforms the read input file into an output file and a metadata object that specifies the output file's
    # characteristics and how it will be saved.
    def transform(self, processedInput):
        raise NotImplementedError("base class")