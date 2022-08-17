
from enum import IntEnum

from imageOps.core.datacontainer import DataContainer
from .operation import Operation
from imageOps.utils.cuda import *
import numpy as np
import math

class FillMode(IntEnum):
    CONSTANT = 1,
    REFLECTION = 2

class ImageOperation(Operation):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input = None
        self.output = None

    def _Operation__get_blocks_threads(self):

        numThreads = 16
        #Assume the first kernel argument is the image
        input = self._get_kernel_arguments()[0]
        imgHeight = input.shape[0]
        imgWidth = input.shape[1]

        numBlocksX = (math.ceil((imgWidth)/numThreads))
        numBlocksY = (math.ceil((imgHeight)/numThreads))

        return (numBlocksX,numBlocksY,1),(numThreads,numThreads,1)

    def run(self,input:np.array) -> DataContainer:
        
        self.input = input
        self.output = np.zeros_like(self.input,dtype=np.float32)
        self.dims = np.array((self.input.shape[0],self.input.shape[1]),dtype=np.uint32)

        return super()._Operation__run()

