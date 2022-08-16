
from abc import ABC, abstractmethod
from ast import Pass
from enum import Enum, IntEnum
import numbers
from typing import Any, List, Tuple
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

    def _Operation__get_blocks_threads(self):

        numThreads = 16
        #Assume the first kernel argument is the image
        input = self._get_kernel_arguments()[0]
        imgHeight = input.shape[0]
        imgWidth = input.shape[1]

        numBlocksX = (math.ceil((imgWidth)/numThreads))
        numBlocksY = (math.ceil((imgHeight)/numThreads))

        return (numBlocksX,numBlocksY,1),(numThreads,numThreads,1)

