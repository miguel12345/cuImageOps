
from abc import ABC, abstractmethod
from ast import Pass
import numbers
from typing import Any, List, Tuple

from .cuda.stream import CudaStream

from .datacontainer import DataContainer
from .cuda.context import CudaContext
from cuImageOps.utils.cuda import *
import numpy as np
import math

class Operation(ABC):

    _defaultStream = None

    def __init__(self,stream: CudaStream = None) -> None:
        super().__init__()
        self.dataContainers : List[DataContainer] = []
        self.module = None
        self.kernel = None
        self.stream = stream.stream

        if self.stream is None:

            if Operation._defaultStream is None:
                Operation._defaultStream = CudaStream()
            
            self.stream = Operation._defaultStream


    def __del__(self):
        if self.module is not None:
            err, = cuda.cuModuleUnload(self.module)
            check_error(err)

