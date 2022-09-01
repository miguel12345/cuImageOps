from abc import ABC
from typing import List
from cuda import cuda
import cuImageOps.utils.cuda as cuda_utils
from .cuda.stream import CudaStream
from .datacontainer import DataContainer


class Operation(ABC):

    _defaultStream = None

    def __init__(self, stream: CudaStream = None) -> None:
        super().__init__()
        self.data_containers: List[DataContainer] = []
        self.module = None
        self.kernel = None
        self.stream: CudaStream = None

        if stream is None:

            if Operation._defaultStream is None:
                Operation._defaultStream = CudaStream()

            self.stream = Operation._defaultStream
        else:
            self.stream = stream

    def __del__(self):
        print("Destroying operation")
        print(f"self.stream {self.stream}")
        if self.stream is not None and self.module is not None:
            (err,) = cuda.cuModuleUnload(self.module)
            cuda_utils.check_error(err)
