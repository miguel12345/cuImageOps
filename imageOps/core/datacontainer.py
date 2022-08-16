from typing import Any, Tuple
from cuda import cuda
import numpy as np

from imageOps.utils.cuda import check_error

class DataContainer():
    def __init__(self, hostBuffer: np.array, stream: any) -> None:
        self.deviceBuffer = None
        self.hostBuffer = hostBuffer
        self.shape = self.hostBuffer.shape
        self.stream = stream
        self.dtype = self.hostBuffer.dtype
        self.deviceBufferPointer = None

    def gpu(self):

        assert self.hostBuffer is not None

        bufferSize = self.hostBuffer.size * self.hostBuffer.itemsize
        err, self.deviceBuffer = cuda.cuMemAlloc(bufferSize)
        check_error(err)
        
        err, = cuda.cuMemcpyHtoDAsync(
            self.deviceBuffer, self.hostBuffer.ctypes.data, bufferSize, self.stream
            )
            
        check_error(err)

        self.deviceBufferPointer = np.array([int(self.deviceBuffer)],dtype=np.uint64)

    def memAddr(self) -> int:
        if self.deviceBuffer is None:
            return self.hostBuffer.ctypes.data
        else:
            return self.deviceBufferPointer.ctypes.data

    def cpu(self):

        self.hostBuffer[:] = 0.0

        #Copy data from device to host
        err, = cuda.cuMemcpyDtoHAsync(
        self.hostBuffer.ctypes.data, self.deviceBuffer, self.hostBuffer.size * self.hostBuffer.itemsize, self.stream
        )
        check_error(err)

        
        #Syncronize stream
        err, = cuda.cuStreamSynchronize(self.stream)
        check_error(err)

        return self

    def numpy(self) -> np.array:
        return self.hostBuffer

    def __del__(self):
        if self.deviceBuffer is not None:
            err, = cuda.cuMemFree(self.deviceBuffer)
            check_error(err)



    