from typing import Any, Tuple
from cuda import cuda
import numpy as np
from cuImageOps.core.cuda.stream import CudaStream

from cuImageOps.utils.utils import check_error, is_np_array_uninitialized


class DataContainer:
    def __init__(self, hostBuffer: np.array, stream: any) -> None:
        self.device_buffer = None
        self.host_buffer = hostBuffer
        self.shape = self.host_buffer.shape
        self.stream: CudaStream = stream
        self.dtype = self.host_buffer.dtype
        self.device_buffer_pointer = None

    def gpu(self):

        assert self.host_buffer is not None

        buffer_size = self.host_buffer.size * self.host_buffer.itemsize
        err, self.device_buffer = cuda.cuMemAlloc(buffer_size)
        check_error(err)

        # If the host buffer is uninitalized (i.e. output buffers), dont waste time copying data to the device
        skip_copy = False
        if is_np_array_uninitialized(self.host_buffer):
            skip_copy = True

        if not skip_copy:
            (err,) = cuda.cuMemcpyHtoDAsync(
                self.device_buffer,
                self.host_buffer.ctypes.data,
                buffer_size,
                self.stream.native_ptr(),
            )

            check_error(err)

        self.device_buffer_pointer = np.array(
            [int(self.device_buffer)], dtype=np.uint64
        )

    def memAddr(self) -> int:
        if self.device_buffer is None:
            return self.host_buffer.ctypes.data
        else:
            return self.device_buffer_pointer.ctypes.data

    def cpu(self):

        # Copy data from device to host
        (err,) = cuda.cuMemcpyDtoHAsync(
            self.host_buffer.ctypes.data,
            self.device_buffer,
            self.host_buffer.size * self.host_buffer.itemsize,
            self.stream.native_ptr(),
        )
        check_error(err)

        # Syncronize stream
        (err,) = cuda.cuStreamSynchronize(self.stream.native_ptr())
        check_error(err)

        return self

    def numpy(self) -> np.array:
        return self.host_buffer

    def __del__(self):

        if self.stream is not None and self.device_buffer is not None:
            (err,) = cuda.cuMemFree(self.device_buffer)
            check_error(err)
