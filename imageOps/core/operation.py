
from abc import ABC, abstractmethod
from ast import Pass
import numbers
from typing import Any, List, Tuple

from .cuda.stream import CudaStream

from .datacontainer import DataContainer
from .cuda.context import CudaContext
from imageOps.utils.cuda import *
import numpy as np
import math

class Operation(ABC):

    _defaultStream = None

    def __init__(self,stream: CudaStream = None) -> None:
        super().__init__()
        self.dataContainers : List[DataContainer] = []
        self.module = None
        self.stream = stream.stream

        if self.stream is None:

            if Operation._defaultStream is None:
                Operation._defaultStream = CudaStream()
            
            self.stream = Operation._defaultStream


    def __compile_kernel(self,debug=True) -> Any:

        kernel_name_b = str.encode(self.__get_kernel_name())

        # Create program
        err, prog = nvrtc.nvrtcCreateProgram(str.encode(self.__get_kernel()), kernel_name_b, 0, [], [])

        check_error(err)

        # Compile program
        opts = [b"--gpu-architecture=compute_61",b"--include-path=imageOps/core/cuda"]

        if debug:
            opts.extend([b"--device-debug",b"--generate-line-info"])

        err, = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)

        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
            compileLog = b" " * logSize
            nvrtc.nvrtcGetProgramLog(prog, compileLog)
            raise RuntimeError("Nvrtc Compile error: {}".format(compileLog))

        check_error(err)

        # Get PTX from compilation 
        err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
        
        check_error(err)

        ptx = b" " * ptxSize
        err, = nvrtc.nvrtcGetPTX(prog, ptx)

        check_error(err)

        # Load PTX as module data and retrieve function
        ptx = np.char.array(ptx)
        # Note: Incompatible --gpu-architecture would be detected here
        err, self.module = cuda.cuModuleLoadData(ptx.ctypes.data)
        check_error(err)
        err, kernel = cuda.cuModuleGetFunction(self.module, kernel_name_b)
        check_error(err)

        return kernel

    def __copy_arg_data(self,stream):

        kernelArgs = self._get_kernel_arguments()

        #Allocate buffers and copy data
        for hostBuffer in kernelArgs:
            
            dc = DataContainer(hostBuffer,stream)

            #Keep it on CPU if scalar
            if len(hostBuffer.shape) > 0:
                dc.gpu()

            self.dataContainers.append(dc)

    def __get_blocks_threads(self):

        numThreads = 16
        #Assume the first kernel argument is the input
        input = self.__get_kernel_arguments()[0]
        inputLen = input.size
        numBlocks = (math.ceil((inputLen)/numThreads))

        return (numBlocks,1,1),(numThreads,1,1)

    def __run_kernel(self,kernel,stream) -> DataContainer:

        args = np.array([arg.memAddr() for arg in self.dataContainers], dtype=np.uint64)

        # args = np.array([int(arg) for arg in self.kernelArgumentBuffers], dtype=np.uint64)

        blocks, threads = self.__get_blocks_threads()

        err, = cuda.cuLaunchKernel(
        kernel,
        *blocks,
        *threads,
        0,
        stream,  # stream
        args.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
        )
    
        check_error(err)

        outIdx = self.__get_kernel_out_idx()

        return self.dataContainers[outIdx]

    @abstractmethod
    def __get_kernel(self) -> str:
        pass

    @abstractmethod
    def __get_kernel_name(self) -> str:
        pass
    
    @abstractmethod
    def _get_kernel_arguments(self)-> List[np.array]:
        pass

    @abstractmethod
    def __get_kernel_out_idx(self) -> int:
        pass

    def __run(self) -> DataContainer:
    
        kernel = self.__compile_kernel()
        
        self.__copy_arg_data(self.stream)

        outputDataContainer = self.__run_kernel(kernel,self.stream)

        return outputDataContainer

    def __del__(self):
        if self.module is not None:
            err, = cuda.cuModuleUnload(self.module)
            check_error(err)

