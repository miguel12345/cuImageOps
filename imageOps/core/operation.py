
from abc import ABC, abstractmethod
from ast import Pass
import numbers
from tkinter import E
from typing import Any, List, Tuple
from imageOps.utils.cuda import *
import numpy as np
import math

class Operation(ABC):

    def __init__(self) -> None:
        super().__init__()
        self.kernelArgumentBuffers = []


    def __compile_kernel(self) -> Any:

        kernel_name_b = str.encode(self.__get_kernel_name())

        # Create program
        #err, prog = nvrtc.nvrtcCreateProgram(str.encode(self.__get_kernel()), kernel_name_b, 1, [b"D:\Projects\Pessoal\imageOps\imageOps"], [b"utils.cu"])
        err, prog = nvrtc.nvrtcCreateProgram(str.encode(self.__get_kernel()), kernel_name_b, 0, [], [])

        check_error(err)

        # Compile program
        opts = [b"--fmad=false", b"--gpu-architecture=compute_61",b"--device-debug",b"--generate-line-info",b"--include-path=imageOps\core\cuda"]
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
        err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
        check_error(err)
        err, kernel = cuda.cuModuleGetFunction(module, kernel_name_b)
        check_error(err)

        return kernel

    def __init_context(self):
        
        # Initialize CUDA Driver API
        err, = cuda.cuInit(0)
        check_error(err)

        # Retrieve handle for device 0
        err, cuDevice = cuda.cuDeviceGet(0)
        check_error(err)

        # Create context
        err, context = cuda.cuCtxCreate(0, cuDevice)
        check_error(err)

        return context

    def __copy_arg_data(self,stream):

        kernelArgs = self._get_kernel_arguments()

        #Allocate buffers and copy data
        for kernelArg in kernelArgs:
            kernelArgLen = 1

            #If scalar, use the value directly
            if len(kernelArg.shape) > 0:
                kernelArgLen = kernelArg.size
                bufferSize = kernelArgLen * kernelArg.itemsize
                err, kernelArgBuffer = cuda.cuMemAlloc(bufferSize)
                check_error(err)
                self.kernelArgumentBuffers.append(kernelArgBuffer)
                err, = cuda.cuMemcpyHtoDAsync(
                    kernelArgBuffer, kernelArg.ctypes.data, bufferSize, stream
                    )
                check_error(err)
            else:
                self.kernelArgumentBuffers.append(kernelArg)

        err, = cuda.cuStreamSynchronize(stream)

    def __get_blocks_threads(self):

        numThreads = 16
        #Assume the first kernel argument is the input
        input = self.__get_kernel_arguments()[0]
        inputLen = input.size
        numBlocks = (math.ceil((inputLen)/numThreads))

        return (numBlocks,1,1),(numThreads,1,1)

    def __run_kernel(self,kernel,stream):

        kernelArgBufferPointers = [np.array([int(kernelArgumentBuffer)], dtype=np.uint64) if isinstance(kernelArgumentBuffer,cuda.CUdeviceptr) else kernelArgumentBuffer
                                    for kernelArgumentBuffer in self.kernelArgumentBuffers]
    
        args = np.array([arg.ctypes.data for arg in kernelArgBufferPointers], dtype=np.uint64)

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

        #Copy output

        outIdx = self.__get_kernel_out_idx()
        outKernelArg = self._get_kernel_arguments()[outIdx]
        outBufferSize = outKernelArg.size * outKernelArg.itemsize

        err, = cuda.cuMemcpyDtoHAsync(
        outKernelArg.ctypes.data, self.kernelArgumentBuffers[outIdx], outBufferSize, stream
        )
        check_error(err)
        err, = cuda.cuStreamSynchronize(stream)
        check_error(err)

        return outKernelArg

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

    def run(self):
        
        self.__init_context()
        kernel = self.__compile_kernel()
        
        err, stream = cuda.cuStreamCreate(0)
        check_error(err)

        self.__copy_arg_data(stream)

        output = self.__run_kernel(kernel,stream)

        return output

