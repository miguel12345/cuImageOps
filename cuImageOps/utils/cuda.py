from typing import List, Tuple
from cuda import cuda, nvrtc
import numpy as np
import math
from cuImageOps.core.datacontainer import DataContainer

def check_error(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            message = cuda.cuGetErrorString(err)
            raise RuntimeError(f"Cuda Error: {message}")
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"Nvrtc Error: {message}")
    else:
        raise RuntimeError(f"Unknown error type: {err}")

def compile_module(module_path, debug=True):
    err, prog = nvrtc.nvrtcCreateProgram(str.encode(open(module_path).read()), str.encode(module_path), 0, [], [])

    check_error(err)
    opts = [b"--gpu-architecture=compute_61", b"--include-path=cuImageOps/core/cuda"]

    if debug:
        opts.extend([b"--device-debug", b"--generate-line-info"])
    (err, ) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        err, logSize = nvrtc.nvrtcGetProgramLogSize(prog)
        compileLog = b" " * logSize
        nvrtc.nvrtcGetProgramLog(prog, compileLog)
        raise RuntimeError(f"Nvrtc Compile error: {compileLog.decode()}")
    check_error(err)
    err, ptxSize = nvrtc.nvrtcGetPTXSize(prog)
    check_error(err)
    ptx = b" " * ptxSize
    (err, ) = nvrtc.nvrtcGetPTX(prog, ptx)
    check_error(err)
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    check_error(err)
    return module

def get_kernel(module,kernelName):
    err, kernel = cuda.cuModuleGetFunction(module, str.encode(kernelName))
    check_error(err)

    return kernel

def run_kernel(kernel:cuda.CUfunction, blocks:Tuple, threads:Tuple, dataContainers:List[DataContainer], stream:cuda.CUstream):

    args = np.array([arg.memAddr() for arg in dataContainers], dtype=np.uint64)

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

def get_kernel_launch_dims(inputImage: np.array, threadsPerBlock: int = 16):

    numThreads = threadsPerBlock
    #Assume the first kernel argument is the image
    imgHeight = inputImage.shape[0]
    imgWidth = inputImage.shape[1]

    numBlocksX = (math.ceil((imgWidth)/numThreads))
    numBlocksY = (math.ceil((imgHeight)/numThreads))

    return (numBlocksX,numBlocksY,1),(numThreads,numThreads,1)

def copy_data_to_device(data:List[np.array], stream: cuda.CUstream) -> List[DataContainer]:

    dataContainers = []

    #Allocate buffers and copy data
    for hostBuffer in data:
        
        dc = DataContainer(hostBuffer,stream)

        #Keep it on CPU if scalar
        if len(hostBuffer.shape) > 0:
            dc.gpu()

        dataContainers.append(dc)

    return dataContainers