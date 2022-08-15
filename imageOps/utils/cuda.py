from cuda import cuda, nvrtc

def check_error(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            message = cuda.cuGetErrorString(err)
            raise RuntimeError("Cuda Error: {}".format(message))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError("Nvrtc Error: {}".format(message))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))