import math
import numpy as np
from cuda import cuda, nvrtc


def gaussian(x: float, sigma: float):
    return (1 / ((math.sqrt(2 * math.pi)) * sigma)) * math.exp(
        -((x**2)) / (2 * (sigma**2))
    )


class ArrayWithExtra(np.ndarray):
    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.uninitialized = getattr(obj, "uninitialized", False)


def is_np_array_uninitialized(arr: np.array):
    return hasattr(arr, "uninitialized") and arr.uninitialized


def create_np_array_uninitialized(shape, dtype):
    empty_arr = np.empty(shape, dtype).view(ArrayWithExtra)
    empty_arr.uninitialized = True
    return empty_arr


def create_np_array_uninitialized_like(other: np.array, dtype=None):
    empty_arr: np.array = np.empty_like(other, dtype=dtype).view(ArrayWithExtra)
    empty_arr.uninitialized = True
    return empty_arr


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
