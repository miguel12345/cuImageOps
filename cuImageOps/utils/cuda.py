import os
from typing import List, Tuple
import math
from cuda import cuda, nvrtc
import numpy as np
import cuImageOps
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.datacontainer import DataContainer
from cuImageOps.utils.utils import check_error


def compile_module(module_path, debug=True):
    err, prog = nvrtc.nvrtcCreateProgram(
        str.encode(open(module_path, encoding="utf-8").read()),
        str.encode(module_path),
        0,
        [],
        [],
    )

    check_error(err)
    cuda_include_path = os.path.join(os.path.dirname(cuImageOps.__file__), "utils")
    opts = [
        b"--gpu-architecture=compute_61",
        str.encode(f"--include-path={cuda_include_path}"),
    ]

    if debug:
        opts.extend([b"--device-debug", b"--generate-line-info"])
    (compile_err,) = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)
    # Check for warnings in the compilation log, and fail if any

    err, log_size = nvrtc.nvrtcGetProgramLogSize(prog)
    compile_log = b" " * log_size
    nvrtc.nvrtcGetProgramLog(prog, compile_log)
    compile_log = compile_log.decode()

    if compile_err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"Nvrtc Compile error: {compile_log}")
    if "warning" in compile_log:
        raise RuntimeError(
            f"Nvrtc Warning when compiling, treating as error: {compile_log}"
        )

    check_error(err)
    err, ptx_size = nvrtc.nvrtcGetPTXSize(prog)
    check_error(err)
    ptx = b" " * ptx_size
    (err,) = nvrtc.nvrtcGetPTX(prog, ptx)
    check_error(err)
    ptx = np.char.array(ptx)
    err, module = cuda.cuModuleLoadData(ptx.ctypes.data)
    check_error(err)
    return module


def get_kernel(module, kernel_name):
    err, kernel = cuda.cuModuleGetFunction(module, str.encode(kernel_name))
    check_error(err)

    return kernel


def run_kernel(
    kernel: cuda.CUfunction,
    blocks: Tuple,
    threads: Tuple,
    data_containers: List[DataContainer],
    stream: CudaStream,
):

    args = np.array([arg.memAddr() for arg in data_containers], dtype=np.uint64)

    (err,) = cuda.cuLaunchKernel(
        kernel,
        *blocks,
        *threads,
        0,
        stream.native_ptr(),  # stream
        args.ctypes.data,  # kernel arguments
        0,  # extra (ignore)
    )

    check_error(err)


def get_kernel_launch_dims(input_image: np.array, threads_per_block: int = 16):

    num_threads = threads_per_block
    # Assume the first kernel argument is the image
    img_height = input_image.shape[0]
    img_width = input_image.shape[1]

    num_blocks_x = math.ceil((img_width) / num_threads)
    num_blocks_y = math.ceil((img_height) / num_threads)

    return (num_blocks_x, num_blocks_y, 1), (num_threads, num_threads, 1)


def copy_data_to_device(
    data: List[np.array], stream: cuda.CUstream
) -> List[DataContainer]:

    data_containers = []

    # Allocate buffers and copy data
    for host_buffer in data:

        data_container = DataContainer(host_buffer, stream)

        # Keep it on CPU if scalar
        if len(host_buffer.shape) > 0:
            data_container.gpu()

        data_containers.append(data_container)

    return data_containers


def allocate_data_to_device(
    data: List[np.array], stream: cuda.CUstream
) -> List[DataContainer]:

    data_containers = []

    # Allocate buffers and copy data
    for host_buffer in data:

        data_container = DataContainer(host_buffer, stream)

        # Keep it on CPU if scalar
        if len(host_buffer.shape) > 0:
            data_container.gpu()

        data_containers.append(data_container)

    return data_containers
