from cuda import cuda

from imageOps.utils.cuda import check_error

class CudaContext():

    def __init__(self, deviceIdx: int = 0) -> None:
        
        self.context = None

        # Initialize CUDA Driver API
        err, = cuda.cuInit(0)
        check_error(err)

        # Retrieve handle for device 0
        err, cuDevice = cuda.cuDeviceGet(deviceIdx)
        check_error(err)

        # Create context
        err, self.context = cuda.cuCtxCreate(0, cuDevice)
        check_error(err)

    def __del__(self):
        err, = cuda.cuCtxDestroy(self.context)
        check_error(err)