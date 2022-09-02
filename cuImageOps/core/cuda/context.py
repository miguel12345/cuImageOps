from cuda import cuda

from cuImageOps.utils.utils import check_error


class CudaContext:

    __defaultContext = None

    @staticmethod
    def default_context():

        if CudaContext.__defaultContext is None:
            CudaContext.__defaultContext = CudaContext()

        return CudaContext.__defaultContext

    @staticmethod
    def destroy_default_context():

        if CudaContext.__defaultContext is None:
            del CudaContext.__defaultContext

    def __init__(self, deviceIdx: int = 0) -> None:

        self.context = None

        # Initialize CUDA Driver API
        (err,) = cuda.cuInit(0)
        check_error(err)

        # Retrieve handle for device 0
        err, cuDevice = cuda.cuDeviceGet(deviceIdx)
        check_error(err)

        # Create context
        err, self.context = cuda.cuCtxCreate(0, cuDevice)
        check_error(err)

    def __del__(self):

        if self.context is not None:
            (err,) = cuda.cuCtxDestroy(self.context)
            check_error(err)
