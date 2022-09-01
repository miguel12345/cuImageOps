from cuda import cuda
from cuImageOps.core.cuda.context import CudaContext
from cuImageOps.utils.utils import check_error


class CudaStream:
    _defaultContext = None

    def __init__(self, context: CudaContext = None) -> None:
        super().__init__()
        self.module = None
        self.context = context
        self.stream = None

        if self.context is None:

            if CudaStream._defaultContext is None:
                CudaStream._defaultContext = CudaContext()

            self.context = CudaStream._defaultContext

        err, self.stream = cuda.cuStreamCreate(0)
        check_error(err)

    def native_ptr(self):
        return self.stream

    def __del__(self):

        print("Destroying stream")
        if self.stream is not None:
            (err,) = cuda.cuStreamDestroy(self.stream)
            check_error(err)
