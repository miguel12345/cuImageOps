from cuda import cuda
from cuImageOps.core.cuda.context import CudaContext
from cuImageOps.utils.utils import check_error


class CudaStream:
    def __init__(self, context: CudaContext = None) -> None:
        super().__init__()
        self.module = None
        self.context = context
        self.stream = None

        if self.context is None:
            self.context = CudaContext.default_context()

        err, self.stream = cuda.cuStreamCreate(0)
        check_error(err)

    def native_ptr(self):
        return self.stream

    def __del__(self):

        if self.stream is not None:
            (err,) = cuda.cuStreamDestroy(self.stream)
            check_error(err)
