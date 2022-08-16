from cuda import cuda
from imageOps.core.cuda.context import CudaContext

from imageOps.utils.cuda import check_error

class CudaStream():
    _defaultContext = None

    def __init__(self, context: CudaContext = None) -> None:
        super().__init__()
        self.module = None
        self.context = context

        if self.context is None:

            if CudaStream._defaultContext is None:
                CudaStream._defaultContext = CudaContext()

            self.context = CudaStream._defaultContext
        
        err, self.stream = cuda.cuStreamCreate(0)
        check_error(err)

    def __del__(self):
        err, = cuda.cuStreamDestroy(self.stream)
        check_error(err)