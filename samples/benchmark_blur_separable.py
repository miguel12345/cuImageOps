import timeit
import cv2
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.affine.affine import Affine
from cuImageOps.operations.gaussianblur.gaussianblur import GaussianBlur
from cuImageOps.operations.rotate.rotate import Rotate

image = cv2.imread(
    "/workspaces/imageOps/tests/data/grayscale_dog.jpg", cv2.IMREAD_UNCHANGED
)

stream = CudaStream()

for kernel_size in [1, 3, 7, 9, 11, 13, 15, 17, 19, 21]:

    separableGaussianBlurOp = GaussianBlur(
        kernel_size=kernel_size,
        sigma=100.0,
        use_separable_filter=True,
        interpolation_mode=InterpolationMode.POINT,
        fillMode=FillMode.REFLECTION,
        stream=stream,
    )
    nonseparableGaussianBlurOp = GaussianBlur(
        kernel_size=kernel_size,
        sigma=100.0,
        use_separable_filter=False,
        interpolation_mode=InterpolationMode.POINT,
        fillMode=FillMode.REFLECTION,
        stream=stream,
    )

    def runSeparableGaussianBlurOp():
        separableGaussianBlurOp.run(image).cpu()

    def runNonSeparableGaussianBlurOp():
        nonseparableGaussianBlurOp.run(image).cpu()

    print(
        f"Separable implementation for kernel size {kernel_size} takes {timeit.timeit(runSeparableGaussianBlurOp,number=100)} seconds"
    )
    print(
        f"Full implementation for kernel size {kernel_size} takes {timeit.timeit(runNonSeparableGaussianBlurOp,number=100)} seconds"
    )
