from functools import partial
import timeit
import cv2
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.gaussianblur.gaussianblur import GaussianBlur

image = cv2.imread("/workspace/tests/data/grayscale_dog.jpg", cv2.IMREAD_UNCHANGED)

stream = CudaStream()

for kernel_size in [1, 3, 7, 9, 11, 13, 15, 17, 19, 21]:

    separable_gaussian_blur_op = GaussianBlur(
        kernel_size=kernel_size,
        sigma=100.0,
        use_separable_filter=True,
        interpolation_mode=InterpolationMode.POINT,
        fillMode=FillMode.REFLECTION,
        stream=stream,
    )
    non_separable_gaussian_blur_op = GaussianBlur(
        kernel_size=kernel_size,
        sigma=100.0,
        use_separable_filter=False,
        interpolation_mode=InterpolationMode.POINT,
        fillMode=FillMode.REFLECTION,
        stream=stream,
    )

    def run_op(operation, img):
        operation.run(img).cpu()

    print(
        f"Separable implementation for kernel size {kernel_size} takes {timeit.timeit(partial(run_op,separable_gaussian_blur_op,image),number=100)} seconds"
    )
    print(
        f"Full implementation for kernel size {kernel_size} takes {timeit.timeit(partial(run_op,non_separable_gaussian_blur_op,image),number=100)} seconds"
    )
