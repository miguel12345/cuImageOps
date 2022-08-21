import cv2
import numpy as np
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.affine.affine import Affine
from cuImageOps.operations.gaussianblur.gaussianblur import GaussianBlur
from cuImageOps.operations.rotate.rotate import Rotate

stream = CudaStream()

image = cv2.imread(
    "/workspaces/imageOps/tests/data/grayscale_dog.jpg", cv2.IMREAD_UNCHANGED
)

gaussian_blur_op = GaussianBlur(
    kernel_size=5,
    sigma=100.0,
    use_separable_filter=True,
    interpolation_mode=InterpolationMode.POINT,
    fillMode=FillMode.REFLECTION,
    stream=stream,
)
gaussian_blur_result = gaussian_blur_op.run(image).cpu().numpy()
gaussian_blur_result = cv2.putText(
    gaussian_blur_result,
    "Gaussian blur",
    (5, 20),
    cv2.FONT_HERSHEY_PLAIN,
    1.0,
    (0, 0, 0),
)


affine_op = Affine(
    translate=(10, 10),
    rotate=45,
    scale=(0.5, 1.0),
    interpolationMode=InterpolationMode.LINEAR,
    fillMode=FillMode.REFLECTION,
    stream=stream,
)
affine_result = affine_op.run(image).cpu().numpy()
affine_result = cv2.putText(
    affine_result, "Affine", (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0)
)

final_result = cv2.hconcat([affine_result, gaussian_blur_result])

cv2.imshow("Result", final_result.astype(np.uint8))
cv2.waitKey()
