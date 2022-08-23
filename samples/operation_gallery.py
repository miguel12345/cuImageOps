import cv2
import numpy as np
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.affine.affine import Affine
from cuImageOps.operations.boxblur.boxblur import BoxBlur
from cuImageOps.operations.gaussianblur.gaussianblur import GaussianBlur

stream = CudaStream()

image = cv2.imread("/workspace/tests/data/grayscale_dog.jpg", cv2.IMREAD_UNCHANGED)
image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

results = []

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
    (0, 0, 255),
)
results.append(gaussian_blur_result)

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
    affine_result, "Affine", (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255)
)
results.append(affine_result)

box_blur_op = BoxBlur(
    kernel_size=5,
    fillMode=FillMode.REFLECTION,
    stream=stream,
)
box_blur_result = box_blur_op.run(image).cpu().numpy()
box_blur_result = cv2.putText(
    box_blur_result, "Box Blur", (5, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255)
)
results.append(box_blur_result)


final_result = cv2.hconcat(results)

cv2.imshow("Result", final_result.astype(np.uint8))
cv2.waitKey()
