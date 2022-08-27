import cv2
import numpy as np
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.affine.affine import Affine
from cuImageOps.operations.boxblur.boxblur import BoxBlur
from cuImageOps.operations.distortion.distortion import Distortion
from cuImageOps.operations.gaussianblur.gaussianblur import GaussianBlur
from cuImageOps.operations import HistogramEqualization

stream = CudaStream()

image = cv2.imread("/workspace/tests/data/grayscale_dog.jpg", cv2.IMREAD_UNCHANGED)
# image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def draw_label(img, label):
    return cv2.putText(
        cv2.cvtColor(img, cv2.COLOR_GRAY2BGR),
        label,
        (5, 20),
        cv2.FONT_HERSHEY_PLAIN,
        1.0,
        (0, 0, 255),
    )


results = []

results.append(draw_label(image.astype(np.float32), "Original"))


affine_op = Affine(
    translate=(10, 10),
    rotate=45,
    scale=(0.5, 1.0),
    interpolationMode=InterpolationMode.LINEAR,
    fillMode=FillMode.REFLECTION,
    stream=stream,
)
results.append(draw_label(affine_op.run(image).cpu().numpy(), "Affine transformation"))

box_blur_op = BoxBlur(
    kernel_size=7,
    fillMode=FillMode.REFLECTION,
    stream=stream,
)
results.append(draw_label(box_blur_op.run(image).cpu().numpy(), "Box blur"))


gaussian_blur_op = GaussianBlur(
    kernel_size=7,
    sigma=100.0,
    use_separable_filter=True,
    interpolation_mode=InterpolationMode.POINT,
    fillMode=FillMode.REFLECTION,
    stream=stream,
)

results.append(draw_label(gaussian_blur_op.run(image).cpu().numpy(), "Gaussian blur"))

barrel_distortion_op = Distortion(
    k1=0.7,
    fillMode=FillMode.CONSTANT,
    interpolationMode=InterpolationMode.LINEAR,
    stream=stream,
)

results.append(
    draw_label(barrel_distortion_op.run(image).cpu().numpy(), "Barrel distortion")
)


pincushion_distortion_op = Distortion(
    k1=-0.7,
    fillMode=FillMode.CONSTANT,
    interpolationMode=InterpolationMode.LINEAR,
    stream=stream,
)
results.append(
    draw_label(
        pincushion_distortion_op.run(image).cpu().numpy(), "Pincushion distortion"
    )
)

histogram_equalization_op = HistogramEqualization(
    stream=stream,
)
results.append(
    draw_label(
        histogram_equalization_op.run(image).cpu().numpy().astype(np.float32),
        "Histogram equalization",
    )
)

grid_width = 3
rows = []

for i in range(0, len(results), grid_width):
    max_idx = min(len(results), i + grid_width)
    imgs_to_concat = results[i:max_idx]
    while len(imgs_to_concat) < grid_width:
        imgs_to_concat.append(np.zeros_like(results[0], dtype=np.float32))

    rows.append(cv2.hconcat(imgs_to_concat))

final_result = cv2.vconcat(rows)

cv2.imwrite("images/operation_gallery.png", final_result)
cv2.imshow("Result", final_result.astype(np.uint8))
cv2.waitKey()
