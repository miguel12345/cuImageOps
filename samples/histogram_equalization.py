import cv2
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.operations import HistogramEqualization, Histogram
import matplotlib.pyplot as plt
import numpy as np


def render_histogram(histogram):
    fig, (ax) = plt.subplots()
    ax.bar(np.arange(histogram.shape[0]), histogram.squeeze(), width=2)
    ax.set_title("Histogram")
    ax.set_xlim(0, 255)
    ax.set_ylim(0, histogram.max())
    fig.canvas.draw()
    return np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]


stream = CudaStream()

image = cv2.imread(
    "/workspace/tests/data/Unequalized_Hawkes_Bay_NZ.jpg", cv2.IMREAD_UNCHANGED
)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Perform histogram equalization

histogram_op = Histogram(stream=stream)

before_histogram = histogram_op.run(image).cpu().numpy()
before_histogram_plot_img = render_histogram(before_histogram)

histogram_equalization_op = HistogramEqualization(
    stream=stream,
)

equalized_img = histogram_equalization_op.run(image).cpu().numpy()
after_histogram = histogram_op.run(equalized_img).cpu().numpy()

# Display

after_histogram_plot_img = render_histogram(after_histogram)

before_histogram_plot_img = cv2.resize(
    before_histogram_plot_img, (image.shape[1], image.shape[0])
)
after_histogram_plot_img = cv2.resize(
    after_histogram_plot_img, (image.shape[1], image.shape[0])
)

image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
equalized_image_rgb = cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR)

final_img = cv2.vconcat(
    [
        cv2.hconcat([image_rgb, before_histogram_plot_img]),
        cv2.hconcat([equalized_image_rgb, after_histogram_plot_img]),
    ]
)

cv2.imwrite("samples/output/histogram_equalization.jpg", final_img)
cv2.imshow("Comparison", final_img)
cv2.waitKey()
