import cv2
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.operations import HistogramEqualization

stream = CudaStream()

image = cv2.imread(
    "/workspace/tests/data/Unequalized_Hawkes_Bay_NZ.jpg", cv2.IMREAD_UNCHANGED
)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


histogram_equalization_op = HistogramEqualization(
    stream=stream,
)

result = histogram_equalization_op.run(image).cpu().numpy()

cv2.imshow("Result", result)
cv2.waitKey()
