import cv2
import numpy as np
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.gaussianblur.gaussianblur import GaussianBlur
from cuImageOps.operations.rotate.rotate import Rotate

stream = CudaStream()

image = cv2.imread("tests/data/grayscale_dog.jpg",cv2.IMREAD_UNCHANGED)

op = GaussianBlur(kernelSize=5,sigma=100.0,interpolationMode=InterpolationMode.POINT,fillMode=FillMode.REFLECTION,stream=stream)

result = op.run(image).cpu().numpy()

cv2.imshow("Result",result.astype(np.uint8))
cv2.waitKey()

