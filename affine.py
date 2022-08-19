import cv2
import numpy as np
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.imageoperation import InterpolationMode
from cuImageOps.operations.affine.affine import Affine

stream = CudaStream()

image = cv2.imread("tests/data/rgb_dog.jpg",cv2.IMREAD_UNCHANGED)

op = Affine(translate=(0,0),rotate=90,scale=(1.,.5),interpolationMode=InterpolationMode.LINEAR,stream=stream)

result = op.run(image).cpu().numpy()

cv2.imshow("Result",result.astype(np.uint8))
cv2.waitKey()

