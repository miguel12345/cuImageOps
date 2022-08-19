import cv2
import numpy as np
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.imageoperation import InterpolationMode
from cuImageOps.operations.translate.translate import Translate

stream = CudaStream()

img = cv2.imread("tests/data/red_rectangle_transparent_background.png",cv2.IMREAD_UNCHANGED)


xDelta = 0.0

op = Translate((xDelta,0.0),interpolationMode=InterpolationMode.LINEAR,stream=stream)


while xDelta < 50.0:
    print(f"xDelta {xDelta}")
    op.translate = (xDelta,xDelta)
    result = op.run(img).cpu().numpy()

    cv2.imshow("Output",result.astype(np.uint8))
    cv2.imwrite("transparent_test.png",result)
    cv2.waitKey(100)
    xDelta += 0.2