import cv2
import numpy as np
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.affine.affine import Affine
from cuImageOps.operations.gaussianblur.gaussianblur import GaussianBlur
from cuImageOps.operations.rotate.rotate import Rotate

stream = CudaStream()

image = cv2.imread("/workspaces/imageOps/tests/data/grayscale_dog.jpg",cv2.IMREAD_UNCHANGED)

gaussianBlurOp = GaussianBlur(kernelSize=5,sigma=100.0,useSeparableFilter=True,interpolationMode=InterpolationMode.POINT,fillMode=FillMode.REFLECTION,stream=stream)
gaussianBlurResult = gaussianBlurOp.run(image).cpu().numpy()
gaussianBlurResult = cv2.putText(gaussianBlurResult,"Gaussian blur",(5,20),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0))


affineOp = Affine(translate=(10,10),rotate=45,scale=(.5,1.),interpolationMode=InterpolationMode.LINEAR,fillMode=FillMode.REFLECTION,stream=stream)
affineResult = affineOp.run(image).cpu().numpy()
affineResult = cv2.putText(affineResult,"Affine",(5,20),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0))

finalResult = cv2.hconcat([affineResult,gaussianBlurResult])

cv2.imshow("Result",finalResult.astype(np.uint8))
cv2.waitKey()

