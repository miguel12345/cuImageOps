from textwrap import fill
from imageOps.core.cuda.stream import CudaStream
from imageOps.core.imageoperation import InterpolationMode
from imageOps.operations.rotate.rotate import Rotate
from imageOps.operations.scale.scale import *

import os

# os.add_dll_directory(os.path.join(os.environ["CUDA_PATH"],"bin"))

# op = SAXPY(a=np.array(2.0,dtype=np.float32),x=np.array([10.0],dtype=np.float32),y=np.array([5.0],dtype=np.float32))
# output = op.run()
# print(f"SAXPY Output {output}")


# op = Translate(image=np.ones((20,10),dtype=np.float32),translate=[1,1],fillMode=FillMode.CONSTANT)
# output = op.run()
# print(f"Translate Output \n {output}")



import cv2

stream = CudaStream()

image = cv2.imread("tests/data/rgb_dog.jpg",cv2.IMREAD_UNCHANGED)
op = Rotate(theta=0,pivot=[0.5,0.5],fillMode=FillMode.CONSTANT,interpolationMode=InterpolationMode.LINEAR, stream=stream)
output = op.run(image.astype(np.float32)).cpu().numpy()
cv2.imshow("Result",output.astype(np.uint8))
cv2.waitKey()
print(f"Translate Output \n {output}")