from textwrap import fill
from imageOps.core.cuda.stream import CudaStream
from imageOps.operations.saxpy.saxpy import *
from imageOps.operations.translate.translate import *

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

image = cv2.imread("tests/data/grayscale_dog.jpg",cv2.IMREAD_UNCHANGED)
op = Translate(image=image.astype(np.float32),translate=[10,40],fillMode=FillMode.REFLECTION, stream=stream)
output = op.run().cpu().numpy()
cv2.imshow("Result",output.astype(np.uint8))
cv2.waitKey()
print(f"Translate Output \n {output}")