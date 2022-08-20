from typing import List
import numpy as np
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
from cuImageOps.utils.utils import gaussian

class GaussianBlur(ImageOperation):

    def __init__(self, kernelSize: int, sigma: float, fillMode: FillMode = FillMode.CONSTANT,interpolationMode = InterpolationMode.POINT, **kwargs) -> None:
        super().__init__(**kwargs)

        if kernelSize % 2 == 0:
            raise ValueError("Gaussian blur only accepts odd kernel sizes")

        if kernelSize < 0:
            raise ValueError("Gaussian blur only positive kernel sizes")

        self.kernelSize = kernelSize
        self.sigma = sigma
        self.fillMode = fillMode
        self.interpolationMode = interpolationMode
    
    def _Operation__get_kernel(self) -> str:
        return open(f"cuImageOps/operations/{self._Operation__get_kernel_name()}/{self._Operation__get_kernel_name()}.cu").read()

    def _Operation__get_kernel_name(self) -> str:
        return "gaussianblur"

    def __compute_convolution_kernel(self) -> np.array:
        effectiveKernelSize = self.kernelSize-1
        gaussian1d =np.expand_dims(np.array([gaussian(x,self.sigma) for x in range(int(-effectiveKernelSize/2),int(effectiveKernelSize/2)+1)],dtype=np.float32),-1)
        gaussianKernel = np.matmul(gaussian1d,gaussian1d.T)
        #Renoramlitze kernel
        gaussianKernel /= gaussianKernel.sum()
        return gaussianKernel

    def _get_kernel_arguments(self)-> List[np.array]:
        return [self.input,self.output,self.__compute_convolution_kernel(),np.array(self.kernelSize,dtype=np.uint32),self.dims,np.array(self.fillMode,dtype=np.uint32),np.array(self.interpolationMode,dtype=np.uint32)]

    def _Operation__get_kernel_out_idx(self) -> int:
        return 1