from typing import List
import numpy as np
from cuImageOps.core.datacontainer import DataContainer
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
from cuImageOps.utils.cuda import *
from cuImageOps.utils.utils import gaussian

class GaussianBlur(ImageOperation):

    def __init__(self, kernelSize: int, sigma: float, useSeparableFilter: bool = False, fillMode: FillMode = FillMode.CONSTANT,interpolationMode = InterpolationMode.POINT, **kwargs) -> None:
        super().__init__(**kwargs)

        if kernelSize % 2 == 0:
            raise ValueError("Gaussian blur only accepts odd kernel sizes")

        if kernelSize < 0:
            raise ValueError("Gaussian blur only positive kernel sizes")

        self.kernelSize = kernelSize
        self.sigma = sigma
        self.fillMode = fillMode
        self.interpolationMode = interpolationMode
        self.useSeparableFilter = useSeparableFilter
    
    def __get_module_path(self) -> str:
        return "cuImageOps/operations/gaussianblur/gaussianblur.cu"

    def __compute_2d_convolution_kernel(self) -> np.array:
        effectiveKernelSize = self.kernelSize-1
        gaussian1d =np.expand_dims(np.array([gaussian(x,self.sigma) for x in range(int(-effectiveKernelSize/2),int(effectiveKernelSize/2)+1)],dtype=np.float32),-1)
        gaussianKernel = np.matmul(gaussian1d,gaussian1d.T)
        #Renormalize kernel
        gaussianKernel /= gaussianKernel.sum()
        return gaussianKernel

    def __compute_1d_convolution_kernel(self) -> np.array:
        effectiveKernelSize = self.kernelSize - 1
        gaussianKernel1d = np.array([gaussian(x, self.sigma) for x in range(int(-effectiveKernelSize / 2), int(effectiveKernelSize / 2) + 1)], dtype=np.float32)
        gaussianKernel1d /= gaussianKernel1d.sum()
        return gaussianKernel1d

    def run(self,imageInput:np.array) -> DataContainer:

        imageInput = imageInput.astype(np.float32)
        
        inputShape = imageInput.shape

        if len(imageInput.shape) <= 2:
            inputShape = (*inputShape,1)

        self.input = imageInput
        self.output = np.zeros_like(self.input,dtype=np.float32)
        self.intermediateOutput = np.zeros_like(self.input,dtype=np.float32) #For separable filter
        self.dims = np.array(inputShape,dtype=np.uint32)

        if self.module is None:
            self.module = compile_module(self.__get_module_path(),debug=True)
            self.kernel = get_kernel(self.module,"gaussianblur")
            self.kernelSeparableHorizontal = get_kernel(self.module,"gaussianblurHorizontal")
            self.kernelSeparableVertical = get_kernel(self.module,"gaussianblurVertical")

        if self.useSeparableFilter:
            convKernelWeights = self.__compute_1d_convolution_kernel()
        else:
            convKernelWeights = self.__compute_2d_convolution_kernel()

        kernelArguments = [self.output,self.intermediateOutput,self.input,convKernelWeights,np.array(self.kernelSize,dtype=np.uint32),self.dims,np.array(self.fillMode,dtype=np.uint32),np.array(self.interpolationMode,dtype=np.uint32)]

        dataContainers = copy_data_to_device(kernelArguments,self.stream)

        blocks, threads = get_kernel_launch_dims(self.input)

        if self.useSeparableFilter:
            #Run horizontal kernel on input and write to intermediate output
            run_kernel(self.kernelSeparableHorizontal,blocks,threads,dataContainers[1:],self.stream)
            #Run vertical kernel on intermediate output and write to final output
            run_kernel(self.kernelSeparableVertical,blocks,threads,[dataContainers[0],dataContainers[1]]+dataContainers[3:],self.stream)
        else:
            run_kernel(self.kernel,blocks,threads,[dataContainers[0]]+dataContainers[2:],self.stream)

        #Output is at index 0
        return dataContainers[0]