from typing import List, Tuple
from cuImageOps.core.datacontainer import DataContainer
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
import numpy as np

from cuImageOps.utils.cuda import compile_module, copy_data_to_device, get_kernel, get_kernel_launch_dims, run_kernel

class Translate(ImageOperation):

    def __init__(self, translate:Tuple[float,float], fillMode: FillMode = FillMode.CONSTANT, interpolationMode = InterpolationMode.POINT, **kwargs) -> None:
        super().__init__(**kwargs)
        self._translate = translate
        self.fillMode = np.array(int(fillMode),np.uint32)
        self.interpolationMode = np.array(int(interpolationMode),np.uint32)
    
    @property
    def translate(self):
        return self._translate

    @translate.setter
    def translate(self,t):
        self._translate = t

    def __get_module_path(self) -> str:
        return "cuImageOps/operations/translate/translate.cu"

    def __get_kernel_name(self) -> str:
        return "translate"

    def run(self,imageInput:np.array) -> DataContainer:

        imageInput = imageInput.astype(np.float32)
        
        inputShape = imageInput.shape

        if len(imageInput.shape) <= 2:
            inputShape = (*inputShape,1)

        self.input = imageInput
        self.output = np.zeros_like(self.input,dtype=np.float32)
        self.dims = np.array(inputShape,dtype=np.uint32)

        if self.module is None:
            self.module = compile_module(self.__get_module_path(),debug=True)
            self.kernel = get_kernel(self.module,self.__get_kernel_name())

        kernelArguments = [self.output,self.input,np.array([*self._translate],dtype=np.float32),self.dims,self.fillMode,self.interpolationMode]

        dataContainers = copy_data_to_device(kernelArguments,self.stream)

        blocks, threads = get_kernel_launch_dims(self.input)

        run_kernel(self.kernel,blocks,threads,dataContainers,self.stream)

        #Output is at index 0
        return dataContainers[0]