from typing import List, Tuple
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
from cuImageOps.utils.cuda import *
from cuImageOps.utils.geometric import *
import numpy as np

class Affine(ImageOperation):

    def __init__(self, translate:Tuple[float,float] = (0.,0.), rotate:float = 0.0, scale:Tuple[float,float] = (1.,1.), shear:Tuple[float,float] = (0.,0.),pivot:Tuple[float,float] = (0.5,0.5), fillMode: FillMode = FillMode.CONSTANT,interpolationMode = InterpolationMode.POINT, **kwargs) -> None:
        super().__init__(**kwargs)
        self.translate = translate
        self.rotate = rotate
        self.scale = scale
        self.shear = shear
        self.pivot = pivot
        self.fillMode = np.array(int(fillMode),np.uint32)
        self.interpolationMode = np.array(int(interpolationMode),np.uint32)
    
    def __compute_affine_matrix(self, image: np.array) -> np.array:
        pivotAbs = (self.pivot[0] * (image.shape[1]-1), self.pivot[1] * (image.shape[0]-1))
        Tp = translate_mat(-pivotAbs[0],-pivotAbs[1])
        S = scale_mat(self.scale[0],self.scale[1])
        STp = np.matmul(S,Tp)
        R = rotation_mat_deg(self.rotate)
        RSTp = np.matmul(R,STp)
        T = translate_mat(self.translate[0]+pivotAbs[0],self.translate[1]+pivotAbs[1])
        TRSTp = np.matmul(T,RSTp)

        #Since the kernel needs to calculate the original point to sample from based on the final position, we need to use the inverted matrix

        return np.linalg.inv(TRSTp)
    

    def __get_module_path(self) -> str:
        return os.path.join(os.path.dirname(cuImageOps.__file__), "operations", self.__get_kernel_name(), f"{self.__get_kernel_name()}.cu")

    def __get_kernel_name(self) -> str:
        return "affine"

    def _get_kernel_arguments(self)-> List[np.array]:
        return [self.input,self.output,self.__compute_affine_matrix(self.input),self.dims,self.fillMode,self.interpolationMode]

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

        kernelArguments = [self.output,self.input,self.__compute_affine_matrix(self.input),self.dims,self.fillMode,self.interpolationMode]

        dataContainers = copy_data_to_device(kernelArguments,self.stream)

        blocks, threads = get_kernel_launch_dims(self.input)

        run_kernel(self.kernel,blocks,threads,dataContainers,self.stream)

        #Output is at index 0
        return dataContainers[0]