from typing import List, Tuple
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
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
    

    def _Operation__get_kernel(self) -> str:
        return open(f"cuImageOps/operations/{self._Operation__get_kernel_name()}/{self._Operation__get_kernel_name()}.cu").read()

    def _Operation__get_kernel_name(self) -> str:
        return "affine"

    def _get_kernel_arguments(self)-> List[np.array]:
        return [self.input,self.output,self.__compute_affine_matrix(self.input),self.dims,self.fillMode,self.interpolationMode]

    def _Operation__get_kernel_out_idx(self) -> int:
        return 1