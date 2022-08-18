from typing import List, Tuple
from imageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
import numpy as np

class Rotate(ImageOperation):

    def __init__(self, theta:Tuple[float],pivot:Tuple[float,float] = (0.0,0.0), fillMode: FillMode = FillMode.CONSTANT,interpolationMode = InterpolationMode.POINT, **kwargs) -> None:
        super().__init__(**kwargs)
        self.theta = np.array(theta,dtype=np.float32)
        self.pivot = np.array([*pivot],dtype=np.float32)
        self.fillMode = np.array(int(fillMode),np.uint32)
        self.interpolationMode = np.array(int(interpolationMode),np.uint32)
    
    def _Operation__get_kernel(self) -> str:
        return open(f"imageOps/operations/{self._Operation__get_kernel_name()}/{self._Operation__get_kernel_name()}.cu").read()

    def _Operation__get_kernel_name(self) -> str:
        return "rotate"

    def _get_kernel_arguments(self)-> List[np.array]:
        return [self.input,self.output,self.theta,self.pivot,self.dims,self.fillMode,self.interpolationMode]

    def _Operation__get_kernel_out_idx(self) -> int:
        return 1