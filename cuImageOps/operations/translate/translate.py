from typing import List, Tuple
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
import numpy as np

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

    def _Operation__get_kernel(self) -> str:
        return open("cuImageOps/operations/translate/translate.cu").read()

    def _Operation__get_kernel_name(self) -> str:
        return "translate"

    def _get_kernel_arguments(self)-> List[np.array]:
        return [self.input,np.array([*self._translate],dtype=np.float32),self.output,self.dims,self.fillMode,self.interpolationMode]

    def _Operation__get_kernel_out_idx(self) -> int:
        return 2