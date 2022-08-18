from typing import List, Tuple
from imageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
import numpy as np

class Translate(ImageOperation):

    def __init__(self, translate:Tuple[float,float], fillMode: FillMode, interpolationMode = InterpolationMode.POINT, **kwargs) -> None:
        super().__init__(**kwargs)
        self.translate = np.array([*translate],dtype=np.float32)
        self.fillMode = np.array(int(fillMode),np.uint32)
        self.interpolationMode = np.array(int(interpolationMode),np.uint32)
    
    def _Operation__get_kernel(self) -> str:
        return open("imageOps/operations/translate/translate.cu").read()

    def _Operation__get_kernel_name(self) -> str:
        return "translate"

    def _get_kernel_arguments(self)-> List[np.array]:
        return [self.input,self.translate,self.output,self.dims,self.fillMode,self.interpolationMode]

    def _Operation__get_kernel_out_idx(self) -> int:
        return 2