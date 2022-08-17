from typing import List, Tuple
from imageOps.core.imageoperation import FillMode, ImageOperation
import numpy as np

class Translate(ImageOperation):

    def __init__(self, translate:Tuple[int,int], fillMode: FillMode, **kwargs) -> None:
        super().__init__(**kwargs)
        self.translate = np.array([*translate],dtype=np.float32)
        self.fillMode = np.array(int(fillMode),np.uint32)
    
    def _Operation__get_kernel(self) -> str:
        return open("imageOps/operations/translate/translate.cu").read()

    def _Operation__get_kernel_name(self) -> str:
        return "translate"

    def _get_kernel_arguments(self)-> List[np.array]:
        return [self.input,self.translate,self.output,self.dims,self.fillMode]

    def _Operation__get_kernel_out_idx(self) -> int:
        return 2