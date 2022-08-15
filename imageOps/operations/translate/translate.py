from typing import List, Tuple
from imageOps.core.image_operation import FillMode, ImageOperation
import numpy as np

class Translate(ImageOperation):

    def __init__(self, image:np.array, translate:Tuple[int,int], fillMode: FillMode) -> None:
        super().__init__()
        self.image = image
        self.translate = np.array([*translate],dtype=np.float32)
        self.out = np.zeros_like(self.image,dtype=np.float32)
        self.dims = np.array((self.image.shape[0],self.image.shape[1]),dtype=np.uint32)
        self.fillMode = np.array(int(fillMode),np.uint32)
    
    def _Operation__get_kernel(self) -> str:
        return open("imageOps\\operations\\translate\\translate.cu").read()

    def _Operation__get_kernel_name(self) -> str:
        return "translate"

    def _get_kernel_arguments(self)-> List[np.array]:
        return [self.image,self.translate,self.out,self.dims,self.fillMode]

    def _Operation__get_kernel_out_idx(self) -> int:
        return 2