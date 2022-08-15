from typing import List, Tuple
from imageOps.core.operation import Operation
import numpy as np

class SAXPY(Operation):

    def __init__(self, a:np.array, x:np.array, y: np.array) -> None:
        super().__init__()
        self.a = a
        self.x = x
        self.y = y
        self.out = np.zeros_like(self.x,dtype=np.float32)
        self.outLen = np.array(len(self.out),dtype=np.uint32)
    
    def _Operation__get_kernel(self) -> str:
        return open("imageOps\operations\saxpy\saxpy.cu").read()

    def _Operation__get_kernel_name(self) -> str:
        return "saxpy"

    def _Operation__get_kernel_arguments(self)-> List[np.array]:
        return [self.a,self.x,self.y,self.out,self.outLen]

    def _Operation__get_kernel_out_idx(self) -> int:
        return 3