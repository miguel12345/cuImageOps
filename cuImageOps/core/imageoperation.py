from enum import IntEnum
from .operation import Operation


class FillMode(IntEnum):
    CONSTANT = (1,)
    REFLECTION = 2


class InterpolationMode(IntEnum):
    POINT = (1,)
    LINEAR = 2


class ImageOperation(Operation):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.input = None
        self.output = None
        self.dims = None
