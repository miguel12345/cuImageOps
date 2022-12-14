import os
from typing import Tuple
import numpy as np
import cuImageOps
from cuImageOps.core.datacontainer import DataContainer
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode

import cuImageOps.utils.cuda as cuda_utils
from cuImageOps.utils.utils import create_np_array_uninitialized_like


class Translate(ImageOperation):
    def __init__(
        self,
        translate: Tuple[float, float],
        fillMode: FillMode = FillMode.CONSTANT,
        interpolationMode=InterpolationMode.POINT,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._translate = translate
        self.fill_mode = np.array(int(fillMode), np.uint32)
        self.interpolation_mode = np.array(int(interpolationMode), np.uint32)
        self.dims = None

    @property
    def translate(self):
        return self._translate

    @translate.setter
    def translate(self, t):
        self._translate = t

    def __get_module_path(self) -> str:
        return os.path.join(
            os.path.dirname(cuImageOps.__file__),
            "operations",
            self.__get_kernel_name(),
            f"{self.__get_kernel_name()}.cu",
        )

    def __get_kernel_name(self) -> str:
        return "translate"

    def run(self, image_input: np.array, debug: bool = False) -> DataContainer:
        """Runs the operation on an image and returns the data container for the result

        Args:
            image_input (np.array): Input image to be processed

        Returns:
            DataContainer: Result of the operation. Can be transfered to cpu using cpu()
        """

        input_shape = image_input.shape

        if len(image_input.shape) <= 2:
            input_shape = (*input_shape, 1)

        self.input = image_input
        self.output = create_np_array_uninitialized_like(self.input)
        self.dims = np.array(input_shape, dtype=np.uint32)

        if self.module is None:
            self.module = cuda_utils.compile_module(
                self.__get_module_path(), debug=debug
            )
            self.kernel = cuda_utils.get_kernel(self.module, self.__get_kernel_name())

        kernel_arguments = [
            self.output,
            self.input,
            np.array([*self._translate], dtype=np.float32),
            self.dims,
            self.fill_mode,
            self.interpolation_mode,
        ]

        data_containers = cuda_utils.copy_data_to_device(kernel_arguments, self.stream)

        blocks, threads = cuda_utils.get_kernel_launch_dims(self.input)

        cuda_utils.run_kernel(
            self.kernel, blocks, threads, data_containers, self.stream
        )

        # Output is at index 0
        return data_containers[0]
