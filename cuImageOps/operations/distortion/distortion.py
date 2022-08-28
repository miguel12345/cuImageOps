from typing import List, Tuple
import os
import numpy as np
from cuImageOps.core.datacontainer import DataContainer
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
import cuImageOps
import cuImageOps.utils.cuda as cuda_utils
import cuImageOps.utils.geometric as geometric_utils


class Distortion(ImageOperation):
    def __init__(
        self,
        k1: float = 0.0,
        k2: float = 0.0,
        k3: float = 0.0,
        p1: float = 0.0,
        p2: float = 0.0,
        fillMode: FillMode = FillMode.CONSTANT,
        interpolationMode=InterpolationMode.POINT,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.fill_mode = np.array(int(fillMode), np.uint32)
        self.interpolation_mode = np.array(int(interpolationMode), np.uint32)
        self.dims = None

    def __get_module_path(self) -> str:
        return os.path.join(
            os.path.dirname(cuImageOps.__file__),
            "operations",
            self.__get_kernel_name(),
            f"{self.__get_kernel_name()}.cu",
        )

    def __get_kernel_name(self) -> str:
        return "distortion"

    def run(self, image_input: np.array) -> DataContainer:
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
        self.output = np.zeros_like(self.input)
        self.dims = np.array(input_shape, dtype=np.uint32)

        if self.module is None:
            self.module = cuda_utils.compile_module(
                self.__get_module_path(), debug=True
            )
            self.kernel = cuda_utils.get_kernel(self.module, self.__get_kernel_name())

        kernel_arguments = [
            self.output,
            self.input,
            np.array(self.k1, dtype=np.float32),
            np.array(self.k2, dtype=np.float32),
            np.array(self.k3, dtype=np.float32),
            np.array(self.p1, dtype=np.float32),
            np.array(self.p2, dtype=np.float32),
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
