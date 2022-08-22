from typing import List, Tuple
import os
import numpy as np
from cuImageOps.core.datacontainer import DataContainer
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
import cuImageOps
import cuImageOps.utils.cuda as cuda_utils
import cuImageOps.utils.geometric as geometric_utils


class Affine(ImageOperation):
    def __init__(
        self,
        translate: Tuple[float, float] = (0.0, 0.0),
        rotate: float = 0.0,
        scale: Tuple[float, float] = (1.0, 1.0),
        shear: Tuple[float, float] = (0.0, 0.0),
        pivot: Tuple[float, float] = (0.5, 0.5),
        fillMode: FillMode = FillMode.CONSTANT,
        interpolationMode=InterpolationMode.POINT,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.translate = translate
        self.rotate = rotate
        self.scale = scale
        self.shear = shear
        self.pivot = pivot
        self.fill_mode = np.array(int(fillMode), np.uint32)
        self.interpolation_mode = np.array(int(interpolationMode), np.uint32)
        self.dims = None

    def __compute_affine_matrix(self, image: np.array) -> np.array:
        pivot_abs = (
            self.pivot[0] * (image.shape[1] - 1),
            self.pivot[1] * (image.shape[0] - 1),
        )
        t_p = geometric_utils.translate_mat(-pivot_abs[0], -pivot_abs[1])
        s = geometric_utils.scale_mat(self.scale[0], self.scale[1])
        s_t_p = np.matmul(s, t_p)
        r = geometric_utils.rotation_mat_deg(self.rotate)
        r_s_t_p = np.matmul(r, s_t_p)
        t = geometric_utils.translate_mat(
            self.translate[0] + pivot_abs[0], self.translate[1] + pivot_abs[1]
        )
        t_r_s_t_p = np.matmul(t, r_s_t_p)

        # Since the kernel needs to calculate the original point to sample from based on the final position, we need to use the inverted matrix

        return np.linalg.inv(t_r_s_t_p)

    def __get_module_path(self) -> str:
        return os.path.join(
            os.path.dirname(cuImageOps.__file__),
            "operations",
            self.__get_kernel_name(),
            f"{self.__get_kernel_name()}.cu",
        )

    def __get_kernel_name(self) -> str:
        return "affine"

    def _get_kernel_arguments(self) -> List[np.array]:
        return [
            self.input,
            self.output,
            self.__compute_affine_matrix(self.input),
            self.dims,
            self.fill_mode,
            self.interpolation_mode,
        ]

    def run(self, image_input: np.array) -> DataContainer:
        """Runs the operation on an image and returns the data container for the result

        Args:
            image_input (np.array): Input image to be processed

        Returns:
            DataContainer: Result of the operation. Can be transfered to cpu using cpu()
        """

        image_input = image_input.astype(np.float32)

        input_shape = image_input.shape

        if len(image_input.shape) <= 2:
            input_shape = (*input_shape, 1)

        self.input = image_input
        self.output = np.zeros_like(self.input, dtype=np.float32)
        self.dims = np.array(input_shape, dtype=np.uint32)

        if self.module is None:
            self.module = cuda_utils.compile_module(
                self.__get_module_path(), debug=True
            )
            self.kernel = cuda_utils.get_kernel(self.module, self.__get_kernel_name())

        kernel_arguments = [
            self.output,
            self.input,
            self.__compute_affine_matrix(self.input),
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
