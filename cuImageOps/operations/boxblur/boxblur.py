import os
import numpy as np
import cuImageOps
from cuImageOps.core.datacontainer import DataContainer
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
import cuImageOps.utils.cuda as cuda_utils


class BoxBlur(ImageOperation):
    def __init__(
        self,
        kernel_size: int,
        use_separable_filter: bool = False,
        fillMode: FillMode = FillMode.CONSTANT,
        interpolation_mode: InterpolationMode = InterpolationMode.POINT,
        **kwargs
    ) -> None:
        """Instantiates a box blur operation

        Args:
            kernel_size (int): Kernel size. Must be a positive odd number
            use_separable_filter (bool, optional): Whether to use separable filters, improving performance on larger kernel sizes. Defaults to False.
            fillMode (FillMode, optional): Policy for sampling pixels outside the frame. Defaults to FillMode.CONSTANT.
            interpolation_mode (InterpolationMode, optional): Interpolation mode, can be point or linear. Defaults to InterpolationMode.POINT.

        Raises:
            ValueError: Raised when the kernel size is negative or even
        """
        super().__init__(**kwargs)

        if kernel_size % 2 == 0:
            raise ValueError("Box blur only accepts odd kernel sizes")

        if kernel_size < 2:
            raise ValueError("Box blur only accepts kernel sizes greater than one")

        self.kernel_size = kernel_size
        self.fill_mode = fillMode
        self.interpolation_mode = interpolation_mode
        self.use_separable_filter = use_separable_filter
        self.dims = None
        self.intermediate_output = None
        self.kernel_separable_horizontal = None
        self.kernel_separable_vertical = None

    def __get_module_path(self) -> str:
        return os.path.join(
            os.path.dirname(cuImageOps.__file__),
            "operations",
            "blur",
            "blur.cu",
        )

    def __compute_2d_convolution_kernel(self) -> np.array:
        weights_1d = np.expand_dims(
            np.array(
                [1] * self.kernel_size,
                dtype=np.float32,
            ),
            -1,
        )
        kernel_weights = np.matmul(weights_1d, weights_1d.T)
        # Renormalize kernel
        kernel_weights /= kernel_weights.sum()
        return kernel_weights

    def __compute_1d_convolution_kernel(self) -> np.array:
        weights_1d = np.expand_dims(
            np.array(
                [1] * self.kernel_size,
                dtype=np.float32,
            ),
            -1,
        )
        # Renormalize kernel
        weights_1d /= weights_1d.sum()
        return weights_1d

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
        self.intermediate_output = np.zeros_like(
            self.input, dtype=np.float32
        )  # For separable filter
        self.dims = np.array(input_shape, dtype=np.uint32)

        if self.module is None:
            self.module = cuda_utils.compile_module(
                self.__get_module_path(), debug=True
            )
            self.kernel = cuda_utils.get_kernel(self.module, "blur")
            self.kernel_separable_horizontal = cuda_utils.get_kernel(
                self.module, "blurHorizontal"
            )
            self.kernel_separable_vertical = cuda_utils.get_kernel(
                self.module, "blurVertical"
            )

        if self.use_separable_filter:
            conv_kernel_weights = self.__compute_1d_convolution_kernel()
        else:
            conv_kernel_weights = self.__compute_2d_convolution_kernel()

        kernel_arguments = [
            self.output,
            self.intermediate_output,
            self.input,
            conv_kernel_weights,
            np.array(self.kernel_size, dtype=np.uint32),
            self.dims,
            np.array(self.fill_mode, dtype=np.uint32),
            np.array(self.interpolation_mode, dtype=np.uint32),
        ]

        data_containers = cuda_utils.copy_data_to_device(kernel_arguments, self.stream)

        blocks, threads = cuda_utils.get_kernel_launch_dims(self.input)

        if self.use_separable_filter:
            # Run horizontal kernel on input and write to intermediate output
            cuda_utils.run_kernel(
                self.kernel_separable_horizontal,
                blocks,
                threads,
                data_containers[1:],
                self.stream,
            )
            # Run vertical kernel on intermediate output and write to final output
            cuda_utils.run_kernel(
                self.kernel_separable_vertical,
                blocks,
                threads,
                [data_containers[0], data_containers[1]] + data_containers[3:],
                self.stream,
            )
        else:
            cuda_utils.run_kernel(
                self.kernel,
                blocks,
                threads,
                [data_containers[0]] + data_containers[2:],
                self.stream,
            )

        # Output is at index 0
        return data_containers[0]
