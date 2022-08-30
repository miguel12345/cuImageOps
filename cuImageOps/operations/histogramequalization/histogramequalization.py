import os
import numpy as np
import cuImageOps
from cuda import cuda
from cuImageOps.core.cuda.stream import CudaStream
from cuImageOps.core.datacontainer import DataContainer
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
from cuImageOps.operations.histogram.histogram import Histogram
import cuImageOps.utils.cuda as cuda_utils
from cuImageOps.utils.utils import create_np_array_uninitialized_like, gaussian


class HistogramEqualization(ImageOperation):
    def __init__(self, stream: CudaStream = None, **kwargs) -> None:
        super().__init__(stream=stream, **kwargs)

        self.dims = None
        self.cumulative_distribution_kernel = None
        self.histogram_equalization_kernel = None
        self.global_histogram = None
        self.histogram_op = Histogram(stream=stream)
        self.num_bins = self.histogram_op.num_bins
        self.cumulative_distribution = np.zeros(self.num_bins, dtype=np.uint32)

    def __get_module_path(self) -> str:
        return os.path.join(
            os.path.dirname(cuImageOps.__file__),
            "operations",
            "histogramequalization",
            "histogramequalization.cu",
        )

    def run(self, image_input: np.array, debug: bool = False) -> DataContainer:
        """Runs the operation on an image and returns the data container for the result

        Args:
            image_input (np.array): Input image to be processed

        Returns:
            DataContainer: Result of the operation. Can be transfered to cpu using cpu()
        """

        if self.module is None:
            self.module = cuda_utils.compile_module(
                self.__get_module_path(), debug=debug
            )
            self.cumulative_distribution_kernel = cuda_utils.get_kernel(
                self.module, "cumulative_distribution"
            )
            self.histogram_equalization_kernel = cuda_utils.get_kernel(
                self.module, "histogram_equalization"
            )

        output = create_np_array_uninitialized_like(image_input, dtype=np.uint8)
        cumulative_distribution_min = 0

        (
            cumulative_distribution_dc,
            output_dc,
            num_bins_dc,
            cumulative_distribution_min_dc,
        ) = cuda_utils.copy_data_to_device(
            [
                self.cumulative_distribution,
                output,
                np.array(self.num_bins, np.uint32),
                np.array([cumulative_distribution_min], np.uint32),
            ],
            self.stream,
        )

        self.histogram_op.run(image_input, debug)

        input_dc = self.histogram_op.input_dc
        dims_dc = self.histogram_op.dims_dc
        global_histogram_dc = self.histogram_op.global_histogram_dc

        # Calculate cumulative distribution
        cuda_utils.run_kernel(
            self.cumulative_distribution_kernel,
            (1, 1, 1),
            (1, 1, 1),
            [
                cumulative_distribution_dc,
                global_histogram_dc,
                num_bins_dc,
                cumulative_distribution_min_dc,
            ],
            self.stream,
        )

        blocks, threads = cuda_utils.get_kernel_launch_dims(image_input)

        # Equalize histogram
        cuda_utils.run_kernel(
            self.histogram_equalization_kernel,
            blocks,
            threads,
            [
                output_dc,
                input_dc,
                dims_dc,
                num_bins_dc,
                cumulative_distribution_dc,
                cumulative_distribution_min_dc,
            ],
            self.stream,
        )

        return output_dc
