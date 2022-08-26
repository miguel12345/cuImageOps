import os
import numpy as np
import cuImageOps
from cuImageOps.core.datacontainer import DataContainer
from cuImageOps.core.imageoperation import FillMode, ImageOperation, InterpolationMode
import cuImageOps.utils.cuda as cuda_utils
from cuImageOps.utils.utils import gaussian


class Histogram(ImageOperation):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.dims = None
        self.partial_histogram_kernel = None
        self.global_histogram_kernel = None

        # self.partial_histograms contains all the partial histograms for subg-regions of the image
        self.partial_histograms = None
        # self.global_histogram contains the number of times a value exists in an image
        self.global_histogram = None
        self.num_bins = 256

        self.partial_histograms_dc = None
        self.global_histogram_dc = None
        self.num_partial_histograms_dc = None
        self.input_dc = None
        self.dims_dc = None
        self.num_channels_dc = None

    def __get_module_path(self) -> str:
        return os.path.join(
            os.path.dirname(cuImageOps.__file__),
            "operations",
            "histogram",
            "histogram.cu",
        )

    def run(self, image_input: np.array, debug=False) -> DataContainer:
        """Runs the operation on an image and returns the data container for the result

        Args:
            image_input (np.array): Input image to be processed

        Returns:
            DataContainer: Result of the operation. Can be transfered to cpu using cpu()
        """

        input_shape = image_input.shape
        self.input = image_input
        num_channels = 1

        if len(input_shape) >= 3:
            num_channels = input_shape[2]

        if len(image_input.shape) <= 2:
            input_shape = (*input_shape, 1)

        blocks = (10, 10, 1)
        threads = (16, 16, 1)
        num_partial_histograms = blocks[0] * blocks[1]
        self.partial_histograms = np.zeros(
            (num_partial_histograms, self.num_bins, num_channels), dtype=np.uint32
        )
        self.global_histogram = np.zeros((self.num_bins, num_channels), dtype=np.uint32)

        self.dims = np.array(input_shape, dtype=np.uint32)

        if self.module is None:
            self.module = cuda_utils.compile_module(
                self.__get_module_path(), debug=debug
            )
            self.partial_histogram_kernel = cuda_utils.get_kernel(
                self.module, "partial_histogram"
            )
            self.global_histogram_kernel = cuda_utils.get_kernel(
                self.module, "global_histogram"
            )

        (
            self.partial_histograms_dc,
            self.global_histogram_dc,
            self.num_partial_histograms_dc,
            self.input_dc,
            self.dims_dc,
            self.num_channels_dc,
        ) = cuda_utils.copy_data_to_device(
            [
                self.partial_histograms,
                self.global_histogram,
                np.array(num_partial_histograms, np.uint32),
                self.input,
                self.dims,
                np.array(num_channels, np.uint8),
            ],
            self.stream,
        )

        # Run partial histogram
        cuda_utils.run_kernel(
            self.partial_histogram_kernel,
            blocks,
            threads,
            [
                self.partial_histograms_dc,
                self.input_dc,
                self.dims_dc,
                self.num_channels_dc,
            ],
            self.stream,
        )

        # Run global histogram
        cuda_utils.run_kernel(
            self.global_histogram_kernel,
            (num_partial_histograms, 1, 1),
            (self.num_bins, 1, 1),
            [
                self.global_histogram_dc,
                self.partial_histograms_dc,
                self.num_partial_histograms_dc,
                self.num_channels_dc,
            ],
            self.stream,
        )

        return self.global_histogram_dc
