import pytest
import numpy as np
from cuImageOps.operations.histogramequalization.histogramequalization import (
    HistogramEqualization,
)


@pytest.fixture
def square_image_grayscale():
    return np.array(
        [
            [52, 55, 61, 59, 79, 61, 76, 61],
            [62, 59, 55, 104, 94, 85, 59, 71],
            [63, 65, 66, 113, 144, 104, 63, 72],
            [64, 70, 70, 126, 154, 109, 71, 69],
            [67, 73, 68, 106, 122, 88, 68, 68],
            [68, 79, 60, 70, 77, 66, 58, 75],
            [69, 85, 64, 58, 55, 61, 65, 83],
            [70, 87, 69, 68, 65, 73, 78, 90],
        ],
        dtype=np.uint8,
    )


def test_histogram_equalization_for_small_image(square_image_grayscale, default_stream):

    op = HistogramEqualization(stream=default_stream)

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [
            [0, 12, 53, 32, 190, 53, 174, 53],
            [57, 32, 12, 227, 219, 202, 32, 154],
            [65, 85, 93, 239, 251, 227, 65, 158],
            [73, 146, 146, 247, 255, 235, 154, 130],
            [97, 166, 117, 231, 243, 210, 117, 117],
            [117, 190, 36, 146, 178, 93, 20, 170],
            [130, 202, 73, 20, 12, 53, 85, 194],
            [146, 206, 130, 117, 85, 166, 182, 215],
        ],
        np.uint8,
    )

    assert np.array_equal(result, expected_result)
