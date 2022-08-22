import pytest
import numpy as np
from cuImageOps.core.imageoperation import FillMode
from cuImageOps.operations.boxblur.boxblur import BoxBlur


@pytest.fixture
def square_image_grayscale():
    return np.array(
        [
            [2, 2, 2, 2, 2],
            [2, 4, 2, 4, 2],
            [2, 2, 2, 2, 2],
            [2, 4, 2, 4, 2],
            [2, 2, 2, 2, 2],
        ],
        dtype=np.float32,
    )


def test_gaussianblur_kernel_size_1(square_image_grayscale, default_stream):

    # If we use a gaussian blur with kernel size 1, it should output the same values as the input
    op = BoxBlur(kernel_size=1, fillMode=FillMode.REFLECTION, stream=default_stream)

    result = op.run(square_image_grayscale).cpu().numpy()

    assert np.allclose(result, square_image_grayscale)


def test_gaussianblur_invalid_kernel_size_even(square_image_grayscale, default_stream):

    with pytest.raises(ValueError):

        op = BoxBlur(
            kernel_size=2,
            sigma=10.0,
            fillMode=FillMode.REFLECTION,
            stream=default_stream,
        )
        op.run(square_image_grayscale).cpu().numpy()


def test_gaussianblur_invalid_kernel_size_negative(
    square_image_grayscale, default_stream
):

    with pytest.raises(ValueError):

        op = BoxBlur(
            kernel_size=-3,
            sigma=10.0,
            fillMode=FillMode.REFLECTION,
            stream=default_stream,
        )
        op.run(square_image_grayscale).cpu().numpy()
