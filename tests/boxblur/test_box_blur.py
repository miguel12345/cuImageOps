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
        dtype=np.uint8,
    )


def test_boxblur_kernel_size_3(square_image_grayscale, default_stream):

    op = BoxBlur(
        kernel_size=3,
        use_separable_filter=False,
        fillMode=FillMode.CONSTANT,
        stream=default_stream,
    )
    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.rint(
        np.array(
            [
                [1.1111112, 1.5555556, 1.7777778, 1.5555556, 1.1111112],
                [1.5555556, 2.2222223, 2.4444447, 2.2222223, 1.5555556],
                [1.7777778, 2.4444447, 2.888889, 2.4444447, 1.7777778],
                [1.5555556, 2.2222223, 2.4444447, 2.2222223, 1.5555556],
                [1.1111112, 1.5555556, 1.7777778, 1.5555556, 1.1111112],
            ],
            dtype=np.float32,
        )
    )

    assert np.array_equal(result, expected_result)


def test_boxblur_invalid_kernel_size_even(square_image_grayscale, default_stream):

    with pytest.raises(ValueError):

        op = BoxBlur(
            kernel_size=2,
            fillMode=FillMode.REFLECTION,
            stream=default_stream,
        )
        op.run(square_image_grayscale).cpu().numpy()


def test_boxblur_invalid_kernel_size_negative(square_image_grayscale, default_stream):

    with pytest.raises(ValueError):

        op = BoxBlur(
            kernel_size=-3,
            fillMode=FillMode.REFLECTION,
            stream=default_stream,
        )
        op.run(square_image_grayscale).cpu().numpy()
