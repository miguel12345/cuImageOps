import pytest
import numpy as np
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.affine.affine import Affine
from cuImageOps.operations.gaussianblur.gaussianblur import GaussianBlur
from cuImageOps.operations.rotate.rotate import Rotate
from cuImageOps.operations.scale.scale import Scale


@pytest.fixture
def square_image_grayscale():
    return np.array(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 2, 3, 4, 5],
        ],
        dtype=np.uint8,
    )


def test_gaussianblur_value_ranges(square_image_grayscale, default_stream):

    # If we use a gaussian blur kernel with fill mode set to reflection, we should never get
    # values above the input max or below the input min
    op = GaussianBlur(
        kernel_size=5, sigma=10.0, fillMode=FillMode.REFLECTION, stream=default_stream
    )

    result = op.run(square_image_grayscale).cpu().numpy()

    # Confirm that the result differs from the input
    assert np.array_equal(result, square_image_grayscale) is False

    # Confirm that the values remain within acceptable ranges
    assert result.max() <= square_image_grayscale.max()
    assert result.min() >= square_image_grayscale.min()


def test_gaussianblur_kernel_size_1(square_image_grayscale, default_stream):

    # If we use a gaussian blur with kernel size 1, it should output the same values as the input
    op = GaussianBlur(
        kernel_size=1, sigma=10.0, fillMode=FillMode.REFLECTION, stream=default_stream
    )

    result = op.run(square_image_grayscale).cpu().numpy()

    assert np.array_equal(result, square_image_grayscale)


def test_gaussianblur_invalid_kernel_size_even(square_image_grayscale, default_stream):

    with pytest.raises(ValueError):

        op = GaussianBlur(
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

        op = GaussianBlur(
            kernel_size=-3,
            sigma=10.0,
            fillMode=FillMode.REFLECTION,
            stream=default_stream,
        )
        op.run(square_image_grayscale).cpu().numpy()


def test_gaussianblur_separable_same_results(square_image_grayscale, default_stream):

    # Gaussian blur supports an implementation based on separable filters. We need to ensure the results are the same as the non-separable path
    nonSeparableOp = GaussianBlur(
        kernel_size=5,
        sigma=10.0,
        use_separable_filter=False,
        fillMode=FillMode.CONSTANT,
        stream=default_stream,
    )
    separableOp = GaussianBlur(
        kernel_size=5,
        sigma=10.0,
        use_separable_filter=True,
        fillMode=FillMode.CONSTANT,
        stream=default_stream,
    )

    nonSeparableResult = nonSeparableOp.run(square_image_grayscale).cpu().numpy()
    separableOp = separableOp.run(square_image_grayscale).cpu().numpy()

    assert np.array_equal(nonSeparableResult, separableOp)
