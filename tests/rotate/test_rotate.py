import pytest
import numpy as np
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.rotate.rotate import Rotate
from cuImageOps.operations.scale.scale import Scale


@pytest.fixture
def square_image_grayscale():
    return np.array(
        [
            [1, 1, 1, 1, 5],
            [1, 1, 1, 1, 4],
            [1, 1, 1, 1, 3],
            [1, 1, 1, 1, 2],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )


def test_rotate_90deg(square_image_grayscale, default_stream):

    op = Rotate(
        theta=90, pivot=[0.5, 0.5], fillMode=FillMode.CONSTANT, stream=default_stream
    )

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(result, expected_result)


def test_rotate_90deg_bilinear(square_image_grayscale, default_stream):

    op = Rotate(
        theta=90,
        pivot=[0.5, 0.5],
        fillMode=FillMode.CONSTANT,
        interpolationMode=InterpolationMode.LINEAR,
        stream=default_stream,
    )

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5],
        ],
        dtype=np.float32,
    )

    assert np.allclose(result, expected_result)


def test_rotate_360deg_bilinear(square_image_grayscale, default_stream):

    op = Rotate(
        theta=360,
        pivot=[0.5, 0.5],
        fillMode=FillMode.CONSTANT,
        interpolationMode=InterpolationMode.LINEAR,
        stream=default_stream,
    )

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = square_image_grayscale

    assert np.allclose(result, expected_result)
