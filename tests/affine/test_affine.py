import pytest
import numpy as np
from cuImageOps.core.imageoperation import FillMode, InterpolationMode
from cuImageOps.operations.affine.affine import Affine
from cuImageOps.operations.rotate.rotate import Rotate
from cuImageOps.operations.scale.scale import Scale


@pytest.fixture
def square_image_grayscale():
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 2, 0],
            [0, 1, 1, 2, 0],
            [0, 1, 1, 2, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )


def test_rotate_and_translate(square_image_grayscale, default_stream):

    op = Affine(
        rotate=90,
        translate=[-1, -1],
        pivot=[0.5, 0.5],
        fillMode=FillMode.CONSTANT,
        stream=default_stream,
    )

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [2, 2, 2, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert np.allclose(result, expected_result)


def test_no_op(square_image_grayscale, default_stream):

    op = Affine(
        rotate=0,
        translate=(0, 0),
        scale=(1, 1),
        pivot=[0.5, 0.5],
        fillMode=FillMode.CONSTANT,
        stream=default_stream,
    )

    result = op.run(square_image_grayscale).cpu().numpy()

    assert np.allclose(result, square_image_grayscale)
