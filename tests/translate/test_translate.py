import pytest
import numpy as np
from cuImageOps.core.imageoperation import FillMode

from cuImageOps.operations.translate.translate import Translate


@pytest.fixture
def square_image_grayscale():
    return np.array(
        [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]], dtype=np.float32
    )


@pytest.fixture
def square_image_rgb():
    return np.array(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        ],
        dtype=np.float32,
    )


def test_translate_basic(square_image_grayscale, default_stream):

    op = Translate(translate=[1, 0], fillMode=FillMode.CONSTANT, stream=default_stream)

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.float32
    )

    assert np.array_equal(result, expected_result)


def test_translate_reflection(square_image_grayscale, default_stream):

    op = Translate(
        translate=[2, 2], fillMode=FillMode.REFLECTION, stream=default_stream
    )

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(result, expected_result)


def test_translate_rgb(square_image_rgb, default_stream):

    op = Translate(translate=[1, 1], fillMode=FillMode.CONSTANT, stream=default_stream)

    result = op.run(square_image_rgb).cpu().numpy()

    expected_result = np.array(
        [
            [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
            [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
        ],
        dtype=np.float32,
    )

    assert np.array_equal(result, expected_result)
