import pytest
import numpy as np
from cuImageOps.operations import Histogram


@pytest.fixture
def square_image_grayscale():
    return np.array(
        [
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 2, 3, 254, 5],
            [5, 4, 170, 2, 1],
            [1, 2, 3, 4, 5],
        ],
        dtype=np.uint8,
    )


def test_histogram_values_for_small_image(square_image_grayscale, default_stream):

    op = Histogram(stream=default_stream)

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_histogram = np.zeros((255), dtype=np.uint32)
    expected_histogram[1] = 5
    expected_histogram[2] = 5
    expected_histogram[3] = 4
    expected_histogram[4] = 4
    expected_histogram[5] = 5
    expected_histogram[170] = 1
    expected_histogram[254] = 1

    assert np.array_equal(result, expected_histogram)


def test_histogram_values_for_medium_image(square_image_grayscale, default_stream):

    op = Histogram(stream=default_stream)

    # Here we just take the small image and tile it
    num_tiles = 10
    medium_square_image_grayscale = np.tile(
        square_image_grayscale, (num_tiles, num_tiles)
    )

    result = op.run(medium_square_image_grayscale).cpu().numpy()

    expected_histogram = np.zeros((255), dtype=np.uint32)
    expected_histogram[1] = 5 * (num_tiles * num_tiles)
    expected_histogram[2] = 5 * (num_tiles * num_tiles)
    expected_histogram[3] = 4 * (num_tiles * num_tiles)
    expected_histogram[4] = 4 * (num_tiles * num_tiles)
    expected_histogram[5] = 5 * (num_tiles * num_tiles)
    expected_histogram[170] = 1 * (num_tiles * num_tiles)
    expected_histogram[254] = 1 * (num_tiles * num_tiles)

    assert np.array_equal(result, expected_histogram)


def test_histogram_values_for_large_image(square_image_grayscale, default_stream):

    op = Histogram(stream=default_stream)

    # Here we just take the small image and tile it
    num_tiles = 100
    medium_square_image_grayscale = np.tile(
        square_image_grayscale, (num_tiles, num_tiles)
    )

    result = op.run(medium_square_image_grayscale).cpu().numpy()

    expected_histogram = np.zeros((255), dtype=np.uint32)
    expected_histogram[1] = 5 * (num_tiles * num_tiles)
    expected_histogram[2] = 5 * (num_tiles * num_tiles)
    expected_histogram[3] = 4 * (num_tiles * num_tiles)
    expected_histogram[4] = 4 * (num_tiles * num_tiles)
    expected_histogram[5] = 5 * (num_tiles * num_tiles)
    expected_histogram[170] = 1 * (num_tiles * num_tiles)
    expected_histogram[254] = 1 * (num_tiles * num_tiles)

    assert np.array_equal(result, expected_histogram)


def test_histogram_values_for_very_large_image(square_image_grayscale, default_stream):

    op = Histogram(stream=default_stream)

    # Here we just take the small image and tile it
    num_tiles = 1000
    medium_square_image_grayscale = np.tile(
        square_image_grayscale, (num_tiles, num_tiles)
    )

    result = op.run(medium_square_image_grayscale).cpu().numpy()

    expected_histogram = np.zeros((255), dtype=np.uint32)
    expected_histogram[1] = 5 * (num_tiles * num_tiles)
    expected_histogram[2] = 5 * (num_tiles * num_tiles)
    expected_histogram[3] = 4 * (num_tiles * num_tiles)
    expected_histogram[4] = 4 * (num_tiles * num_tiles)
    expected_histogram[5] = 5 * (num_tiles * num_tiles)
    expected_histogram[170] = 1 * (num_tiles * num_tiles)
    expected_histogram[254] = 1 * (num_tiles * num_tiles)

    assert np.array_equal(result, expected_histogram)
