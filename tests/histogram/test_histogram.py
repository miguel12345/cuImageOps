import pytest
import numpy as np
from cuImageOps.operations import Histogram


@pytest.fixture
def square_image_grayscale():
    return np.array(
        [
            [1, 2, 3, 4, 0],
            [5, 4, 3, 2, 1],
            [1, 2, 3, 255, 5],
            [5, 4, 170, 2, 1],
            [1, 2, 3, 4, 5],
        ],
        dtype=np.uint8,
    )


@pytest.fixture
def square_image_rgb():
    return np.array(
        [
            [[1, 120, 90], [2, 7, 90], [3, 5, 1], [4, 120, 111], [5, 66, 4]],
            [[5, 0, 0], [4, 90, 120], [3, 1, 23], [2, 99, 121], [1, 43, 33]],
            [[1, 77, 2], [2, 77, 2], [3, 77, 2], [254, 88, 88], [5, 22, 23]],
            [[5, 1, 1], [4, 4, 4], [170, 170, 170], [2, 2, 2], [1, 1, 1]],
            [[1, 255, 2], [2, 255, 2], [3, 255, 2], [4, 255, 2], [5, 255, 2]],
        ],
        dtype=np.uint8,
    )


def test_histogram_values_for_small_image(square_image_grayscale, default_stream):

    op = Histogram(stream=default_stream)

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_histogram = np.zeros((256, 1), dtype=np.uint32)
    expected_histogram[1, 0] = 5
    expected_histogram[2, 0] = 5
    expected_histogram[3, 0] = 4
    expected_histogram[4, 0] = 4
    expected_histogram[5, 0] = 4
    expected_histogram[0, 0] = 1
    expected_histogram[170, 0] = 1
    expected_histogram[255, 0] = 1

    assert np.array_equal(result, expected_histogram)


def test_histogram_values_for_medium_image(square_image_grayscale, default_stream):

    op = Histogram(stream=default_stream)

    # Here we just take the small image and tile it
    num_tiles = 10
    medium_square_image_grayscale = np.tile(
        square_image_grayscale, (num_tiles, num_tiles)
    )

    result = op.run(medium_square_image_grayscale).cpu().numpy()

    expected_histogram = np.zeros((256, 1), dtype=np.uint32)
    expected_histogram[1, 0] = 5 * (num_tiles * num_tiles)
    expected_histogram[2, 0] = 5 * (num_tiles * num_tiles)
    expected_histogram[3, 0] = 4 * (num_tiles * num_tiles)
    expected_histogram[4, 0] = 4 * (num_tiles * num_tiles)
    expected_histogram[5, 0] = 4 * (num_tiles * num_tiles)
    expected_histogram[170, 0] = 1 * (num_tiles * num_tiles)
    expected_histogram[255, 0] = 1 * (num_tiles * num_tiles)
    expected_histogram[0, 0] = 1 * (num_tiles * num_tiles)

    assert np.array_equal(result, expected_histogram)


def test_histogram_values_for_large_image(square_image_grayscale, default_stream):

    op = Histogram(stream=default_stream)

    # Here we just take the small image and tile it
    num_tiles = 100
    medium_square_image_grayscale = np.tile(
        square_image_grayscale, (num_tiles, num_tiles)
    )

    result = op.run(medium_square_image_grayscale).cpu().numpy()

    expected_histogram = np.zeros((256, 1), dtype=np.uint32)
    expected_histogram[1, 0] = 5 * (num_tiles * num_tiles)
    expected_histogram[2, 0] = 5 * (num_tiles * num_tiles)
    expected_histogram[3, 0] = 4 * (num_tiles * num_tiles)
    expected_histogram[4, 0] = 4 * (num_tiles * num_tiles)
    expected_histogram[5, 0] = 4 * (num_tiles * num_tiles)
    expected_histogram[170, 0] = 1 * (num_tiles * num_tiles)
    expected_histogram[255, 0] = 1 * (num_tiles * num_tiles)
    expected_histogram[0, 0] = 1 * (num_tiles * num_tiles)

    assert np.array_equal(result, expected_histogram)


def test_histogram_values_for_very_large_image(square_image_grayscale, default_stream):

    op = Histogram(stream=default_stream)

    # Here we just take the small image and tile it
    num_tiles = 1000
    medium_square_image_grayscale = np.tile(
        square_image_grayscale, (num_tiles, num_tiles)
    )

    result = op.run(medium_square_image_grayscale).cpu().numpy()

    expected_histogram = np.zeros((256, 1), dtype=np.uint32)
    expected_histogram[1, 0] = 5 * (num_tiles * num_tiles)
    expected_histogram[2, 0] = 5 * (num_tiles * num_tiles)
    expected_histogram[3, 0] = 4 * (num_tiles * num_tiles)
    expected_histogram[4, 0] = 4 * (num_tiles * num_tiles)
    expected_histogram[5, 0] = 4 * (num_tiles * num_tiles)
    expected_histogram[170, 0] = 1 * (num_tiles * num_tiles)
    expected_histogram[255, 0] = 1 * (num_tiles * num_tiles)
    expected_histogram[0, 0] = 1 * (num_tiles * num_tiles)

    assert np.array_equal(result, expected_histogram)


def test_histogram_values_for_small_image_rgb(square_image_rgb, default_stream):

    op = Histogram(stream=default_stream)

    r_histogram = np.histogram(square_image_rgb[:, :, 0], 256, (0, 255))[0]
    g_histogram = np.histogram(square_image_rgb[:, :, 1], 256, (0, 255))[0]
    b_histogram = np.histogram(square_image_rgb[:, :, 2], 256, (0, 255))[0]

    histograms = np.stack([r_histogram, g_histogram, b_histogram], axis=1)

    result = op.run(square_image_rgb).cpu().numpy()

    assert np.array_equal(result, histograms)


def test_histogram_values_for_very_large_image_rgb(square_image_rgb, default_stream):

    op = Histogram(stream=default_stream)

    medium_square_image_rgb = np.tile(square_image_rgb, (1000, 1000, 1))

    r_histogram = np.histogram(medium_square_image_rgb[:, :, 0], 256, (0, 255))[0]
    g_histogram = np.histogram(medium_square_image_rgb[:, :, 1], 256, (0, 255))[0]
    b_histogram = np.histogram(medium_square_image_rgb[:, :, 2], 256, (0, 255))[0]

    histograms = np.stack([r_histogram, g_histogram, b_histogram], axis=1)

    result = op.run(medium_square_image_rgb).cpu().numpy()

    assert np.array_equal(result, histograms)


def test_histogram_values_for_very_large_image_rgb(square_image_rgb, default_stream):

    op = Histogram(stream=default_stream)

    medium_square_image_rgb = np.tile(square_image_rgb, (1000, 1000, 1))

    r_histogram = np.histogram(medium_square_image_rgb[:, :, 0], 256, (0, 255))[0]
    g_histogram = np.histogram(medium_square_image_rgb[:, :, 1], 256, (0, 255))[0]
    b_histogram = np.histogram(medium_square_image_rgb[:, :, 2], 256, (0, 255))[0]

    histograms = np.stack([r_histogram, g_histogram, b_histogram], axis=1)

    result = op.run(medium_square_image_rgb).cpu().numpy()

    assert np.array_equal(result, histograms)
