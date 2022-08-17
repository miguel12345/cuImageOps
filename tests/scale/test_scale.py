import pytest
import numpy as np
from imageOps.core.imageoperation import FillMode

from imageOps.operations.scale.scale import Scale



@pytest.fixture
def square_image_grayscale():
    return np.array(
        [
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1]
        ]
        ,dtype=np.float32)

def test_scale_down_center_pivot(square_image_grayscale,default_stream):

    op = Scale(scale=[0.5,0.5],pivot=[0.5,0.5],fillMode=FillMode.CONSTANT,stream=default_stream)

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [
        [0,0,0,0,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,1,1,1,0],
        [0,0,0,0,0]
        ]
        ,dtype=np.float32)

    assert np.array_equal(result,expected_result)

def test_scale_down_tl_pivot(square_image_grayscale,default_stream):

    op = Scale(scale=[0.5,0.5],pivot=[0.0,0.0],fillMode=FillMode.CONSTANT,stream=default_stream)

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [
        [1,1,1,0,0],
        [1,1,1,0,0],
        [1,1,1,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0]
        ]
        ,dtype=np.float32)

    assert np.array_equal(result,expected_result)

def test_scale_down_br_pivot(square_image_grayscale,default_stream):

    op = Scale(scale=[0.5,0.5],pivot=[1.0,1.0],fillMode=FillMode.CONSTANT,stream=default_stream)

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,1,1,1],
        [0,0,1,1,1],
        [0,0,1,1,1]
        ]
        ,dtype=np.float32)

    assert np.array_equal(result,expected_result)
