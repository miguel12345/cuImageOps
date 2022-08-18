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

@pytest.fixture
def square_image_grayscale_asymmetric():
    return np.array(
        [
        [1,2,3,4,5],
        [5,4,3,2,1],
        [1,2,3,4,5],
        [1,2,3,4,5],
        [5,4,3,2,1],
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

def test_scale_hflip(square_image_grayscale_asymmetric,default_stream):

    op = Scale(scale=[-1.0,1.0],pivot=[0.5,0.5],fillMode=FillMode.CONSTANT,stream=default_stream)

    result = op.run(square_image_grayscale_asymmetric).cpu().numpy()

    expected_result = np.array(
        [
        [5,4,3,2,1],
        [1,2,3,4,5],
        [5,4,3,2,1],
        [5,4,3,2,1],
        [1,2,3,4,5]
        ]
        ,dtype=np.float32)

    assert np.array_equal(result,expected_result)

def test_scale_vflip(square_image_grayscale_asymmetric,default_stream):

    op = Scale(scale=[1.0,-1.0],pivot=[0.5,0.5],fillMode=FillMode.CONSTANT,stream=default_stream)

    result = op.run(square_image_grayscale_asymmetric).cpu().numpy()

    expected_result = np.array(
        [
        [5,4,3,2,1],
        [1,2,3,4,5],
        [1,2,3,4,5],
        [5,4,3,2,1],
        [1,2,3,4,5],
        ]
        ,dtype=np.float32)

    assert np.array_equal(result,expected_result)
