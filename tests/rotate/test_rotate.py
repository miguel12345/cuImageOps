import pytest
import numpy as np
from imageOps.core.imageoperation import FillMode
from imageOps.operations.rotate.rotate import Rotate

from imageOps.operations.scale.scale import Scale



@pytest.fixture
def square_image_grayscale():
    return np.array(
        [
        [1,1,1,1,5],
        [1,1,1,1,4],
        [1,1,1,1,3],
        [1,1,1,1,2],
        [1,1,1,1,1]
        ]
        ,dtype=np.float32)

def test_rotate_90deg(square_image_grayscale,default_stream):

    op = Rotate(theta=90,pivot=[0.5,0.5],fillMode=FillMode.CONSTANT,stream=default_stream)

    result = op.run(square_image_grayscale).cpu().numpy()

    expected_result = np.array(
        [
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [1,2,3,4,5]
        ]
        ,dtype=np.float32)

    assert np.array_equal(result,expected_result)
