from os.path import abspath, dirname
import pytest
import sys

repo_path = abspath(dirname(dirname(__file__)))
sys.path.insert(1, repo_path)


from cuImageOps.core.cuda.stream import CudaStream

@pytest.fixture(scope="session")
def default_stream():
    return CudaStream()