import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--output-dir", action="store", help="Output directory from rocprofv3"
    )
    parser.addoption("--mode", action="store", help="Test mode: good or bad")
    parser.addoption("--process-type", action="store", help="Process type tested")


@pytest.fixture
def output_dir(request):
    val = request.config.getoption("--output-dir")
    if val is None:
        pytest.skip("--output-dir not provided")
    return val


@pytest.fixture
def mode(request):
    val = request.config.getoption("--mode")
    if val is None:
        pytest.skip("--mode not provided")
    return val


@pytest.fixture
def process_type(request):
    val = request.config.getoption("--process-type")
    if val is None:
        pytest.skip("--process-type not provided")
    return val
