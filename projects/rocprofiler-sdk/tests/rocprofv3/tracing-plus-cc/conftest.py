#!/usr/bin/env python3

import json
import pytest
import pandas as pd


def pytest_addoption(parser):
    parser.addoption("--input-dir", action="store", help="Path to output dir.")


@pytest.fixture
def input_dir(request):
    dirname = request.config.getoption("--input-dir")
    return dirname
