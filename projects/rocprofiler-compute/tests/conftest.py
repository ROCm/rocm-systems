import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--call-binary",
        action="store_true",
        default=False,
        help="Call standalone binary instead of main function during tests",
    )
