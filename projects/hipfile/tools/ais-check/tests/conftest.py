"""
Shared test fixtures for the ais-check suite.

The tool ships as an executable named ``ais-check`` (no ``.py`` extension and a
hyphen), so it cannot be imported with a normal ``import`` statement. This shim
loads it from source as a module named ``ais_check`` and exposes it as a
session-scoped fixture.
"""

import importlib.util
import os
from importlib.machinery import SourceFileLoader

import pytest

_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ais-check")


def _load_ais_check():
    # The script has no .py extension, so spec_from_file_location can't infer a
    # loader; supply a source loader explicitly.
    loader = SourceFileLoader("ais_check", _SCRIPT_PATH)
    spec = importlib.util.spec_from_loader("ais_check", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def ais_check():
    """The ais-check script loaded as an importable module."""
    return _load_ais_check()
