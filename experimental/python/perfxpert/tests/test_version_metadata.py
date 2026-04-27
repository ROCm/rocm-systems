"""Version metadata consistency tests."""

from importlib.metadata import version

import perfxpert


def test_runtime_version_matches_installed_metadata() -> None:
    assert perfxpert.__version__ == version("perfxpert")
