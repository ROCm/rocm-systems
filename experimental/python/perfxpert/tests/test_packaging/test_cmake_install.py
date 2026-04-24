"""Static checks for the CMake-driven perfxpert install path."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
CMAKELISTS = REPO_ROOT / "CMakeLists.txt"


def test_cmake_install_lets_pip_resolve_dependencies():
    text = CMAKELISTS.read_text(encoding="utf-8")
    assert "--no-deps" not in text


def test_cmake_install_does_not_reference_missing_script():
    text = CMAKELISTS.read_text(encoding="utf-8")
    assert "scripts/perfxpert" not in text
