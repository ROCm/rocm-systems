# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Import checks shared by the stdlib-only ML API helpers."""

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]

STDLIB_IMPORT_SCRIPT = """
import importlib
import sys
from pathlib import Path

assert sys.dont_write_bytecode, "Isolated import checks must disable bytecode writes"
project_root = Path(sys.argv[1])
module_root = project_root / "src"
if not module_root.is_dir():
    module_root = project_root
sys.path[:0] = [str(project_root / "tests"), str(module_root)]

from conftest import ProfileModeImportGuard

with ProfileModeImportGuard():
    modules_before_package = set(sys.modules)
    importlib.import_module("ml_api_tracing")
    package_imports = set(sys.modules) - modules_before_package
    assert package_imports == {"ml_api_tracing"}, (
        f"Package initializer imported unexpected modules: {sorted(package_imports)}"
    )
    module_name = sys.argv[2]
    module = importlib.import_module(module_name)
    imported_roots = {name.partition(".")[0] for name in sys.modules}
    project_imports = imported_roots & ProfileModeImportGuard.ALLOWED_PROJECT_MODULES
    assert project_imports == {"ml_api_tracing"}, (
        f"{module_name} loaded unexpected project modules: {sorted(project_imports)}"
    )
    expected_path = module_root.joinpath(*module_name.split(".")).with_suffix(".py")
    actual_path = Path(module.__file__).resolve()
    assert actual_path == expected_path.resolve(), (
        f"{module_name} loaded from {actual_path}; expected {expected_path.resolve()}"
    )
"""


def run_stdlib_import(module_name):
    import_process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-S",
            "-c",
            STDLIB_IMPORT_SCRIPT,
            str(PROJECT_ROOT),
            module_name,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=False,
    )
    assert import_process.returncode == 0, import_process.stderr


@pytest.fixture
def assert_stdlib_import():
    return run_stdlib_import
