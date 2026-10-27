# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Import checks shared by the stdlib-only ML API helpers."""

import subprocess
import sys
from functools import partial

import pytest

STDLIB_IMPORT_SCRIPT = """
import importlib
import sys
from pathlib import Path

project_root = Path(sys.argv[1]).resolve().parents[3]
module_root = project_root / "src"
if not module_root.is_dir():
    module_root = project_root
sys.path[:0] = [str(project_root / "tests"), str(module_root)]

from conftest import ProfileModeImportGuard

with ProfileModeImportGuard():
    modules_before_package = set(sys.modules)
    importlib.import_module("ml_api_tracing")
    assert set(sys.modules) - modules_before_package == {"ml_api_tracing"}
    module_name = sys.argv[2]
    module = importlib.import_module(module_name)
    imported_roots = {name.partition(".")[0] for name in sys.modules}
    assert imported_roots & ProfileModeImportGuard.ALLOWED_PROJECT_MODULES == {
        "ml_api_tracing"
    }
    expected_path = module_root.joinpath(*module_name.split(".")).with_suffix(".py")
    assert Path(module.__file__).resolve() == expected_path.resolve()
"""


def run_stdlib_import(test_file, module_name):
    import_process = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            STDLIB_IMPORT_SCRIPT,
            str(test_file),
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
def assert_stdlib_import(request):
    return partial(run_stdlib_import, request.path)
