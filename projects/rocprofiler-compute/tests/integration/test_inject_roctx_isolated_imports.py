# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Fresh-interpreter checks of the ML API tracing recording imports. No GPU."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from profile_import_utils import PROFILE_IMPORT_SETUP, PROJECT_TESTS_ROOT

from ml_api_tracing.inject_roctx import installer
from ml_api_tracing.inject_roctx.backends import torch_trace_collector as loader

CANONICAL_RECORDING_PROBE = """
from typing import get_type_hints
package_name = 'ml_api_tracing.inject_roctx.backends'
if sys.argv[3] == 'ordinary':
    __import__(package_name, fromlist=('*',))
else:
    importlib.import_module(package_name)
assert {name for name in sys.modules if name.startswith('ml_api_tracing')} == {
    'ml_api_tracing',
    'ml_api_tracing.inject_roctx',
    'ml_api_tracing.inject_roctx.backends',
}
modules = {}
for name in ('backend_contract', 'marker_ranges', 'installer',
             'backends.torch_trace_collector', 'backends.torch', 'backends.triton'):
    module_name = 'ml_api_tracing.inject_roctx.' + name
    assert module_name not in sys.modules, module_name
    if sys.argv[3] == 'ordinary':
        module = __import__(module_name, fromlist=('*',))
    else:
        module = importlib.import_module(module_name)
    modules[name] = module
backend_contract = modules['backend_contract']
marker_ranges = modules['marker_ranges']
torch = modules['backends.torch']
triton = modules['backends.triton']
torch_trace_collector = modules['backends.torch_trace_collector']
assert torch.marker_ranges is marker_ranges and triton.marker_ranges is marker_ranges
assert torch.torch_trace_collector is torch_trace_collector
assert isinstance(backend_contract.get_backend('torch'), torch.TorchBackend)
assert isinstance(backend_contract.get_backend('triton'), triton.TritonBackend)
assert backend_contract.get_backend('missing') is None
for module in modules.values():
    assert importlib.import_module(module.__name__) is module
    for value in vars(module).values():
        if callable(value) and getattr(value, '__module__', None) == module.__name__:
            get_type_hints(value)
assert_project_origins()
assert not forbidden_attempts, forbidden_attempts
print('canonical state and optional imports preserved')
"""


@pytest.mark.parametrize("import_mode", ["ordinary", "dynamic"])
def test_recording_uses_canonical_state_without_framework_imports(
    tmp_path, import_mode
):
    python_root = Path(installer.__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-I",
            "-c",
            PROFILE_IMPORT_SETUP + CANONICAL_RECORDING_PROBE,
            str(python_root),
            str(PROJECT_TESTS_ROOT),
            import_mode,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
        check=True,
    )
    assert completed.stdout.strip() == "canonical state and optional imports preserved"


@pytest.fixture
def relocated_runtime(tmp_path):
    source_root = Path(loader.__file__).resolve().parents[3]
    runtime_root = tmp_path / "share" / "rocprofiler-compute"
    for relative_path in (
        "config.py",
        "utils/__init__.py",
        "utils/logger.py",
        "utils/utils_common.py",
        "utils/native_tool_finder.py",
        "ml_api_tracing/__init__.py",
        "ml_api_tracing/inject_roctx/__init__.py",
        "ml_api_tracing/inject_roctx/backends/__init__.py",
        "ml_api_tracing/inject_roctx/backends/torch_trace_collector.py",
    ):
        destination = runtime_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_root / relative_path, destination)
    shutil.copytree(
        source_root / "vendored",
        runtime_root / "vendored",
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    return runtime_root


@pytest.mark.parametrize("libdir", ["lib", "lib64"])
def test_find_collector_infers_actual_relocated_root(
    relocated_runtime, tmp_path, libdir
):
    artifact = tmp_path / libdir / "rocprofiler-compute" / "torch_trace_collector.so"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"artifact lookup only")
    probe = (
        "import json, sys\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "from ml_api_tracing.inject_roctx.backends import "
        "torch_trace_collector as loader\n"
        "print(json.dumps([str(loader._PACKAGE_ROOT), loader.__file__, "
        "loader.find_prebuilt_artifacts.__module__, str(loader._find_collector())]))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-I", "-c", probe, str(relocated_runtime)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(completed.stdout) == [
        str(relocated_runtime),
        str(
            relocated_runtime
            / "ml_api_tracing/inject_roctx/backends/torch_trace_collector.py"
        ),
        "utils.native_tool_finder",
        str(artifact),
    ]
