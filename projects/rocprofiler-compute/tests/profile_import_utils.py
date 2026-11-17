# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Run a fresh Python that imports a module under the profile-mode import limits.

Used by the unit and the integration tests of ML API tracing.
"""

import subprocess
import sys
from pathlib import Path

PROJECT_TESTS_ROOT = Path(__file__).resolve().parent

PROFILE_IMPORT_SETUP = """
import builtins
import importlib
import importlib.util
import sys
from pathlib import Path

assert sys.dont_write_bytecode, "Isolated import checks must disable bytecode writes"
module_root = Path(sys.argv[1]).resolve()
sys.path[:0] = [sys.argv[2], str(module_root)]

from conftest import ProfileModeImportGuard

forbidden_prefixes = {
    "torch", "triton", "pandas", "numpy", "rich", "tabulate", "plotly",
    "dash", "sqlalchemy", "yaml", "utils.analysis_orm", "utils.utils_analysis",
    "ml_api_tracing.analyze",
}
forbidden_attempts = []
profile_policy = ProfileModeImportGuard()
real_import = builtins.__import__

def check_import(name, hook):
    # Vendored PyYAML probes this unavailable optional extension before its
    # pure-Python fallback. The base policy still denies that extension.
    if name != "yaml._yaml" and any(
        name == prefix or name.startswith(prefix + ".")
        for prefix in forbidden_prefixes
    ):
        forbidden_attempts.append((hook, name))
        raise ImportError("Forbidden profile dependency: " + name)
    if sys.version_info >= (3, 10):
        profile_policy._raise_if_forbidden(name)

def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    resolved = name
    if level:
        resolved = importlib.util.resolve_name(
            "." * level + name, globals["__package__"]
        )
    check_import(resolved, "builtin")
    for child in fromlist or ():
        if child != "*":
            check_import(resolved + "." + child, "builtin")
    return real_import(name, globals, locals, fromlist, level)

class ImportDenialFinder:
    def find_spec(self, fullname, path, target=None):
        check_import(fullname, "meta_path")
        return None

def assert_project_origins():
    project_roots = {"ml_api_tracing", "utils", "config", "vendored"}
    for name, module in tuple(sys.modules.items()):
        if name.partition(".")[0] not in project_roots:
            continue
        candidate = module_root.joinpath(*name.split("."))
        expected = (
            candidate / "__init__.py"
            if candidate.is_dir() else candidate.with_suffix(".py")
        )
        if expected.is_file():
            assert Path(module.__file__).resolve() == expected.resolve(), name
            assert Path(module.__spec__.origin).resolve() == expected.resolve(), name
        else:
            assert module.__spec__.origin is None, name
            assert list(module.__path__), name
            for path in module.__path__:
                Path(path).resolve().relative_to(module_root)

builtins.__import__ = guarded_import
sys.meta_path.insert(0, ImportDenialFinder())
# Prove both hooks deny imports before checking the actual feature imports.
for import_dependency in (
    lambda: __import__("", {"__package__": "ml_api_tracing"}, None, ("analyze",), 1),
    lambda: importlib.import_module("triton"),
):
    try:
        import_dependency()
    except ImportError:
        pass
    else:
        raise AssertionError("A forbidden import escaped the denial hooks")
assert forbidden_attempts == [
    ("builtin", "ml_api_tracing.analyze"), ("meta_path", "triton")
], forbidden_attempts
forbidden_attempts.clear()
"""

PROFILE_IMPORT_SCRIPT = """
module_name = sys.argv[4]
assert module_name not in sys.modules
if sys.argv[3] == "ordinary":
    module = __import__(module_name, fromlist=("*",))
else:
    module = importlib.import_module(module_name)
assert module is sys.modules[module_name]
assert_project_origins()
assert not forbidden_attempts, forbidden_attempts
print("profile import boundaries preserved")
"""


def run_profile_import(module_name, module_root):
    for import_mode in ("ordinary", "dynamic"):
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                "-I",
                "-c",
                PROFILE_IMPORT_SETUP + PROFILE_IMPORT_SCRIPT,
                str(module_root),
                str(PROJECT_TESTS_ROOT),
                import_mode,
                module_name,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=30,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        assert completed.stdout.strip() == "profile import boundaries preserved"
