#
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ABI compatibility.
#
# When a user pins amdsmi_interface.py from one ROCm version against a
# libamd_smi.so from another, the contract we promise is:
#   * Common amdsmi_* symbols still in the older .so keep working.
#   * Symbols added in a newer wrapper but missing from the older .so
#     fail cleanly via ctypes' AttributeError -- no silent corruption,
#     no fake success.
#   * If the .so cannot be loaded at all, importing the wrapper still
#     succeeds (degraded mode) so doc/lint tooling keeps working; the
#     first call into a wrapped symbol raises OSError with a diagnostic.
#
# These tests pin those guarantees without needing a real older .so on
# disk. AMDSMI_LIB_OVERRIDE forces the loader to a path of our choosing,
# and ctypes.CDLL is monkey-patched to a FakeCDLL that simulates the
# library surface.

import ast
import ctypes
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_py_interface() -> Path:
    """Directory holding amdsmi_wrapper.py, across source and installed layouts.

    Source tree: ``py-interface/`` under ``projects/amdsmi``. Installed
    package: the ``amdsmi`` package in site-packages -- in CI this test runs
    from the installed tests directory, where ``py-interface/`` is absent and
    the wrapper instead ships inside the importable ``amdsmi`` package.
    """
    candidates = [REPO_ROOT / "py-interface"]
    try:
        import amdsmi

        candidates.append(Path(amdsmi.__file__).resolve().parent)
    except ImportError:
        pass
    for cand in candidates:
        if (cand / "amdsmi_wrapper.py").is_file():
            return cand
    return candidates[0]


PY_INTERFACE = _find_py_interface()


class FakeCDLL:
    """Stand-in for a real ctypes.CDLL bound to libamd_smi.so.

    Exposes only the curated `available_symbols` set; everything else
    raises AttributeError (matching the dlsym-NULL behaviour of a real
    ctypes.CDLL when the .so does not export the requested symbol).
    """

    def __init__(self, path, mode=0, available_symbols=None):
        self._path = path
        self._mode = mode
        self._available = set(available_symbols or ())
        self._cache = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in self._available:
            # Mirrors ctypes' AttributeError when dlsym returns NULL --
            # the wrapper must propagate this cleanly, NOT silently.
            raise AttributeError("%s: undefined symbol: %s" % (self._path, name))
        if name not in self._cache:
            # Return a callable that mimics a ctypes function pointer
            # well enough that the wrapper's `restype = ...` /
            # `argtypes = [...]` assignments don't blow up.
            class _FakeFn:
                __slots__ = ("restype", "argtypes")

                def __init__(self):
                    self.restype = None
                    self.argtypes = []

                def __call__(self, *a, **kw):
                    return 0  # AMDSMI_STATUS_SUCCESS

            self._cache[name] = _FakeFn()
        return self._cache[name]


# A curated stable subset that has shipped in every ROCm 6.x amdsmi.so.
# These are the ABI-stable surface we promise older-.so users.
STABLE_SYMBOLS = {
    "amdsmi_init",
    "amdsmi_shut_down",
    "amdsmi_get_processor_handles",
    "amdsmi_get_socket_handles",
    "amdsmi_get_lib_version",
    "amdsmi_status_code_to_string",
    "amdsmi_free_name_value_pairs",
}


def _scan_unguarded_bindings() -> set:
    """Return amdsmi_* names bound at wrapper module top level, outside any
    try/except.

    These bindings REQUIRE the symbol to exist in the loaded .so (otherwise
    ``import amdsmi_wrapper`` itself raises), so every name here must be in
    STABLE_SYMBOLS or older-.so users cannot import the wrapper at all.

    Parsed with ``ast`` rather than a line regex so reflowing the generated
    wrapper cannot change the result: only assignments that sit directly in
    the module body are inspected; anything nested in an ``ast.Try`` (i.e.
    guarded) is excluded by construction.
    """
    tree = ast.parse((PY_INTERFACE / "amdsmi_wrapper.py").read_text())
    unguarded = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        value = stmt.value
        if (
            isinstance(value, ast.Attribute)
            and value.attr.startswith("amdsmi_")
            and isinstance(value.value, ast.Subscript)
            and isinstance(value.value.value, ast.Name)
            and value.value.value.id == "_libraries"
        ):
            unguarded.add(value.attr)
    return unguarded


def _import_fresh_wrapper() -> ModuleType:
    """(Re)import amdsmi_wrapper.py from PY_INTERFACE, returning the module."""
    if str(PY_INTERFACE) not in sys.path:
        sys.path.insert(0, str(PY_INTERFACE))
    sys.modules.pop("amdsmi_wrapper", None)
    return importlib.import_module("amdsmi_wrapper")


class _Patch:
    """Tiny context manager: patch ctypes.CDLL to a FakeCDLL factory."""

    def __init__(self, available):
        self._available = available
        self._orig = None

    def __enter__(self):
        self._orig = ctypes.CDLL
        available = self._available
        ctypes.CDLL = lambda path, mode=0: FakeCDLL(path, mode, available)
        return self

    def __exit__(self, *exc):
        ctypes.CDLL = self._orig


class AbiCompatTest(unittest.TestCase):
    """Wrapper handles old-.so / missing-symbol scenarios without surprises."""

    def setUp(self):
        # Snapshot env so we never leak AMDSMI_LIB_OVERRIDE into other tests.
        self._saved_env = {k: os.environ.get(k) for k in ("AMDSMI_LIB_OVERRIDE",)}
        os.environ["AMDSMI_LIB_OVERRIDE"] = "/tmp/amdsmi-fake-libamd_smi.so"

    def tearDown(self):
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        sys.modules.pop("amdsmi_wrapper", None)

    def test_override_routes_through_loader(self):
        # AMDSMI_LIB_OVERRIDE must be honoured so the rest of these tests
        # (and ABI-debug workflows in the field) can swap libraries.
        with _Patch(STABLE_SYMBOLS):
            w = _import_fresh_wrapper()
        self.assertEqual(
            w._loaded_lib_path,
            "/tmp/amdsmi-fake-libamd_smi.so",
            "AMDSMI_LIB_OVERRIDE was ignored: %r" % w._loaded_lib_path,
        )

    def test_stable_symbols_resolve_against_older_so(self):
        # Older .so: only the stable subset is exported. Wrapper must
        # successfully bind every stable amdsmi_* it touches at module
        # init -- regression here means wrappers built against newer
        # headers cannot be used against older libraries at all.
        with _Patch(STABLE_SYMBOLS):
            w = _import_fresh_wrapper()
        for sym in STABLE_SYMBOLS:
            self.assertTrue(hasattr(w, sym), "wrapper lost stable symbol %s" % sym)

    def test_unguarded_bindings_are_stable(self):
        # Contract: any amdsmi_* binding the wrapper performs WITHOUT a
        # try/except guard MUST be in STABLE_SYMBOLS, or `import amdsmi`
        # against an older .so will fail at module-import time. This is
        # the real ABI gate; STABLE_SYMBOLS by itself is just a list and
        # cannot catch the case where someone adds a new unguarded binding
        # for a symbol that only exists in newer .so revisions.
        unguarded = _scan_unguarded_bindings()
        unstable = unguarded - STABLE_SYMBOLS
        self.assertFalse(
            unstable,
            "wrapper has unguarded bindings for non-stable symbol(s): %s -- "
            "either wrap them in try/except AttributeError or add to "
            "STABLE_SYMBOLS in this test (and the corresponding ABI promise)." % sorted(unstable),
        )

    def test_missing_symbol_fails_cleanly_not_silently(self):
        # When the .so does NOT export a symbol, ctypes raises
        # AttributeError. The wrapper must surface that, not silently
        # treat the symbol as a no-op or return a fake handle.
        with _Patch(STABLE_SYMBOLS):
            w = _import_fresh_wrapper()
            # Pull the live FakeCDLL out of the wrapper.
            fake = w._libraries["libamd_smi.so"]
            self.assertIsInstance(fake, FakeCDLL)
            with self.assertRaises(AttributeError):
                fake.amdsmi_some_symbol_that_was_added_in_a_future_release()

    def test_degraded_import_when_library_unavailable(self):
        # If ctypes.CDLL outright fails (no library at all), importing
        # the wrapper must STILL succeed -- doc/lint tooling depends on
        # this. _loaded_lib_path is None and _libraries['libamd_smi.so']
        # is the diagnostic sentinel.
        orig = ctypes.CDLL

        def _fail(path, mode=0):
            raise OSError("no such file: %s" % path)

        ctypes.CDLL = _fail
        try:
            w = _import_fresh_wrapper()
        finally:
            ctypes.CDLL = orig
        self.assertIsNone(w._loaded_lib_path, "_loaded_lib_path must be None on load failure")
        sentinel = w._libraries["libamd_smi.so"]
        self.assertIsInstance(
            sentinel, w._MissingLibrary, "load failure must yield _MissingLibrary sentinel"
        )
        # Calling any wrapped function on the sentinel raises OSError
        # with a diagnostic, not a silent no-op.
        with self.assertRaises(OSError):
            sentinel.amdsmi_init()
        # And the sentinel honestly reports unknown attrs as AttributeError
        # (no truthy claim of arbitrary symbol presence).
        with self.assertRaises(AttributeError):
            getattr(sentinel, "definitely_not_an_amdsmi_function")


WRAPPER_SRC = PY_INTERFACE / "amdsmi_wrapper.py"
DISABLE_SYSTEM_FALLBACK_TOOL = REPO_ROOT / "tools" / "disable_system_fallback.py"


def _import_wrapper_from(dir_path: str, module_name: str) -> ModuleType:
    """Import a copied amdsmi_wrapper.py from dir_path under a unique name."""
    wrapper = Path(dir_path) / "amdsmi_wrapper.py"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, wrapper)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WheelLoaderContractTest(unittest.TestCase):
    """The pip-wheel loader contract: prefer the bundled libamd_smi_python.so
    (case 2), and refuse to fall back to a system library when the bundle is
    missing -- the property that stops a wheel from silently loading an
    unrelated system libamd_smi.so (e.g. PyTorch's)."""

    def setUp(self):
        self._saved_override = os.environ.pop("AMDSMI_LIB_OVERRIDE", None)
        self._tmp = Path(tempfile.mkdtemp(prefix="amdsmi-loader-"))
        if WRAPPER_SRC.is_file():
            shutil.copy(WRAPPER_SRC, self._tmp / "amdsmi_wrapper.py")
        self._modname = "amdsmi_wrapper_contract_%d" % id(self)

    def tearDown(self):
        if self._saved_override is not None:
            os.environ["AMDSMI_LIB_OVERRIDE"] = self._saved_override
        sys.modules.pop(self._modname, None)
        shutil.rmtree(self._tmp, ignore_errors=True)

    @unittest.skipUnless(WRAPPER_SRC.is_file(), "amdsmi_wrapper.py not found")
    def test_bundled_python_so_is_preferred(self):
        bundled = self._tmp / "libamd_smi_python.so"
        bundled.write_bytes(b"")  # presence is all the loader checks
        attempted = []
        orig = ctypes.CDLL

        def _record(path, mode=0):
            attempted.append(str(path))
            return FakeCDLL(path, mode, STABLE_SYMBOLS)

        ctypes.CDLL = _record
        try:
            mod = _import_wrapper_from(self._tmp, self._modname)
        finally:
            ctypes.CDLL = orig
        self.assertEqual(mod._loaded_lib_path, str(bundled))
        self.assertTrue(attempted, "loader never called ctypes.CDLL")
        self.assertEqual(attempted[0], str(bundled), "bundled .so was not the first load attempt")

    @unittest.skipUnless(
        WRAPPER_SRC.is_file() and DISABLE_SYSTEM_FALLBACK_TOOL.is_file(),
        "wrapper or disable_system_fallback.py not found (installed layout)",
    )
    def test_wheel_refuses_system_fallback_when_bundle_missing(self):
        wrapper = self._tmp / "amdsmi_wrapper.py"
        subprocess.check_call([sys.executable, str(DISABLE_SYSTEM_FALLBACK_TOOL), str(wrapper)])
        mod = _import_wrapper_from(self._tmp, self._modname)
        self.assertFalse(mod._AMDSMI_ALLOW_SYSTEM_FALLBACK)
        # No bundled .so beside the wrapper and no override: the loader must
        # raise rather than load a system libamd_smi.so.
        with self.assertRaises(OSError) as ctx:
            mod._load_library()
        self.assertIn("refusing to fall back", str(ctx.exception))


class DisableSystemFallbackToolTest(unittest.TestCase):
    """tools/disable_system_fallback.py flips the loader flag exactly once."""

    @unittest.skipUnless(
        WRAPPER_SRC.is_file() and DISABLE_SYSTEM_FALLBACK_TOOL.is_file(),
        "wrapper or disable_system_fallback.py not found (installed layout)",
    )
    def test_flag_flipped_and_double_run_guarded(self):
        tmp = Path(tempfile.mkdtemp(prefix="amdsmi-disable-"))
        try:
            wrapper = tmp / "amdsmi_wrapper.py"
            shutil.copy(WRAPPER_SRC, wrapper)
            self.assertIn("_AMDSMI_ALLOW_SYSTEM_FALLBACK = True", wrapper.read_text())
            subprocess.check_call([sys.executable, str(DISABLE_SYSTEM_FALLBACK_TOOL), str(wrapper)])
            patched = wrapper.read_text()
            self.assertIn("_AMDSMI_ALLOW_SYSTEM_FALLBACK = False", patched)
            self.assertNotIn("_AMDSMI_ALLOW_SYSTEM_FALLBACK = True", patched)
            # Anchor is gone now -> a second run must fail (count != 1).
            rc = subprocess.call([sys.executable, str(DISABLE_SYSTEM_FALLBACK_TOOL), str(wrapper)])
            self.assertNotEqual(rc, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
