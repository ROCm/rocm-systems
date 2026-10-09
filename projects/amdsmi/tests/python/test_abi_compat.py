# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
import builtins
import ctypes
import importlib
import importlib.util
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import unittest
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def _find_py_interface() -> Path:
    """Directory holding amdsmi_wrapper.py, across source and installed layouts.

    Resolution order is deliberate: the wheel-loader tests copy this wrapper
    expecting the committed system variant (``_AMDSMI_ALLOW_SYSTEM_FALLBACK =
    True``), so it must NOT resolve to a coexisting pip wheel's wrapper (which
    ships the flag flipped to ``False``).

      1. ``py-interface/`` in the source checkout.
      2. The co-installed ``share/amd_smi/amdsmi`` copy (installed alongside the
         tests, always the system variant, and immune to a wheel shadowing
         ``import amdsmi`` on sys.path).
      3. Only then the imported ``amdsmi`` package, as a last resort.
    """
    candidates = [REPO_ROOT / "py-interface", REPO_ROOT / "amdsmi"]
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


class _LibOverrideEnvMixin:
    """Pin AMDSMI_LIB_OVERRIDE for the test, then restore the prior env.

    Snapshots the variable so a test can never leak it into the rest of the
    suite, and drops the cached wrapper module so the next import re-runs the
    loader.
    """

    OVERRIDE_PATH = "/tmp/amdsmi-fake-libamd_smi.so"

    def setUp(self):
        self._saved_override = os.environ.get("AMDSMI_LIB_OVERRIDE")
        os.environ["AMDSMI_LIB_OVERRIDE"] = self.OVERRIDE_PATH

    def tearDown(self):
        if self._saved_override is None:
            os.environ.pop("AMDSMI_LIB_OVERRIDE", None)
        else:
            os.environ["AMDSMI_LIB_OVERRIDE"] = self._saved_override
        sys.modules.pop("amdsmi_wrapper", None)


class AbiCompatTest(_LibOverrideEnvMixin, unittest.TestCase):
    """Wrapper handles old-.so / missing-symbol scenarios without surprises."""

    def test_override_routes_through_loader(self):
        # AMDSMI_LIB_OVERRIDE must be honoured so the rest of these tests
        # (and ABI-debug workflows in the field) can swap libraries.
        with _Patch(STABLE_SYMBOLS):
            w = _import_fresh_wrapper()
        self.assertEqual(
            w._loaded_lib_path,
            self.OVERRIDE_PATH,
            "AMDSMI_LIB_OVERRIDE was ignored: %r" % w._loaded_lib_path,
        )

    def test_stable_symbols_resolve_against_older_so(self):
        # Simulate a user with an OLDER/mismatched .so loaded: it exports only
        # the stable subset, none of the newer symbols. The contract is:
        #   * every common (stable) symbol the user requests still works, and
        #   * a symbol the older .so does not export fails cleanly, not silently.
        with _Patch(STABLE_SYMBOLS):
            w = _import_fresh_wrapper()
        lib = w._libraries["libamd_smi.so"]
        self.assertIsInstance(lib, FakeCDLL)

        # Common symbols resolve and are callable against the older .so.
        for sym in STABLE_SYMBOLS:
            self.assertTrue(hasattr(w, sym), "wrapper lost stable symbol %s" % sym)
            fn = getattr(lib, sym)
            self.assertEqual(fn(), 0, "stable symbol %s not callable against older .so" % sym)

        # A newer symbol the older .so lacks must raise, not return a no-op.
        with self.assertRaises(AttributeError):
            lib.amdsmi_get_gpu_a_future_only_symbol()

    def test_unguarded_bindings_are_stable(self):
        # Contract: any amdsmi_* binding the wrapper performs WITHOUT a
        # try/except guard MUST be in STABLE_SYMBOLS, or `import amdsmi`
        # against an older .so will fail at module-import time. This is
        # the real ABI gate; STABLE_SYMBOLS by itself is just a list and
        # cannot catch the case where someone adds a new unguarded binding
        # for a symbol that only exists in newer .so revisions.
        unguarded = _scan_unguarded_bindings()
        # Guard against the scan itself silently matching nothing (e.g. the
        # generator renames the "_libraries" binding): an empty result would
        # trivially satisfy the assertion below and turn this gate into a no-op.
        self.assertTrue(
            unguarded,
            "scan found no unguarded amdsmi_* bindings -- the wrapper layout "
            "changed and _scan_unguarded_bindings() needs updating, otherwise "
            "this ABI gate no longer checks anything.",
        )
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

    @unittest.skipUnless(WRAPPER_SRC.is_file(), "amdsmi_wrapper.py not found")
    def test_relocatable_unloadable_lib_falls_through_to_system(self):
        # TheRock relocatable layout: wrapper at <root>/share/amd_smi/amdsmi,
        # library at <root>/lib. A present-but-unloadable relocatable .so
        # (missing deps -> OSError) must fall through to the system SONAME
        # rather than shadow it.
        import re as _re

        soname = _re.search(r'_AMDSMI_LIB_SONAME = "([^"]+)"', WRAPPER_SRC.read_text()).group(1)
        pkg = self._tmp / "share" / "amd_smi" / "amdsmi"
        pkg.mkdir(parents=True)
        shutil.copy(WRAPPER_SRC, pkg / "amdsmi_wrapper.py")
        reloc = self._tmp / "lib" / soname
        reloc.parent.mkdir()
        reloc.write_bytes(b"")  # present, but CDLL will raise below
        attempted = []
        orig = ctypes.CDLL

        def _cdll(path, mode=0):
            attempted.append(str(path))
            if str(path) == str(reloc):
                raise OSError("simulated missing dependency")
            return FakeCDLL(path, mode, STABLE_SYMBOLS)

        ctypes.CDLL = _cdll
        try:
            mod = _import_wrapper_from(str(pkg), self._modname)
        finally:
            ctypes.CDLL = orig
        # the relocatable path was tried, then the loader fell through to the
        # bare system SONAME instead of returning a broken/_MissingLibrary.
        self.assertIn(str(reloc), attempted)
        self.assertEqual(mod._loaded_lib_path, mod._AMDSMI_LIB_SONAME)


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
            # Anchor is gone now -> a second run is a no-op (idempotent), so a
            # rebuild that reuses the already-flipped staged wrapper succeeds.
            rc = subprocess.call([sys.executable, str(DISABLE_SYSTEM_FALLBACK_TOOL), str(wrapper)])
            self.assertEqual(rc, 0)
            self.assertIn("_AMDSMI_ALLOW_SYSTEM_FALLBACK = False", wrapper.read_text())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    @unittest.skipUnless(
        WRAPPER_SRC.is_file() and DISABLE_SYSTEM_FALLBACK_TOOL.is_file(),
        "wrapper or disable_system_fallback.py not found (installed layout)",
    )
    def test_ambiguous_wrapper_fails_loud(self):
        # Neither the True anchor nor exactly one False replacement present:
        # the tool must refuse (non-zero exit) rather than stage a wheel with
        # an ambiguous loader-fallback flag.
        tmp = Path(tempfile.mkdtemp(prefix="amdsmi-disable-"))
        try:
            wrapper = tmp / "amdsmi_wrapper.py"
            text = WRAPPER_SRC.read_text().replace(
                "_AMDSMI_ALLOW_SYSTEM_FALLBACK = True", "_AMDSMI_ALLOW_SYSTEM_FALLBACK_UNSET = True"
            )
            wrapper.write_text(text)
            rc = subprocess.call([sys.executable, str(DISABLE_SYSTEM_FALLBACK_TOOL), str(wrapper)])
            self.assertNotEqual(rc, 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def _stdlib_top_level_names() -> frozenset:
    """Top-level stdlib module names, computed without sys.stdlib_module_names.

    That attribute is 3.10+, but pyproject declares ``requires-python >=3.6``
    and CI runs a 3.6-3.14 matrix, so deriving the set from ``sysconfig`` keeps
    the guard test running on every leg instead of skipping the older half.
    """
    names = set(sys.builtin_module_names)
    stdlib = sysconfig.get_paths().get("stdlib")
    if stdlib:
        for base in (stdlib, os.path.join(stdlib, "lib-dynload")):
            try:
                entries = os.listdir(base)
            except OSError:
                continue
            for entry in entries:
                top = entry.split(".")[0]
                if top.isidentifier():
                    names.add(top)
    return frozenset(names)


STDLIB_TOP_LEVEL = _stdlib_top_level_names()


class _ProfileModeLikeGuard:
    """Minimal clone of rocprofiler-compute's profile-mode import guard.

    Allows stdlib plus a small ROCm whitelist; every other top-level package
    raises ImportError (the parent class, not ModuleNotFoundError) from both
    the meta_path finder and the builtins.__import__ hook. rocm_sdk is
    deliberately left off the whitelist so a stray third-party import in the
    loader is caught here.
    """

    _ALLOWED = frozenset({"amdsmi", "amdsmi_wrapper", "amdsmi_interface", "hip", "rocprofv3"})

    def __enter__(self):
        sys.meta_path.insert(0, self)
        self._real_import = builtins.__import__
        builtins.__import__ = self._guarded_import
        return self

    def __exit__(self, *exc):
        builtins.__import__ = self._real_import
        if self in sys.meta_path:
            sys.meta_path.remove(self)

    def _forbid(self, fullname):
        top = fullname.split(".")[0]
        if top in STDLIB_TOP_LEVEL or top in self._ALLOWED:
            return
        raise ImportError("forbidden package in profile mode: %s" % top)

    def find_spec(self, fullname, path, target=None):
        self._forbid(fullname)
        return None

    def _guarded_import(self, name, *args, **kwargs):
        # An already-imported module needs no load path, so re-importing it
        # must not trip the guard: the window is process-wide and any tracer,
        # plugin or background thread may re-import something benign.
        if name not in sys.modules:
            level = args[3] if len(args) > 3 else kwargs.get("level", 0)
            if level == 0:
                self._forbid(name)
        return self._real_import(name, *args, **kwargs)


class RestrictiveImportGuardTest(_LibOverrideEnvMixin, unittest.TestCase):
    """Importing the wrapper must survive a hostile import environment.

    Reproduces the downstream break where the loader did ``import rocm_sdk`` in
    its search path but caught only ``(ModuleNotFoundError, FileNotFoundError)``;
    a profile-mode guard raised a bare ImportError that escaped and aborted
    ``import amdsmi``. rocm_sdk is kept off the whitelist so such an import is
    rejected here rather than in downstream TheRock CI.

    Coverage is split deliberately. AMDSMI_LIB_OVERRIDE short-circuits
    ``_load_library()`` at step 1, so an override-based test only exercises
    module-level statements and step 1 -- it can never reach step 3, the
    relocatable-ROCm-tree branch, which is exactly where a rocm_sdk import
    would live (rocm_sdk *is* TheRock's relocatable-tree package). The second
    test therefore runs with no override so the loader walks steps 2-4.
    """

    def test_module_scope_survives_restrictive_import_guard(self):
        # Scope: module-level statements plus _load_library() step 1 only.
        with _ProfileModeLikeGuard(), _Patch(STABLE_SYMBOLS):
            w = _import_fresh_wrapper()
        self.assertEqual(
            w._loaded_lib_path,
            self.OVERRIDE_PATH,
            "wrapper module scope or the AMDSMI_LIB_OVERRIDE branch failed "
            "under a restrictive import guard -- an unguarded third-party "
            "import likely crept in above the loader",
        )

    @unittest.skipUnless(WRAPPER_SRC.is_file(), "amdsmi_wrapper.py not found")
    def test_relocatable_branch_survives_restrictive_import_guard(self):
        # TheRock relocatable layout: wrapper at <root>/share/amd_smi/amdsmi,
        # library at <root>/lib. With no override the loader walks steps 2-4
        # and must resolve the relocatable .so under the guard.
        import re as _re

        del os.environ["AMDSMI_LIB_OVERRIDE"]
        soname = _re.search(r'_AMDSMI_LIB_SONAME = "([^"]+)"', WRAPPER_SRC.read_text()).group(1)
        tmp = Path(tempfile.mkdtemp(prefix="amdsmi-guard-"))
        modname = "amdsmi_wrapper_guard_%d" % id(self)
        try:
            pkg = tmp / "share" / "amd_smi" / "amdsmi"
            pkg.mkdir(parents=True)
            shutil.copy(WRAPPER_SRC, pkg / "amdsmi_wrapper.py")
            reloc = tmp / "lib" / soname
            reloc.parent.mkdir()
            reloc.write_bytes(b"")  # presence is all the loader checks
            with _ProfileModeLikeGuard(), _Patch(STABLE_SYMBOLS):
                mod = _import_wrapper_from(str(pkg), modname)
            # Resolving to the relocatable path (not the bare SONAME, not the
            # _MissingLibrary sentinel) proves step 3 both ran and completed.
            self.assertEqual(
                mod._loaded_lib_path,
                str(reloc),
                "the relocatable-tree branch of _load_library() did not resolve "
                "under a restrictive import guard -- an unguarded third-party "
                "import likely crept back into the load path",
            )
        finally:
            sys.modules.pop(modname, None)
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
