"""Tests for hip_runtime_supports_ais()."""

# The FakeHip double mirrors the HIP C API: its method names must match the
# camelCase symbols ais-check resolves, and it writes back through ctypes
# byref() pointers via the _obj member (the only way to emulate an out-param).
# Both are deliberate, so silence the corresponding pylint checks here.
# pylint: disable=missing-function-docstring,invalid-name,protected-access
# pylint: disable=too-many-instance-attributes,too-many-arguments,too-few-public-methods
# pylint: disable=redefined-outer-name,unused-argument

import pytest

_AIS_SYMBOLS = frozenset({b"hipAmdFileWrite", b"hipAmdFileRead"})


class _Method:
    """Callable that tolerates argtypes/restype attribute assignment."""

    def __init__(self, fn):
        self._fn = fn
        self.argtypes = None
        self.restype = None

    def __call__(self, *args):
        return self._fn(*args)


class FakeHip:
    """
    Stand-in for a ctypes.CDLL handle. Only the attributes a given test wants
    are defined; anything missing raises AttributeError, matching a HIP runtime
    too old to export a symbol. The fake methods accept argtypes/restype
    assignment (as ctypes _FuncPtr objects do) and ignore it.
    """

    def __init__(
        self,
        *,
        version=60000000,
        version_err=0,
        present_symbols=_AIS_SYMBOLS,
        proc_err=0,
        proc_status=0,
        null_ptr=False,
        omit_get_proc_address=False,
        omit_runtime_version=False
    ):
        self._version = version
        self._version_err = version_err
        self._present = set(present_symbols)
        self._proc_err = proc_err
        self._proc_status = proc_status
        self._null_ptr = null_ptr

        if not omit_runtime_version:
            self.hipRuntimeGetVersion = _Method(self._runtime_get_version)
        self.hipGetErrorString = _Method(self._get_error_string)
        if not omit_get_proc_address:
            self.hipGetProcAddress = _Method(self._get_proc_address)

    def _runtime_get_version(self, version_ptr):
        version_ptr._obj.value = self._version
        return self._version_err

    def _get_error_string(self, _err):
        return b"fake error"

    def _get_proc_address(self, symbol, func_ptr, _version, _flags, status_ptr):
        status_ptr._obj.value = self._proc_status
        if symbol in self._present and not self._null_ptr:
            func_ptr._obj.value = 0xDEAD0000
        else:
            func_ptr._obj.value = None
        return self._proc_err


@pytest.fixture
def patch_runtimes(monkeypatch, ais_check):
    """Helper to stub find_hip_runtimes() and ctypes.CDLL together."""

    def install(handles):
        # handles: dict of path -> FakeHip (or an exception instance to raise)
        monkeypatch.setattr(
            ais_check, "find_hip_runtimes", lambda: dict.fromkeys(handles, False)
        )

        def fake_cdll(path, *_args, **_kwargs):
            obj = handles[path]
            if isinstance(obj, Exception):
                raise obj
            return obj

        monkeypatch.setattr(ais_check.ctypes, "CDLL", fake_cdll)

    return install


def test_library_with_both_symbols_supported(patch_runtimes, ais_check):
    patch_runtimes({"/opt/rocm/lib/libamdhip64.so": FakeHip()})

    assert ais_check.hip_runtime_supports_ais() == {
        "/opt/rocm/lib/libamdhip64.so": True
    }


def test_library_missing_one_symbol_not_supported(patch_runtimes, ais_check):
    patch_runtimes(
        {"/lib/libamdhip64.so": FakeHip(present_symbols={b"hipAmdFileRead"})}
    )

    assert ais_check.hip_runtime_supports_ais() == {"/lib/libamdhip64.so": False}


def test_cdll_load_failure_skipped(patch_runtimes, ais_check):
    patch_runtimes({"/bad/libamdhip64.so": OSError("cannot load")})

    assert ais_check.hip_runtime_supports_ais() == {"/bad/libamdhip64.so": False}


def test_old_runtime_missing_get_proc_address(patch_runtimes, ais_check):
    # Resolving hipGetProcAddress raises AttributeError -> treated as not capable.
    patch_runtimes({"/old/libamdhip64.so": FakeHip(omit_get_proc_address=True)})

    assert ais_check.hip_runtime_supports_ais() == {"/old/libamdhip64.so": False}


def test_runtime_version_error_reported(patch_runtimes, capsys, ais_check):
    patch_runtimes({"/lib/libamdhip64.so": FakeHip(version_err=3)})

    result = ais_check.hip_runtime_supports_ais()

    assert result == {"/lib/libamdhip64.so": False}
    err = capsys.readouterr().err
    assert "hipRuntimeGetVersion failed" in err
    assert "/lib/libamdhip64.so" in err


def test_proc_address_err_code_fails(patch_runtimes, ais_check):
    patch_runtimes({"/lib/libamdhip64.so": FakeHip(proc_err=1)})

    assert ais_check.hip_runtime_supports_ais() == {"/lib/libamdhip64.so": False}


def test_proc_address_nonzero_status_fails(patch_runtimes, ais_check):
    patch_runtimes({"/lib/libamdhip64.so": FakeHip(proc_status=2)})

    assert ais_check.hip_runtime_supports_ais() == {"/lib/libamdhip64.so": False}


def test_proc_address_null_pointer_fails(patch_runtimes, ais_check):
    patch_runtimes({"/lib/libamdhip64.so": FakeHip(null_ptr=True)})

    assert ais_check.hip_runtime_supports_ais() == {"/lib/libamdhip64.so": False}


def test_multiple_libraries_mixed_support(patch_runtimes, ais_check):
    patch_runtimes(
        {
            "/a/libamdhip64.so": FakeHip(),
            "/b/libamdhip64.so": FakeHip(present_symbols=set()),
            "/c/libamdhip64.so": FakeHip(),
        }
    )

    result = ais_check.hip_runtime_supports_ais()

    assert result == {
        "/a/libamdhip64.so": True,
        "/b/libamdhip64.so": False,
        "/c/libamdhip64.so": True,
    }


def test_no_libraries_found(patch_runtimes, ais_check):
    patch_runtimes({})

    assert ais_check.hip_runtime_supports_ais() == {}
