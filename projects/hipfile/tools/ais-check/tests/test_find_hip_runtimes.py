"""Tests for find_hip_runtimes()."""

# The isolated_env fixture is requested by name (pytest's dependency-injection
# idiom), which pylint reads as a redefinition/unused arg. Test names document
# themselves.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument

import glob as real_glob

import pytest

_LIB = "libamdhip64.so.6"


@pytest.fixture
def isolated_env(monkeypatch, ais_check):
    """
    Strip env vars the function consults, neutralize the hard-coded /opt/rocm
    globs (so a real ROCm install on the dev box can't leak in), and make
    find_library return nothing. Tests opt back into whatever they need.
    """
    for var in ("LD_LIBRARY_PATH", "ROCM_HOME", "ROCM_PATH", "HIP_PATH"):
        monkeypatch.delenv(var, raising=False)

    original_glob = real_glob.glob

    def filtered_glob(pattern, *args, **kwargs):
        if pattern.startswith("/opt/rocm"):
            return []
        return original_glob(pattern, *args, **kwargs)

    monkeypatch.setattr(ais_check.glob, "glob", filtered_glob)
    monkeypatch.setattr(ais_check.ctypes.util, "find_library", lambda _name: None)


def _make_lib(directory, name=_LIB):
    directory.mkdir(parents=True, exist_ok=True)
    lib = directory / name
    lib.write_bytes(b"\x7fELF")
    return lib


def test_empty_when_nothing_found(ais_check, isolated_env):
    assert ais_check.find_hip_runtimes() == {}


def test_ld_library_path(monkeypatch, tmp_path, ais_check, isolated_env):
    lib = _make_lib(tmp_path / "libs")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "libs"))

    result = ais_check.find_hip_runtimes()

    assert result == {str(lib): False}


def test_ld_library_path_skips_empty_entries(
    monkeypatch, tmp_path, ais_check, isolated_env
):
    lib = _make_lib(tmp_path / "libs")
    # Leading/trailing/double colons produce empty entries that must be ignored.
    monkeypatch.setenv("LD_LIBRARY_PATH", f"::{tmp_path / 'libs'}::")

    assert ais_check.find_hip_runtimes() == {str(lib): False}


@pytest.mark.parametrize("var", ["ROCM_HOME", "ROCM_PATH", "HIP_PATH"])
def test_rocm_env_vars_lib_and_lib64(
    monkeypatch, tmp_path, ais_check, isolated_env, var
):
    lib = _make_lib(tmp_path / "lib")
    lib64 = _make_lib(tmp_path / "lib64")
    monkeypatch.setenv(var, str(tmp_path))

    result = ais_check.find_hip_runtimes()

    assert set(result) == {str(lib), str(lib64)}
    assert all(v is False for v in result.values())


def test_values_initialized_false(monkeypatch, tmp_path, ais_check, isolated_env):
    _make_lib(tmp_path / "libs")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "libs"))

    assert all(v is False for v in ais_check.find_hip_runtimes().values())


def test_nonexistent_dirs_are_skipped(monkeypatch, tmp_path, ais_check, isolated_env):
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "does-not-exist"))

    assert ais_check.find_hip_runtimes() == {}


def test_symlinks_dedup_via_realpath(monkeypatch, tmp_path, ais_check, isolated_env):
    real_dir = tmp_path / "real"
    real_lib = _make_lib(real_dir)

    # An unversioned symlink in another dir pointing at the same file.
    link_dir = tmp_path / "link"
    link_dir.mkdir()
    link = link_dir / "libamdhip64.so"
    link.symlink_to(real_lib)

    monkeypatch.setenv("LD_LIBRARY_PATH", f"{real_dir}:{link_dir}")

    result = ais_check.find_hip_runtimes()

    # Both discovered paths resolve to the one real file -> single entry.
    assert result == {str(real_lib): False}


def test_find_library_soname_kept_as_is(monkeypatch, ais_check, isolated_env):
    monkeypatch.setattr(
        ais_check.ctypes.util, "find_library", lambda _name: "libamdhip64.so.6"
    )

    result = ais_check.find_hip_runtimes()

    # A bare soname (not absolute) is preserved verbatim for CDLL-by-name.
    assert result == {"libamdhip64.so.6": False}


def test_find_library_absolute_path_resolved(
    monkeypatch, tmp_path, ais_check, isolated_env
):
    lib = _make_lib(tmp_path / "cache")
    monkeypatch.setattr(ais_check.ctypes.util, "find_library", lambda _name: str(lib))

    assert ais_check.find_hip_runtimes() == {str(lib): False}


def test_glob_metacharacters_in_env_matched_literally(
    monkeypatch, tmp_path, ais_check, isolated_env
):
    """
    A directory whose name contains a glob metacharacter must be matched
    literally (the CodeQL-driven glob.escape hardening), not interpreted.
    """
    star_dir = tmp_path / "wei*rd"
    lib = _make_lib(star_dir)
    # A sibling whose name DOES match the wildcard expansion of 'wei*rd'. If the
    # directory component is not glob.escape()d, the literal '*' is interpreted
    # and this decoy's lib leaks into the result, failing the assertion below.
    decoy = tmp_path / "weiDECOYrd"
    _make_lib(decoy)

    monkeypatch.setenv("LD_LIBRARY_PATH", str(star_dir))

    result = ais_check.find_hip_runtimes()

    assert result == {str(lib): False}


def test_unversioned_and_versioned_both_matched(
    monkeypatch, tmp_path, ais_check, isolated_env
):
    versioned = _make_lib(tmp_path / "libs", "libamdhip64.so.6")
    unversioned = _make_lib(tmp_path / "libs", "libamdhip64.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", str(tmp_path / "libs"))

    result = ais_check.find_hip_runtimes()

    assert set(result) == {str(versioned), str(unversioned)}
