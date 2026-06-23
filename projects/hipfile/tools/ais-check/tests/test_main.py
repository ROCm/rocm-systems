"""Tests for main(): the argument parsing, output, and exit-code contract."""

# The stub_checks fixture is requested by name (pytest dependency injection),
# which pylint reads as redefinition/unused. Test names are self-documenting.
# pylint: disable=missing-function-docstring,redefined-outer-name,unused-argument

from collections import namedtuple

import pytest

_UNAME = namedtuple("uname_result", "sysname nodename release version machine")


@pytest.fixture
def stub_checks(monkeypatch, ais_check):
    """
    Stub the three component checks plus os.uname so main() runs hermetically.
    Returns a setter that fixes each component's support and the HIP library map.
    """

    def configure(*, p2pdma=True, amdgpu=True, hip_libraries=None):
        if hip_libraries is None:
            hip_libraries = {"/opt/rocm/lib/libamdhip64.so": True}
        monkeypatch.setattr(ais_check, "kernel_supports_p2pdma", lambda: p2pdma)
        monkeypatch.setattr(ais_check, "amdgpu_supports_ais", lambda: amdgpu)
        monkeypatch.setattr(
            ais_check, "hip_runtime_supports_ais", lambda: dict(hip_libraries)
        )
        monkeypatch.setattr(
            ais_check.os,
            "uname",
            lambda: _UNAME("Linux", "host", "6.6.0", "#1 SMP", "x86_64"),
        )

    return configure


def _run(monkeypatch, ais_check, argv):
    monkeypatch.setattr(ais_check.sys, "argv", ["ais-check", *argv])
    return ais_check.main()


def test_all_supported_exit_zero(monkeypatch, stub_checks, ais_check):
    stub_checks()
    assert _run(monkeypatch, ais_check, []) == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"p2pdma": False},
        {"amdgpu": False},
        {"hip_libraries": {"/opt/rocm/lib/libamdhip64.so": False}},
        {"hip_libraries": {}},
    ],
)
def test_any_missing_component_exit_nonzero(
    monkeypatch, stub_checks, ais_check, kwargs
):
    stub_checks(**kwargs)
    assert _run(monkeypatch, ais_check, []) != 0


def test_hip_runtime_true_if_any_library_supported(monkeypatch, stub_checks, ais_check):
    stub_checks(
        hip_libraries={
            "/a/libamdhip64.so": False,
            "/b/libamdhip64.so": True,
        }
    )
    assert _run(monkeypatch, ais_check, []) == 0


def test_default_output_lists_components(monkeypatch, capsys, stub_checks, ais_check):
    stub_checks()
    _run(monkeypatch, ais_check, [])

    out = capsys.readouterr().out
    assert "AIS support in:" in out
    assert "Kernel P2PDMA support" in out
    assert "HIP runtime" in out
    assert "amdgpu" in out
    # The uname banner is printed.
    assert "Linux host 6.6.0" in out


def test_quiet_suppresses_output(monkeypatch, capsys, stub_checks, ais_check):
    stub_checks()
    rc = _run(monkeypatch, ais_check, ["-q"])

    assert rc == 0
    assert capsys.readouterr().out == ""


def test_quiet_still_returns_failure_code(monkeypatch, capsys, stub_checks, ais_check):
    stub_checks(p2pdma=False)
    rc = _run(monkeypatch, ais_check, ["--quiet"])

    assert rc != 0
    assert capsys.readouterr().out == ""


def test_verbose_lists_libraries(monkeypatch, capsys, stub_checks, ais_check):
    stub_checks(
        hip_libraries={
            "/opt/rocm/lib/libamdhip64.so.6": True,
            "/usr/lib/libamdhip64.so": False,
        }
    )
    _run(monkeypatch, ais_check, ["-v"])

    out = capsys.readouterr().out
    assert "Found these HIP libraries" in out
    assert "/opt/rocm/lib/libamdhip64.so.6 (AIS supported)" in out
    assert "/usr/lib/libamdhip64.so (AIS NOT supported)" in out


def test_quiet_and_verbose_mutually_exclusive(monkeypatch, ais_check):
    with pytest.raises(SystemExit):
        _run(monkeypatch, ais_check, ["-q", "-v"])
