"""Tests for kernel_supports_p2pdma()."""

# Test functions are self-documenting by name; docstrings add noise.
# pylint: disable=missing-function-docstring

import io
from collections import namedtuple

import pytest

_UNAME = namedtuple("uname_result", "sysname nodename release version machine")


def _fake_uname(release="6.6.0-test"):
    return _UNAME("Linux", "host", release, "#1 SMP", "x86_64")


def _install_fake_openers(monkeypatch, ais_check, files):
    """
    Redirect both ``open`` (builtins) and ``gzip.open`` so that only the paths
    present in ``files`` are readable; everything else raises OSError, matching
    a missing config file.
    """

    def opener(path, *_args, **_kwargs):
        if path in files:
            return io.StringIO(files[path])
        raise OSError(f"no such file: {path}")

    monkeypatch.setattr(ais_check, "open", opener, raising=False)
    monkeypatch.setattr(ais_check.gzip, "open", opener)


@pytest.mark.parametrize(
    "config_path",
    [
        "/boot/config-6.6.0-test",
        "/lib/modules/6.6.0-test/build/.config",
        "/proc/config.gz",
    ],
)
def test_supported_from_each_source(monkeypatch, ais_check, config_path):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {config_path: "CONFIG_FOO=y\nCONFIG_PCI_P2PDMA=y\nCONFIG_BAR=m\n"},
    )

    assert ais_check.kernel_supports_p2pdma() is True


def test_config_present_but_option_absent(monkeypatch, ais_check):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {"/boot/config-6.6.0-test": "CONFIG_FOO=y\nCONFIG_BAR=m\n"},
    )

    assert ais_check.kernel_supports_p2pdma() is False


def test_option_as_module_does_not_count(monkeypatch, ais_check):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {"/boot/config-6.6.0-test": "CONFIG_PCI_P2PDMA=m\n"},
    )

    assert ais_check.kernel_supports_p2pdma() is False


def test_option_commented_out_does_not_count(monkeypatch, ais_check):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {"/boot/config-6.6.0-test": "# CONFIG_PCI_P2PDMA is not set\n"},
    )

    assert ais_check.kernel_supports_p2pdma() is False


def test_no_configs_found_warns_on_stderr(monkeypatch, capsys, ais_check):
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(monkeypatch, ais_check, {})

    assert ais_check.kernel_supports_p2pdma() is False
    assert "No kernel config files found!" in capsys.readouterr().err


def test_first_matching_source_wins(monkeypatch, ais_check):
    """
    A readable /boot config without the option followed by /proc/config.gz with
    it should still report supported (the loop must consult all sources, not
    bail after the first readable one).
    """
    monkeypatch.setattr(ais_check.os, "uname", _fake_uname)
    _install_fake_openers(
        monkeypatch,
        ais_check,
        {
            "/boot/config-6.6.0-test": "CONFIG_FOO=y\n",
            "/proc/config.gz": "CONFIG_PCI_P2PDMA=y\n",
        },
    )

    assert ais_check.kernel_supports_p2pdma() is True
