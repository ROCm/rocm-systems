"""rdci against a fake-GPU rdcd: its options, help, exit statuses and what it prints.

Runs from the installed fake-GPU suite and uses test_fake_gpu.py's harness. Some tests preload
librdc_fake_client.so (tests/fake_amdsmi/fake_rdc_client.cc) into rdci, for replies that a
well-behaved rdcd does not send: a failing call.

Environment, besides test_fake_gpu.py's:
  RDC_FAKE_CLIENT_LIB   librdc_fake_client.so to use (default: next to the fake amd-smi)
"""

import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_fake_gpu import (  # noqa: E402, F401 (rdcd and tools are fixtures)
    FACTOR,
    _all_gpus_group,
    _first_file,
    _missing,
    rdcd,
    tools,
)

CLIENT_NAME = "librdc_fake_client.so"
# rdc.h's rdc_status_t, which rdci also exits with.
RDC_ST_OK = 0


@pytest.fixture(scope="module")
def client(tools):
    """librdc_fake_client.so: RDC_FAKE_CLIENT_LIB, else the one next to the fake amd-smi."""
    path = _first_file(
        os.environ.get("RDC_FAKE_CLIENT_LIB"), os.path.join(os.path.dirname(tools[2]), CLIENT_NAME)
    )
    if path is None:
        _missing(CLIENT_NAME)
    return path


def _rdci(d, *args, client=None, timeout=30, **client_env):
    """rdci against `d` as d.rdci() runs it, with bytes for arguments and output; with `client`,
    the fake client is preloaded with the given RDC_FAKE_CLIENT_* settings."""
    env = {os.fsencode(k): os.fsencode(v) for k, v in d.rdci_env.items()}
    if client is not None:
        env[b"LD_PRELOAD"] = os.fsencode(client)
        for key, value in client_env.items():
            env[key.encode()] = value if isinstance(value, bytes) else str(value).encode()
    cmd = [os.fsencode(d.rdci_path)]
    cmd += [a if isinstance(a, bytes) else str(a).encode() for a in args]
    cmd += [b"-u", b"--host", f"127.0.0.1:{d.port}".encode()]
    try:
        r = subprocess.run(cmd, env=env, capture_output=True, timeout=timeout * FACTOR)
    except subprocess.TimeoutExpired:
        pytest.fail(f"rdci {args} timed out")
    d.check_alive(f"rdci {args}")
    return r


def _text(r):
    return (r.stdout + r.stderr).decode(errors="replace")


# ---------------------------------------------------------------------------
# rdci config: -m takes its value, --host works, and failures are reported
# ---------------------------------------------------------------------------
CONFIG_SET = ("-s", "-p", "400", "-x", "1500", "-m", "1200")  # inside the fake's ranges


def test_config_sets_every_limit_through_host(rdcd):
    gid = _all_gpus_group(rdcd, "cfg")
    try:
        out = rdcd.ok("config", "-g", gid, *CONFIG_SET)
        assert f"Successfully configured GPU Id belongs to group: {gid}" in out, out
    finally:
        rdcd.rdci("config", "-g", gid, "-c")


def test_config_reports_a_failed_watch_with_its_group_id(rdcd, client):
    gid = _all_gpus_group(rdcd, "cfg")
    try:
        r = _rdci(
            rdcd,
            "config",
            "-g",
            gid,
            "-s",
            "-p",
            "400",
            client=client,
            RDC_FAKE_CLIENT_FAIL="rdc_field_watch",
        )
        out = _text(r)
        assert r.returncode != RDC_ST_OK, out
        assert f"Error creating field watch for group id: {gid}\n" in out, out
        # The next --set replaces the field group the failed one left unwatched.
        rdcd.ok("config", "-g", gid, "-s", "-p", "400")
    finally:
        rdcd.rdci("config", "-g", gid, "-c")


def test_config_set_fails_when_its_old_field_group_stays(rdcd, client):
    gid = _all_gpus_group(rdcd, "cfg")
    try:
        rdcd.ok("config", "-g", gid, "-s", "-p", "400")
        r = _rdci(
            rdcd,
            "config",
            "-g",
            gid,
            "-s",
            "-x",
            "1500",
            client=client,
            RDC_FAKE_CLIENT_FAIL="rdc_group_field_destroy",
        )
        out = _text(r)
        assert r.returncode != RDC_ST_OK, out
        assert "Successfully removed field group" not in out, out
    finally:
        rdcd.rdci("config", "-g", gid, "-c")


def test_config_clear_without_a_field_group(rdcd):
    gid = _all_gpus_group(rdcd, "cfg")
    out = rdcd.ok("config", "-g", gid, "-c")
    assert f"Successfully cleared all configuration for group: {gid}" in out, out
    assert "Successfully deleted the field group" not in out, out


def test_config_clear_fails_when_its_field_group_stays(rdcd, client):
    gid = _all_gpus_group(rdcd, "cfg")
    rdcd.ok("config", "-g", gid, "-s", "-p", "400")
    try:
        r = _rdci(
            rdcd,
            "config",
            "-g",
            gid,
            "-c",
            client=client,
            RDC_FAKE_CLIENT_FAIL="rdc_group_field_destroy",
        )
        out = _text(r)
        assert r.returncode != RDC_ST_OK, out
        assert "Successfully cleared all configuration" not in out, out
    finally:
        rdcd.rdci("config", "-g", gid, "-c")


def test_config_set_stops_at_a_failed_setting(rdcd):
    gid = _all_gpus_group(rdcd, "cfg")
    try:
        # The fake takes GFX clock limits of 500-2100 MHz; the power limit after it is valid.
        r = rdcd.rdci("config", "-g", gid, "-s", "-x", "9999", "-p", "400")
        assert r.returncode != RDC_ST_OK, r.stdout + r.stderr
        assert "Successfully configured" not in r.stdout, r.stdout
    finally:
        rdcd.rdci("config", "-g", gid, "-c")
