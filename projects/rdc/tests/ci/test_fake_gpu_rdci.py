"""rdci against a fake-GPU rdcd: its options, help, exit statuses and what it prints, and the
checks rdcd applies to names and policies it gets from any client.

Runs from the installed fake-GPU suite and uses test_fake_gpu.py's harness. Some tests preload
librdc_fake_client.so (tests/fake_amdsmi/fake_rdc_client.cc) into rdci, for replies that a
well-behaved rdcd does not send: a failing call, or names with control characters.

Environment, besides test_fake_gpu.py's:
  RDC_FAKE_CLIENT_LIB   librdc_fake_client.so to use (default: next to the fake amd-smi)
"""

import json
import os
import re
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_gpu_golden as golden  # noqa: E402
from test_fake_gpu import (  # noqa: E402, F401 (rdcd and tools are fixtures)
    FACTOR,
    PREFIX,
    Rdcd,
    _all_gpus_group,
    _created_id,
    _first_file,
    _missing,
    rdcd,
    tools,
)

CLIENT_NAME = "librdc_fake_client.so"
# rdc.h's rdc_status_t, which rdci also exits with.
RDC_ST_OK = 0
RDC_ST_BAD_PARAMETER = 5
RDC_ST_NOT_FOUND = 6


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


def _strings(doc):
    """Every string in a parsed JSON document."""
    if isinstance(doc, str):
        return [doc]
    if isinstance(doc, dict):
        return [s for value in doc.values() for s in _strings(value)]
    if isinstance(doc, list):
        return [s for value in doc for s in _strings(value)]
    return []


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


# ---------------------------------------------------------------------------
# rdci policy: one defined condition per set, help, and the status of each command
# ---------------------------------------------------------------------------
def test_policy_set_without_a_condition_is_refused(rdcd):
    gid = _all_gpus_group(rdcd, "pol")
    r = rdcd.rdci("policy", "-g", gid, "-A", "1")
    assert r.returncode == RDC_ST_BAD_PARAMETER, r.stdout + r.stderr
    # A policy with an undefined condition crashed rdcd's policy thread within a second.
    time.sleep(3 * FACTOR)
    rdcd.check_alive("3 s after rdci policy -A 1")


def test_policy_set_takes_one_condition(rdcd):
    gid = _all_gpus_group(rdcd, "pol")
    r = rdcd.rdci("policy", "-g", gid, "-T", "95", "-P", "300")
    assert r.returncode == RDC_ST_BAD_PARAMETER, r.stdout + r.stderr
    out = rdcd.rdci("policy", "-g", gid, "--get").stdout
    assert "Temperature Limit" not in out and "Power Limit" not in out, out


def test_policy_help_exits_successfully(rdcd):
    r = rdcd.rdci("policy", "--help")
    assert r.returncode == RDC_ST_OK, r.stdout + r.stderr
    assert "Usage" in r.stdout and "rdci Error" not in r.stdout + r.stderr, r.stdout + r.stderr


def test_policy_get_and_clear_on_a_group_without_policy(rdcd):
    gid = _all_gpus_group(rdcd, "pol")
    message = f"No policy is set on GPU group {gid}"
    r = rdcd.rdci("policy", "-g", gid, "--get")
    assert r.returncode == RDC_ST_NOT_FOUND and message in r.stdout, r.stdout + r.stderr
    r = rdcd.rdci("policy", "-g", gid, "--clear")
    assert r.returncode == RDC_ST_OK and message in r.stdout, r.stdout + r.stderr
    assert "Policy is cleared" not in r.stdout, r.stdout


def test_policy_clear_of_a_group_with_one_condition(rdcd):
    gid = _all_gpus_group(rdcd, "pol")
    rdcd.ok("policy", "-g", gid, "-T", "95")
    assert f"Policy is cleared on GPU group {gid}" in rdcd.ok("policy", "-g", gid, "--clear")
    assert "Temperature Limit" not in rdcd.ok("policy", "-g", gid, "--get")


# ---------------------------------------------------------------------------
# The C API, in a child process so a crash cannot take pytest down: rdcd (standalone) and
# librdc (embedded) refuse undefined policy values and unclean names
# ---------------------------------------------------------------------------
_CHILD = r"""
import ctypes, json, sys, time
lib = ctypes.CDLL(sys.argv[1])
target, check, settle = sys.argv[2], sys.argv[3], float(sys.argv[4])
class Policy(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("value", ctypes.c_int64), ("action", ctypes.c_int)]
lib.rdc_init.argtypes = [ctypes.c_uint64]
lib.rdc_connect.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                            ctypes.c_char_p, ctypes.c_char_p]
lib.rdc_start_embedded.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
lib.rdc_stop_embedded.argtypes = [ctypes.c_void_p]
lib.rdc_disconnect.argtypes = [ctypes.c_void_p]
lib.rdc_group_gpu_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
                                     ctypes.POINTER(ctypes.c_uint32)]
lib.rdc_group_field_create.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                       ctypes.POINTER(ctypes.c_uint32), ctypes.c_char_p,
                                       ctypes.POINTER(ctypes.c_uint32)]
lib.rdc_policy_set.argtypes = [ctypes.c_void_p, ctypes.c_uint32, Policy]
lib.rdc_policy_delete.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int]
handle = ctypes.c_void_p()
lib.rdc_init(0)
if target == "embedded":
    status = lib.rdc_start_embedded(0, ctypes.byref(handle))
else:
    status = lib.rdc_connect(target.encode(), ctypes.byref(handle), None, None, None)
assert status == 0, f"start: {status}"
group = ctypes.c_uint32()
assert lib.rdc_group_gpu_create(handle, 0, b"checks", ctypes.byref(group)) == 0
got = {}
if check == "policy":
    for name, policy in (("negative type", Policy(-1, 1, 0)), ("type past the last", Policy(3, 1, 0)),
                         ("negative action", Policy(1, 95000, -1)),
                         ("action past reset", Policy(1, 95000, 2))):
        got[name] = lib.rdc_policy_set(handle, group.value, policy)
    got["delete negative type"] = lib.rdc_policy_delete(handle, group.value, -1)
    got["delete type past the last"] = lib.rdc_policy_delete(handle, group.value, 3)
    # Last: on a tree without the check, the policy thread reads this far below its table.
    got["most negative type"] = lib.rdc_policy_set(handle, group.value,
                                                   Policy(-2**31, 1, 0))
else:
    fields = (ctypes.c_uint32 * 1)(100)
    for name in json.loads(sys.argv[5]):
        raw = bytes.fromhex(name)
        out = ctypes.c_uint32()
        got["group " + name] = lib.rdc_group_gpu_create(handle, 0, raw, ctypes.byref(out))
        got["field group " + name] = lib.rdc_group_field_create(handle, 1, fields, raw,
                                                                ctypes.byref(out))
time.sleep(settle)  # rdc's policy thread checks every policy meanwhile
print(json.dumps(got), flush=True)
if target == "embedded":
    lib.rdc_stop_embedded(handle)
    lib.rdc_shutdown()
else:
    lib.rdc_disconnect(handle)
"""


def _api_child(target, check, env, names=()):
    """Runs _CHILD; returns {call: status} or fails the test."""
    bootstrap = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    if bootstrap is None:
        _missing("librdc_bootstrap.so")
    args = [sys.executable, "-c", _CHILD, bootstrap, target, check, str(2 * FACTOR)]
    args.append(json.dumps([name.hex() for name in names]))
    try:
        r = subprocess.run(args, env=env, capture_output=True, text=True, timeout=120 * FACTOR)
    except subprocess.TimeoutExpired:
        pytest.fail(f"the {check} child hung (killed)")
    assert r.returncode == 0, f"{check} child exited {r.returncode}\n{r.stdout}{r.stderr[-3000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_rdcd_refuses_undefined_policy_values(rdcd):
    got = _api_child(f"127.0.0.1:{rdcd.port}", "policy", dict(os.environ))
    rdcd.check_alive("policies with undefined values")
    assert set(got.values()) == {RDC_ST_BAD_PARAMETER}, got


def test_embedded_rdc_refuses_undefined_policy_values(tools):
    env = dict(os.environ, LD_PRELOAD=tools[2], RDC_FAKE_GPUS="4")
    got = _api_child("embedded", "policy", env)
    assert set(got.values()) == {RDC_ST_BAD_PARAMETER}, got


# ---------------------------------------------------------------------------
# rdci topo: the help lists the options rdci topo takes
# ---------------------------------------------------------------------------
def test_topo_help_lists_the_options_it_takes(rdcd):
    for flag in ("--help", "-h"):
        r = rdcd.rdci("topo", flag)
        assert r.returncode == RDC_ST_OK, (flag, r.stdout + r.stderr)
        assert "--gpu_index" in r.stdout, r.stdout
        assert not re.search(r"(?<![\w-])-g\b|--index\b|--reg\b", r.stdout), r.stdout
    assert "Topology Information" in rdcd.ok("topo", "--gpu_index", "0")


# ---------------------------------------------------------------------------
# rdci dmon: columns, a failed watch, and events from every GPU
# ---------------------------------------------------------------------------
def test_dmon_separates_its_columns(rdcd):
    # DEV_NAME (2) is longer than its column; UUID (8) follows it.
    out = rdcd.ok("dmon", "-i", "0", "-e", "2,8", "-c", "1", "-d", "500")
    name = re.escape(golden.device_name(0))
    assert re.search(rf"^0\s+{name}\s+\S", out, re.M), out


def test_dmon_fails_when_its_watch_fails(rdcd, client):
    r = _rdci(
        rdcd,
        "dmon",
        "-i",
        "0",
        "-e",
        "100",
        "-c",
        "2",
        "-d",
        "200",
        client=client,
        RDC_FAKE_CLIENT_FAIL="rdc_field_watch",
    )
    assert r.returncode != RDC_ST_OK, _text(r)
    assert "rdci-dmon" not in rdcd.ok("group", "-l"), "dmon left its group behind"


def _dmon_events(d, fields):
    """dmon on GPUs 0 and 1 with the fake's events armed once dmon has started and read the
    events already there; returns dmon's output."""
    samples = int(4 + 2 * FACTOR)
    cmd = [d.rdci_path, "dmon", "-i", "0,1", "-e", fields, "-c", str(samples), "-d", "1000"]
    cmd += ["-u", "--host", f"127.0.0.1:{d.port}"]
    proc = subprocess.Popen(
        cmd, env=d.rdci_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        time.sleep(2 * FACTOR)
        d.arm()
        out, _ = proc.communicate(timeout=(samples + 30) * FACTOR)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        pytest.fail("rdci dmon did not end")
    d.check_alive("rdci dmon with events")
    assert proc.returncode == RDC_ST_OK, out
    return out


def test_dmon_reports_the_events_of_every_gpu(tools, tmp_path):
    # GPU 1's event comes first, so it is older than GPU 0's.
    d = Rdcd(tools, tmp_path, gpus=2, RDC_FAKE_EVENTS="1@1,1@0").start()
    try:
        out = _dmon_events(d, "2000,100")
        for gpu in (0, 1):
            assert re.search(rf"^{gpu}\s+\*\*Event: VM_PAGE_FAULT", out, re.M), (gpu, out)
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


def test_dmon_with_event_fields_only_ends_each_line(tools, tmp_path):
    d = Rdcd(tools, tmp_path, gpus=2, RDC_FAKE_EVENTS="1@0").start()
    try:
        out = _dmon_events(d, "2000")
        assert re.search(r"^0\s+\*\*Event: VM_PAGE_FAULT", out, re.M), out
        assert not re.search(r"^\d+\s+\d+(\s|$)", out, re.M), f"GPU indexes run on:\n{out}"
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


# ---------------------------------------------------------------------------
# rdci discovery: partition IDs, a failed profile lookup, a failed version query
# ---------------------------------------------------------------------------
def _gpu_rows(out):
    """`rdci discovery -i` GPU rows: [(GPU index, XCC, DECODER)]."""
    rows = []
    for line in out.splitlines():
        m = re.match(r"^(\d+)\s{2,}.*\s(\S+)\s+(\S+)\s*$", line)
        if m:
            rows.append((int(m.group(1)), m.group(2), m.group(3)))
    return rows


def _partition_rows(out):
    """`rdci discovery -i` partition rows: [(GPU index above, partition ID, XCC, DECODER)]."""
    rows, gpu = [], None
    for line in out.splitlines():
        m = re.match(r"^(\d+)\s{2,}", line)
        if m:
            gpu = int(m.group(1))
            continue
        m = re.match(r"^\s+(g\d+\.\d+)\s+(\S+)\s+\*?(\S+)\s*$", line)
        if m:
            rows.append((gpu, m.group(1), m.group(2), m.group(3)))
    return rows


def test_discovery_shows_each_partition_once_with_its_id(tools, tmp_path):
    # Two sockets of two partitions: GPU indexes 0 and 1 are socket 0's, 2 and 3 socket 1's.
    d = Rdcd(tools, tmp_path, gpus=2, RDC_FAKE_PARTITIONS=2).start()
    try:
        out = d.ok("discovery", "-i")
        rows = _partition_rows(out)
        assert [row[:2] for row in rows] == [(0, "g0.0"), (1, "g0.1"), (2, "g1.0"), (3, "g1.1")], (
            out
        )
        assert all(xcc.isdigit() and int(xcc) > 0 for _, _, xcc, _ in rows), out
        for _, partition, _, _ in rows:
            socket = int(partition[1:].split(".")[0])
            dmon = d.ok("dmon", "-i", partition, "-e", "100", "-c", "1", "-d", "200")
            m = re.search(rf"^{re.escape(partition)}\s+(\S+)", dmon, re.M)
            assert m and m.group(1) == str(golden.expected(100, socket)), dmon
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


def test_discovery_shows_na_for_a_failed_profile_lookup(tools, tmp_path):
    # The partition counts come from another call, so only the profiles fail.
    d = Rdcd(
        tools,
        tmp_path,
        gpus=2,
        RDC_FAKE_PARTITIONS=2,
        RDC_FAKE_FAIL="amdsmi_get_gpu_accelerator_partition_profile_config",
    ).start()
    try:
        d.arm()
        out = d.ok("discovery", "-i")
        cells = [row[1:] for row in _gpu_rows(out)] + [row[2:] for row in _partition_rows(out)]
        assert len(cells) == 8 and set(cells) == {("N/A", "N/A")}, out
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


def test_discovery_version_fails_with_its_query(tools, tmp_path):
    d = Rdcd(tools, tmp_path, RDC_FAKE_FAIL="amdsmi_get_lib_version").start()
    try:
        d.arm()
        r = d.rdci("discovery", "-v")
        assert r.returncode != RDC_ST_OK, r.stdout + r.stderr
        assert "RDCD :" not in r.stdout, r.stdout
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


# ---------------------------------------------------------------------------
# Names: rdcd and librdc refuse names that are not clean text
# ---------------------------------------------------------------------------
# Names that are not clean text: a control character, or not valid UTF-8 (RFC 3629).
UNCLEAN_NAMES = [
    b"esc\x1b[31m",
    b"tab\tname",
    b"del\x7f",
    b"csi\xc2\x9b31m",
    b"lone\x9b",
    b"latin\xe9",
    b"overlong\xc0\x80",
    b"surrogate\xed\xa0\x80",
    b"past\xf4\x90\x80\x80",
    b"cut\xe2\x82",
]
# rdcd gets only valid UTF-8 from rdci: the client's protocol buffers refuse to send the rest.
CONTROL_NAMES = [b"esc\x1b[31m", b"tab\tname", b"del\x7f", b"csi\xc2\x9b31m"]


def test_embedded_rdc_refuses_unclean_names(tools):
    env = dict(os.environ, LD_PRELOAD=tools[2], RDC_FAKE_GPUS="4")
    got = _api_child("embedded", "names", env, UNCLEAN_NAMES + ["calf\u00e9".encode()])
    assert got.pop("group " + "calf\u00e9".encode().hex()) == RDC_ST_OK, got
    assert got.pop("field group " + "calf\u00e9".encode().hex()) == RDC_ST_OK, got
    assert set(got.values()) == {RDC_ST_BAD_PARAMETER}, got


def test_rdcd_refuses_names_with_control_characters(rdcd):
    for name in CONTROL_NAMES:
        r = _rdci(rdcd, "group", "-c", name)
        assert r.returncode == RDC_ST_BAD_PARAMETER, (name, _text(r))
        r = _rdci(rdcd, "fieldgroup", "-c", name, "-f", "100")
        assert r.returncode == RDC_ST_BAD_PARAMETER, (name, _text(r))
    listed = _rdci(rdcd, "group", "-l").stdout + _rdci(rdcd, "fieldgroup", "-l").stdout
    assert not [name for name in CONTROL_NAMES if name in listed], listed
    r = _rdci(rdcd, "group", "-c", "calf\u00e9".encode())
    assert r.returncode == RDC_ST_OK, _text(r)


# ---------------------------------------------------------------------------
# rdci escapes control characters in text from rdcd
# ---------------------------------------------------------------------------
NAME = b'esc\x1b[31m "quote" csi\xc2\x9b back\\slash'
NAME_IN_TEXT = rb'esc\x1b[31m "quote" csi\xc2\x9b back\slash'  # how rdci prints it
NAME_IN_JSON = 'esc\x1b[31m "quote" csi\x9b back\\slash'  # what a JSON parser reads back


def test_rdci_shows_control_characters_from_rdcd_escaped(rdcd, client):
    gid = _all_gpus_group(rdcd, "named")
    fgid = _created_id(rdcd.ok("fieldgroup", "-c", "named_fields", "-f", "100"))
    rdcd.ok("stats", "-s", "named_job", "-g", gid)
    try:
        for command in (
            ("group", "-l"),
            ("group", "-g", gid, "-i"),
            ("fieldgroup", "-l"),
            ("fieldgroup", "-g", fgid, "-i"),
            ("stats", "-j", "named_job", "-v"),
        ):
            r = _rdci(rdcd, *command, client=client, RDC_FAKE_CLIENT_NAME=NAME)
            assert r.returncode == RDC_ST_OK, (command, _text(r))
            assert b"\x1b" not in r.stdout and b"\xc2\x9b" not in r.stdout, (command, r.stdout)
            assert NAME_IN_TEXT in r.stdout, (command, r.stdout)
            r = _rdci(rdcd, *command, "--json", client=client, RDC_FAKE_CLIENT_NAME=NAME)
            assert r.returncode == RDC_ST_OK, (command, _text(r))
            assert b"\x1b" not in r.stdout and b"\xc2\x9b" not in r.stdout, (command, r.stdout)
            try:
                doc = json.loads(r.stdout)
            except ValueError as e:
                pytest.fail(f"rdci {' '.join(command)} --json is not JSON ({e}):\n{r.stdout!r}")
            assert NAME_IN_JSON in _strings(doc), (command, doc)
    finally:
        rdcd.rdci("stats", "-x", "named_job")
        rdcd.rdci("stats", "-r", "named_job")


# ---------------------------------------------------------------------------
# rdci diag and rdcd's diagnostic log
# ---------------------------------------------------------------------------
def test_diag_config_option_prints_no_debug_lines(rdcd):
    gid = _all_gpus_group(rdcd, "diag")
    r = rdcd.rdci("diag", "-g", gid, "-r", "1", "-c", "custom", timeout=120)
    assert "config_test_" not in r.stdout + r.stderr, r.stdout + r.stderr


def _run_calls(d):
    return sum(1 for api, _, _ in d.calls() if api != "amdsmi_get_gpu_event_notification")


def test_rdcd_logs_a_diag_client_that_goes_away_once(tools, tmp_path):
    # Each temperature read takes 50 ms, so the run goes on for seconds after rdci is gone.
    d = Rdcd(tools, tmp_path, RDC_FAKE_DELAY_MS="amdsmi_get_temp_metric=50")
    d.arm()
    d.start()
    try:
        gid = _all_gpus_group(d, "diag")
        log_start = d.out_log.stat().st_size
        cmd = [d.rdci_path, "diag", "-g", gid, "-r", "1", "-u", "--host", f"127.0.0.1:{d.port}"]
        proc = subprocess.Popen(
            cmd, env=d.rdci_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        try:
            first = proc.stdout.readline()
        finally:
            proc.kill()
            proc.communicate()
        assert first.startswith("LOG:"), first
        gone = len(d.calls())
        # The run has ended when the fake has seen no call for 3 s but the event waits rdcd
        # makes every second anyway.
        deadline = time.monotonic() + 120 * FACTOR
        seen, quiet_since = _run_calls(d), time.monotonic()
        while time.monotonic() - quiet_since < 3 * FACTOR:
            assert time.monotonic() < deadline, "the diagnostic run did not end"
            time.sleep(0.5)
            now = _run_calls(d)
            if now != seen:
                seen, quiet_since = now, time.monotonic()
        d.check_alive("a diagnostic run whose client went away")
        later = [api for api, _, _ in d.calls()[gone:]]
        assert "amdsmi_get_temp_metric" in later, f"the run stopped with its client: {later}"
        log = d.out_log.read_text(errors="replace")[log_start:]
        assert "Failed to write log message" not in log, log[-4000:]
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"
