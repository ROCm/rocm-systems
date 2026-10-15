"""RDC's AMD SMI session, against the fake amd-smi: RDC takes and releases its
own AMD SMI reference, keeps every GPU's index when AMD SMI cannot identify a
GPU, and fails cleanly when AMD SMI does not initialize.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import json
import os
import re
import resource
import subprocess
import sys
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (tools is a fixture)
    DMON_FIELDS,
    FACTOR,
    NUM_GPUS,
    PREFIX,
    Rdcd,
    _dmon_rows,
    _field_names,
    _first_file,
    _free_port,
    _missing,
    golden,
    tools,
)

AMDSMI_STATUS_SUCCESS = 0
RDC_ST_FAIL_LOAD_MODULE = 3
# rdcd retries a GPU it could not identify at most once per this many seconds.
RETRY_PERIOD_S = 5

# The host takes its own AMD SMI reference and handle, then runs RDC embedded.
_HOST = r"""
import ctypes, json, sys
smi = ctypes.CDLL(sys.argv[1])  # the preloaded fake: the library RDC calls too
rdc = ctypes.CDLL(sys.argv[2])
for f in ("amdsmi_init", "amdsmi_get_socket_handles", "amdsmi_get_processor_handles",
          "amdsmi_get_gpu_asic_info", "amdsmi_shut_down"):
    getattr(smi, f).restype = ctypes.c_uint32
smi.amdsmi_init.argtypes = [ctypes.c_uint64]
assert smi.amdsmi_init(2) == 0  # AMDSMI_INIT_AMD_GPUS
n = ctypes.c_uint32(8); sockets = (ctypes.c_void_p * 8)()
assert smi.amdsmi_get_socket_handles(ctypes.byref(n), sockets) == 0
m = ctypes.c_uint32(8); procs = (ctypes.c_void_p * 8)()
assert smi.amdsmi_get_processor_handles(ctypes.c_void_p(sockets[0]), ctypes.byref(m), procs) == 0
h = ctypes.c_void_p(procs[0])
def asic():
    return smi.amdsmi_get_gpu_asic_info(h, ctypes.create_string_buffer(4096))
out = {"before": asic()}
handle = ctypes.c_void_p()
out["rdc_init"] = rdc.rdc_init(ctypes.c_uint64(0))
out["rdc_start_embedded"] = rdc.rdc_start_embedded(0, ctypes.byref(handle))
out["running"] = asic()
out["rdc_stop_embedded"] = rdc.rdc_stop_embedded(handle)
out["rdc_shutdown"] = rdc.rdc_shutdown()
out["after"] = asic()
print(json.dumps(out))
"""

# Starts embedded RDC (with its update threads), watches fields on every GPU, reads
# them and stops, several times in one process; prints each round's values as JSON.
_ROUNDS = r"""
import ctypes, json, sys
lib = ctypes.CDLL(sys.argv[1])
fields, gpus, rounds = [int(f) for f in sys.argv[2].split(",")], int(sys.argv[3]), int(sys.argv[4])
class Data(ctypes.Union):
    _fields_ = [("l_int", ctypes.c_int64), ("dbl", ctypes.c_double), ("str", ctypes.c_char * 256)]
class Value(ctypes.Structure):
    _fields_ = [("field_id", ctypes.c_uint32), ("status", ctypes.c_int), ("ts", ctypes.c_uint64),
                ("type", ctypes.c_int), ("value", Data)]
def call(name, *args):
    st = getattr(lib, name)(*args)
    if st != 0:
        sys.exit(f"{name} returned {st}")
call("rdc_init", ctypes.c_uint64(0))
out = []
for r in range(rounds):
    handle, group, fgroup = ctypes.c_void_p(), ctypes.c_uint32(), ctypes.c_uint32()
    call("rdc_start_embedded", 0, ctypes.byref(handle))  # RDC_OPERATION_MODE_AUTO
    call("rdc_group_gpu_create", handle, 1, b"g%d" % r, ctypes.byref(group))  # RDC_GROUP_EMPTY
    for gpu in range(gpus):
        call("rdc_group_gpu_add", handle, group, gpu)
    ids = (ctypes.c_uint32 * len(fields))(*fields)
    call("rdc_group_field_create", handle, len(fields), ids, b"f%d" % r, ctypes.byref(fgroup))
    call("rdc_field_watch", handle, group, fgroup, ctypes.c_uint64(1000000), ctypes.c_double(60), 10)
    call("rdc_field_update_all", handle, 1)
    values = {}
    for gpu in range(gpus):
        for f in fields:
            v = Value()
            call("rdc_field_get_latest_value", handle, gpu, f, ctypes.byref(v))
            values["%d:%d" % (gpu, f)] = v.value.l_int if v.status == 0 else "status %d" % v.status
    out.append(values)
    call("rdc_stop_embedded", handle)
call("rdc_shutdown")
print(json.dumps(out))
"""

# Starts embedded RDC once; prints what rdc_start_embedded returned and left in the handle.
_START_ONCE = r"""
import ctypes, json, sys
lib = ctypes.CDLL(sys.argv[1])
handle = ctypes.c_void_p(0x1234)
st = lib.rdc_start_embedded(0, ctypes.byref(handle))
print(json.dumps({"status": st, "handle": handle.value or 0, "rdc_shutdown": lib.rdc_shutdown()}))
"""


def _no_core_dump():
    # A crashing child must not leave a core (or an apport report) behind.
    resource.setrlimit(resource.RLIMIT_CORE, (1, 1))


@pytest.fixture(scope="module")
def bootstrap():
    path = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    if path is None:
        _missing("librdc_bootstrap.so")
    return path


def _embedded(tools, script, *args, **fake_env):
    """Run `script` with the fake preloaded; return the JSON it printed last."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("RDC_FAKE_")}
    env.update(LD_PRELOAD=tools[2], RDC_FAKE_GPUS=str(NUM_GPUS), RDC_FAKE_EVENT_WAIT_MS="100")
    env.update(fake_env)
    r = subprocess.run(
        [sys.executable, "-c", script, *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=120 * FACTOR,
        preexec_fn=_no_core_dump,
    )
    assert r.returncode == 0, f"child exited {r.returncode}\n{r.stdout}\n{r.stderr[-3000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# RDC's own AMD SMI reference
# ---------------------------------------------------------------------------
def test_host_amd_smi_handles_survive_embedded_rdc(tools, bootstrap):
    out = _embedded(tools, _HOST, tools[2], bootstrap)
    assert out["rdc_start_embedded"] == 0 and out["rdc_stop_embedded"] == 0, out
    # RDC must take and release its own AMD SMI reference, not reset the host's session.
    assert out["before"] == out["running"] == out["after"] == AMDSMI_STATUS_SUCCESS, out


def test_embedded_rdc_restarts_in_one_process(tools, bootstrap):
    # Each stop releases AMD SMI and each start initializes it again, which gives
    # every GPU new handles: a stale handle reads nothing from the fake.
    fields = (100, 502)
    rounds = _embedded(tools, _ROUNDS, bootstrap, ",".join(map(str, fields)), str(NUM_GPUS), "10")
    want = {f"{gpu}:{f}": golden.expected(f, gpu) for gpu in range(NUM_GPUS) for f in fields}
    for n, got in enumerate(rounds):
        assert got == want, f"round {n}: {got}"


# ---------------------------------------------------------------------------
# AMD SMI fails to initialize
# ---------------------------------------------------------------------------
def test_rdc_start_embedded_reports_an_amd_smi_init_failure(tools, bootstrap):
    out = _embedded(tools, _START_ONCE, bootstrap, RDC_FAKE_FAIL="amdsmi_init")
    assert out == {"status": RDC_ST_FAIL_LOAD_MODULE, "handle": 0, "rdc_shutdown": 0}, out


def test_rdcd_exits_when_amd_smi_fails_to_initialize(tools, tmp_path):
    d = Rdcd(tools, tmp_path, RDC_FAKE_FAIL="amdsmi_init")
    d.arm()  # amdsmi_init runs at start
    with open(d.out_log, "w") as out:
        proc = subprocess.Popen(
            [d.rdcd_path, "-u", "-p", str(_free_port())],
            env=d.env,
            stdout=out,
            stderr=subprocess.STDOUT,
            preexec_fn=_no_core_dump,
        )
    try:
        code = proc.wait(timeout=20 * FACTOR)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        pytest.fail(f"rdcd did not exit after AMD SMI failed to initialize\n{d.output()}")
    out = d.output()
    assert code == 1, f"rdcd exited with {code}\n{out}"
    assert "Failed to start API service" in out, out
    assert "RDC server failed to start." in out, out


# ---------------------------------------------------------------------------
# A GPU AMD SMI cannot identify keeps its index
# ---------------------------------------------------------------------------
def _discovery(d):
    """`rdci discovery -l` as {index: device name, or None if unavailable}."""
    out = d.ok("discovery", "-l")
    m = re.search(r"^(\d+) GPUs found", out, re.M)
    assert m, out
    gpus = {}
    for line in out.splitlines():
        row = re.match(r"^(\d+)\s+(.+?)\s*$", line)
        if row and not line.endswith("GPUs found."):
            gpus[int(row.group(1))] = None if row.group(2) == "unavailable" else row.group(2)
    assert len(gpus) == int(m.group(1)), out
    return gpus


def _discovery_views(d):
    """`rdci discovery -l --json` and `rdci discovery -i` (with partitions) as
    {index: device name, or None if unavailable} each."""
    gpus = json.loads(d.ok("discovery", "-l", "--json"))["gpus"]
    from_json = {
        int(g["gpu_index"]): (None if g.get("available") is False else g["device_name"])
        for g in gpus
    }
    from_partitions = {}
    for line in d.ok("discovery", "-i").splitlines():
        row = re.match(r"^(\d+)\s+(.+?)(\s{2,}|$)", line)
        if row and not line.endswith("GPUs found."):
            name = row.group(2).strip()
            from_partitions[int(row.group(1))] = None if name == "unavailable" else name
    return from_json, from_partitions


def _type_calls(d, gpu):
    return sum(1 for api, g, _ in d.calls() if api == "amdsmi_get_processor_type" and g == gpu)


# A GPU the fake fails to identify, first and second in the list: the others keep their
# indexes around it, and the GPU count does not depend on GPU 0.
@pytest.mark.parametrize("unidentified", [0, 1])
def test_a_gpu_amd_smi_cannot_identify_keeps_its_index(tools, tmp_path, unidentified):
    d = Rdcd(tools, tmp_path, RDC_FAKE_FAIL=f"amdsmi_get_processor_type@{unidentified}")
    d.arm()  # rdcd lists the GPUs when it starts
    started = time.monotonic()
    d.start()
    try:
        want = {g: golden.device_name(g) for g in range(NUM_GPUS)}
        want[unidentified] = None
        assert _discovery(d) == want
        assert _discovery_views(d) == (want, want)

        names = _field_names(d)
        gpus = ",".join(str(g) for g in range(NUM_GPUS))
        fields = ",".join(str(f) for f in DMON_FIELDS)
        # About 12 s of samples, so rdcd retries the unidentified GPU along the way.
        out = d.ok("dmon", "-e", fields, "-i", gpus, "-c", "24", "-d", "500", timeout=60)
        rows = _dmon_rows(out)
        assert sorted(rows) == list(range(NUM_GPUS)), out
        for gpu in range(NUM_GPUS):
            for field in DMON_FIELDS:
                got = rows[gpu][names[field]]
                want_value = "N/A" if gpu == unidentified else str(golden.expected(field, gpu))
                assert got == want_value, f"GPU {gpu} {names[field]}: {got}\n{out}"
        # The unidentified GPU has no links; the others still list theirs.
        links = d.ok("link", "-l")
        rows = re.findall(r"^\s*GPU:(\d+)\n(.*)$", links, re.M)
        assert [(int(g), bool(row.strip())) for g, row in rows] == [
            (g, g != unidentified) for g in range(NUM_GPUS)
        ], links
        # rdci topo from another GPU: no link to the unidentified GPU, the others' links read.
        source = next(g for g in range(NUM_GPUS) if g != unidentified)
        topo = d.ok("topo", "-i", str(source))
        rows = re.findall(r"^\| To GPU (\d+)\s*\|\s*(.+?)\s*\|$", topo, re.M)
        assert {int(g): row == "N/A" for g, row in rows} == {
            g: g == unidentified for g in range(NUM_GPUS) if g != source
        }, topo
        # The node topology check passes without the unidentified GPU as a source, instead of
        # failing on its unset handle; the group lists every GPU, so pair indexes are GPU indexes.
        group = re.search(r"group ID (\d+)", d.ok("group", "-c", "all", "-a", gpus))
        assert group, "no group created"
        diag = d.ok("diag", "-g", group.group(1), "-r", "1", timeout=120)
        assert re.search(r"^Node topology check:\s+Pass", diag, re.M), diag
        sources = {int(i) for i in re.findall(r"(\d+)=>\d+ weight:", diag)}
        assert sources == set(range(NUM_GPUS)) - {unidentified}, diag
        # One identification at start, then at most one retry per period.
        elapsed = time.monotonic() - started
        calls = _type_calls(d, unidentified)
        assert 2 <= calls <= 2 + elapsed / RETRY_PERIOD_S, f"{calls} calls in {elapsed:.1f} s"
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


@pytest.mark.parametrize("unidentified", [0, 1])
def test_a_gpu_identified_late_takes_its_own_index(tools, tmp_path, unidentified):
    d = Rdcd(tools, tmp_path, RDC_FAKE_FAIL_ONCE=f"amdsmi_get_processor_type@{unidentified}")
    d.arm()  # only the identification at start fails
    d.start()
    try:
        assert _discovery(d)[unidentified] is None
        names = _field_names(d)
        want = {names[f]: str(golden.expected(f, unidentified)) for f in DMON_FIELDS}
        fields = ",".join(str(f) for f in DMON_FIELDS)
        deadline = time.monotonic() + (RETRY_PERIOD_S + 10) * FACTOR
        while True:
            out = d.ok("dmon", "-e", fields, "-i", str(unidentified), "-c", "1", "-d", "500")
            row = _dmon_rows(out).get(unidentified)
            if row == want:
                break
            assert time.monotonic() < deadline, f"GPU {unidentified} still reads {row}\n{out}"
            time.sleep(0.5)
        assert _discovery(d) == {g: golden.device_name(g) for g in range(NUM_GPUS)}
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"
