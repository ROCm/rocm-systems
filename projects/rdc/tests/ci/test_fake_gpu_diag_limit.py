"""rdcd runs at most two diagnostics at a time, rdc_diagnostic_run() and
rdc_test_case_run() together, and rejects another with RDC_ST_MAX_LIMIT; the
client returns the status rdcd sends for a diagnostic. On fake GPUs.

Runs like test_fake_gpu.py (same environment variables and helpers). A
diagnostic of the group {0, 1} (level 1, or the node topology test case) reads
the link weight from GPU 0 and then from GPU 1. RDC_FAKE_DELAY_MS holds the read
from GPU 1, so a run that has read from GPU 0 holds its slot for the delay.
"""

import ctypes
import os
import subprocess
import sys
import threading
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import FACTOR, PREFIX, Rdcd, _created_id, _free_port, tools  # noqa: F401 (fixture)

RDC_ST_OK = 0
RDC_ST_CLIENT_ERROR = 8
RDC_ST_MAX_LIMIT = 10
RDC_ST_GROUP_NOT_FOUND = 17
RDC_DIAG_NODE_TOPOLOGY = 3
MAX_RUNS = 2  # RdcAPIServiceImpl::kMaxDiagnosticRuns
LINK_WEIGHT = "amdsmi_topo_get_link_weight"
DELAY_S = min(60, 4 * FACTOR)
MISSING_GROUP = 4242
TEST_RESULT_SIZE = 1 << 20  # more than sizeof(rdc_diag_test_result_t)


@pytest.fixture
def rdcd(tools, tmp_path):
    d = Rdcd(tools, tmp_path, RDC_FAKE_DELAY_MS=f"{LINK_WEIGHT}@1={int(DELAY_S * 1000)}").start()
    yield d
    code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


def _pair_group(d, delayed=True):
    gid = _created_id(d.ok("group", "-c", "diag_pair"))
    d.ok("group", "-g", gid, "-a", "0,1")
    if delayed:
        d.arm()  # the delay applies from now on
    return int(gid)


def _link_reads(d, since, gpu):
    return sum(1 for api, g, _ in d.calls()[since:] if api == LINK_WEIGHT and g == gpu)


def _wait_for_link_reads(d, since, gpu, count):
    deadline = time.monotonic() + (DELAY_S + 10) * FACTOR
    while _link_reads(d, since, gpu) < count:
        if time.monotonic() > deadline:
            pytest.fail(f"{count} diagnostics did not read GPU {gpu}'s links\n{d.output()}")
        time.sleep(0.05)


def _lib():
    # Each helper calls rdc_init(); a process may call it more than once.
    lib = ctypes.CDLL(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    lib.rdc_init.argtypes = [ctypes.c_uint64]
    lib.rdc_connect.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)] + [
        ctypes.c_char_p
    ] * 3
    lib.rdc_disconnect.argtypes = [ctypes.c_void_p]
    lib.rdc_test_case_run.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    assert lib.rdc_init(0) == RDC_ST_OK
    return lib


class DiagRun:
    """`rdci diag -r 1` on a group, in the background. result(): (status, output)."""

    def __init__(self, d, gid):
        cmd = [d.rdci_path, "diag", "-g", str(gid), "-r", "1", "-u"]
        cmd += ["--host", f"127.0.0.1:{d.port}"]
        self.proc = subprocess.Popen(
            cmd, env=d.rdci_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

    def running(self):
        return self.proc.poll() is None

    def result(self):
        out, _ = self.proc.communicate(timeout=(DELAY_S + 60) * FACTOR)
        return self.proc.returncode, out  # rdci exits with the RDC status of an error


class CaseRun:
    """rdc_test_case_run() of the node topology test on a group, on a thread with a
    handle of its own. result(): (status, "")."""

    def __init__(self, d, gid):
        self.lib = _lib()
        self.handle = ctypes.c_void_p()
        host = f"127.0.0.1:{d.port}".encode()
        assert self.lib.rdc_connect(host, ctypes.byref(self.handle), None, None, None) == RDC_ST_OK
        self.status = None
        self.thread = threading.Thread(target=self._run, args=(gid,), daemon=True)
        self.thread.start()

    def _run(self, gid):
        result = ctypes.create_string_buffer(TEST_RESULT_SIZE)
        self.status = self.lib.rdc_test_case_run(
            self.handle, gid, RDC_DIAG_NODE_TOPOLOGY, b"", 0, result, None
        )

    def running(self):
        return self.thread.is_alive()

    def result(self):
        self.thread.join((DELAY_S + 60) * FACTOR)
        assert not self.thread.is_alive(), "rdc_test_case_run() did not return"
        assert self.lib.rdc_disconnect(self.handle) == RDC_ST_OK
        return self.status, ""


RUNS = {"diag": DiagRun, "test_case": CaseRun}


@pytest.mark.parametrize("third", sorted(RUNS))
@pytest.mark.parametrize("holders", sorted(RUNS))
def test_a_third_diagnostic_is_rejected(rdcd, holders, third):
    gid = _pair_group(rdcd)
    since = len(rdcd.calls())
    running = [RUNS[holders](rdcd, gid) for _ in range(MAX_RUNS)]
    _wait_for_link_reads(rdcd, since, 0, MAX_RUNS)
    status, out = RUNS[third](rdcd, gid).result()
    assert status == RDC_ST_MAX_LIMIT, out
    assert _link_reads(rdcd, since, 0) == MAX_RUNS, "the rejected run reached the GPUs"
    assert all(run.running() for run in running), "the rejected run waited for a slot"
    for run in running:
        status, out = run.result()
        assert status == RDC_ST_OK, out


def test_runs_that_ended_or_failed_free_their_slots(rdcd):
    gid = _pair_group(rdcd)
    # rdcd returns the error of a diagnostic that fails, and the client reports it.
    for _ in range(MAX_RUNS + 1):
        status, out = DiagRun(rdcd, MISSING_GROUP).result()
        assert status == RDC_ST_GROUP_NOT_FOUND, out
        status, _ = CaseRun(rdcd, MISSING_GROUP).result()
        assert status == RDC_ST_GROUP_NOT_FOUND
    status, out = DiagRun(rdcd, gid).result()
    assert status == RDC_ST_OK, out
    # None of these runs holds a slot any more: two at once are both admitted.
    running = [DiagRun(rdcd, gid), CaseRun(rdcd, gid)]
    for run in running:
        status, out = run.result()
        assert status == RDC_ST_OK, out


def test_a_run_of_a_killed_client_holds_its_slot_until_it_ends(rdcd):
    gid = _pair_group(rdcd)
    since = len(rdcd.calls())
    running = [DiagRun(rdcd, gid) for _ in range(MAX_RUNS)]
    _wait_for_link_reads(rdcd, since, 0, MAX_RUNS)
    running[0].proc.kill()
    running[0].proc.wait()
    # rdcd's handler of the killed client's run is still in the delayed read.
    status, out = DiagRun(rdcd, gid).result()
    assert status == RDC_ST_MAX_LIMIT, out
    status, out = running[1].result()
    assert status == RDC_ST_OK, out
    # Once its delayed read is done, the run ends and frees its slot: both slots take runs again.
    _wait_for_link_reads(rdcd, since, 1, MAX_RUNS)
    deadline = time.monotonic() + (DELAY_S + 10) * FACTOR
    while True:
        results = [run.result() for run in [DiagRun(rdcd, gid) for _ in range(MAX_RUNS)]]
        if RDC_ST_MAX_LIMIT not in [status for status, _ in results]:
            break
        assert time.monotonic() < deadline, "the killed client's run kept its slot"
        time.sleep(0.1)
    for status, out in results:
        assert status == RDC_ST_OK, out


def test_a_diagnostic_without_rdcd_fails(tools):
    rdci = tools[1]
    env = {
        k: v for k, v in os.environ.items() if k != "LD_PRELOAD" and not k.startswith("RDC_FAKE_")
    }
    cmd = [rdci, "diag", "-g", "1", "-r", "1", "-u", "--host", f"127.0.0.1:{_free_port()}"]
    r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60 * FACTOR)
    assert r.returncode == RDC_ST_CLIENT_ERROR, r.stdout + r.stderr


NULL_CONFIG = """
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.rdc_connect.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)] + [ctypes.c_char_p] * 3
for name in ("rdc_test_case_run", "rdc_diagnostic_run"):
    getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_char_p,
                                   ctypes.c_size_t, ctypes.c_void_p, ctypes.c_void_p]
handle = ctypes.c_void_p()
assert lib.rdc_init(ctypes.c_uint64(0)) == 0
assert lib.rdc_connect(sys.argv[2].encode(), ctypes.byref(handle), None, None, None) == 0
results = ctypes.create_string_buffer(1 << 24)  # more than sizeof(rdc_diag_response_t)
group = int(sys.argv[3])
print(lib.rdc_test_case_run(handle, group, 3, None, 0, results, None),
      lib.rdc_diagnostic_run(handle, group, 1, None, 0, results, None))
"""


def test_a_diagnostic_without_a_config_runs(rdcd):
    # The library takes a null config (rdcd's embedded library does too); in a process of its
    # own, since the client crashed on it.
    gid = _pair_group(rdcd, delayed=False)
    lib = os.path.join(PREFIX, "lib", "librdc_bootstrap.so")
    cmd = [sys.executable, "-c", NULL_CONFIG, lib, f"127.0.0.1:{rdcd.port}", str(gid)]
    r = subprocess.run(cmd, env=rdcd.rdci_env, capture_output=True, text=True, timeout=60 * FACTOR)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert r.stdout.split()[-2:] == [str(RDC_ST_OK)] * 2, r.stdout[-2000:]
