"""GPU-less end-to-end tests: rdcd and rdci against the fake amd-smi.

The fake (tests/fake_amdsmi, installed as
share/rdc/rdctst_tests/fake_gpu/librdc_fake_amdsmi.so) is preloaded into rdcd
and answers every amdsmi_* call for simulated GPUs, so these tests run on any
host. The value tests compare what rdci shows for each GPU with the fake's
value table; the fault tests check that rdcd survives each failing call.

Environment:
  RDC_TEST_PREFIX       install prefix of the RDC under test
                        (default: $ROCM_DIR, then /opt/rocm)
  RDC_FAKE_LIB          librdc_fake_amdsmi.so to use (default: from the prefix,
                        then next to this file)
  RDC_CI_REQUIRE_FAKE   1: fail instead of skip when rdcd, rdci, the fake or nm
                        is missing
  RDC_TEST_TIME_FACTOR  multiplies every timeout and wait, for sanitizer builds
                        (default 1)

rdcd allows one instance per host (it locks /var/run/rdcd.lock as root and
/tmp/rdcd.lock otherwise), so this module refuses to run while another rdcd
holds that lock. RDC CI runs it in its own pytest invocation.
"""

import fcntl
import json
import os
import re
import shutil
import signal
import socket
import struct
import subprocess
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fake_gpu_golden as golden  # noqa: E402

PREFIX = os.environ.get("RDC_TEST_PREFIX") or os.environ.get("ROCM_DIR", "/opt/rocm")
REQUIRE = os.environ.get("RDC_CI_REQUIRE_FAKE") == "1"
FACTOR = float(os.environ.get("RDC_TEST_TIME_FACTOR", "1"))
FAKE_NAME = "librdc_fake_amdsmi.so"
NUM_GPUS = 4
DMON_FIELDS = [100, 101, 200, 201, 300, 500, 501, 502]
# rdcd refuses to start while either lock is held (it takes the first as root).
RDCD_LOCKS = ("/var/run/rdcd.lock", "/tmp/rdcd.lock")

# Faults that crash rdcd on this tree, with the code path at fault. Each case
# must still crash; the change that fixes one must remove its entry (its CI
# fails until it does).
DEFERRED_CRASHES: dict = {}

# amdsmi_* calls RDC links that this module does not reach, with the reason.
# The coverage test fails once one of them is reached, so the list stays true.
UNREACHED_APIS = {
    "amdsmi_get_afids_from_cper": "needs CPER records; the fake reports none by default",
    "amdsmi_get_gpu_accelerator_partition_profile_config": "partition profile query",
    "amdsmi_get_gpu_compute_process_gpus": "needs processes; the fake reports none by default",
    "amdsmi_get_lib_version": "component version query, not exposed by rdci",
    "amdsmi_get_power_cap_info": "rdci config has no --host option",
    "amdsmi_set_power_cap": "rdci config has no --host option",
    "amdsmi_set_gpu_clk_limit": "rdci config has no --host option",
    "amdsmi_set_gpu_perf_level": "rdci config has no --host option",
    "amdsmi_reset_gpu": "policy reset action needs a breached threshold",
    "amdsmi_stop_gpu_event_notification": "rdci offers no notification event field",
}

# A failing amdsmi_init() makes rdcd hang, then crash on SIGTERM; the rdcd
# start/stop fix adds that case. amdsmi_shut_down() has no failure to inject.
LIFECYCLE_APIS = {"amdsmi_init", "amdsmi_shut_down"}


def _missing(what):
    message = f"{what} not found (RDC_TEST_PREFIX={PREFIX})"
    if REQUIRE:
        pytest.fail(message)
    pytest.skip(message)


def _first_file(*paths):
    for path in paths:
        if path and os.path.isfile(path):
            return os.path.realpath(path)
    return None


def _rdcd_lock_problem(path, takes_it):
    """Why rdcd could not start because of lock file `path` (None if it could).

    Mirrors rdcd: it refuses to start if either lock file is locked, then takes
    its own (/var/run/rdcd.lock as root, /tmp/rdcd.lock otherwise).
    """
    if not os.path.exists(path):
        return None
    try:
        fd = os.open(path, os.O_RDONLY)
    except PermissionError:
        fd = None  # rdcd cannot read it either and does not count it as locked
    if fd is not None:
        try:
            query = struct.pack("hhqqi", fcntl.F_RDLCK, os.SEEK_SET, 0, 0, 0)
            locked = (
                struct.unpack("hhqqi", fcntl.fcntl(fd, fcntl.F_GETLK, query))[0] != fcntl.F_UNLCK
            )
        finally:
            os.close(fd)
        if locked:
            return f"another rdcd holds {path}"
    if takes_it and not os.access(path, os.W_OK):
        return f"{path} belongs to another user"
    return None


@pytest.fixture(scope="module")
def tools():
    rdcd = _first_file(os.path.join(PREFIX, "bin", "rdcd"))
    rdci = _first_file(os.path.join(PREFIX, "bin", "rdci"))
    fake = _first_file(
        os.environ.get("RDC_FAKE_LIB"),
        os.path.join(PREFIX, "share", "rdc", "rdctst_tests", "fake_gpu", FAKE_NAME),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), FAKE_NAME),
    )
    for path, what in ((rdcd, "rdcd"), (rdci, "rdci"), (fake, FAKE_NAME)):
        if path is None:
            _missing(what)
    for lock in RDCD_LOCKS:
        problem = _rdcd_lock_problem(lock, takes_it=(lock == RDCD_LOCKS[0]) == (os.geteuid() == 0))
        if problem:
            pytest.fail(f"{problem}; rdcd cannot start until that is resolved")
    return rdcd, rdci, fake


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _listening(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


class Rdcd:
    """One rdcd with the fake preloaded, for one fake configuration.

    Faults (RDC_FAKE_FAIL, RDC_FAKE_SENTINEL) take effect after arm(), so a
    test can set up groups first.
    """

    def __init__(self, tools, tmp_path, gpus=NUM_GPUS, **fake_env):
        self.rdcd_path, self.rdci_path, self.fake = tools
        self.call_log = tmp_path / "fake_calls.log"
        self.out_log = tmp_path / "rdcd.log"
        self.armed = tmp_path / "armed"
        # rdci must not load the fake or write to the call log.
        self.rdci_env = {
            k: v
            for k, v in os.environ.items()
            if k != "LD_PRELOAD" and not k.startswith("RDC_FAKE_")
        }
        self.env = dict(self.rdci_env)
        self.env.update(
            LD_PRELOAD=self.fake,
            RDC_FAKE_GPUS=str(gpus),
            RDC_FAKE_LOG=str(self.call_log),
            RDC_FAKE_ARM=str(self.armed),
            RDC_FAKE_EVENT_WAIT_MS="100",
        )
        self.env.update({k: str(v) for k, v in fake_env.items()})
        self.proc = None
        self.port = None

    def start(self):
        for _ in range(5):
            self.port = _free_port()
            log_start = self.out_log.stat().st_size if self.out_log.exists() else 0
            with open(self.out_log, "a") as out:
                self.proc = subprocess.Popen(
                    [self.rdcd_path, "-u", "-p", str(self.port)],
                    env=self.env,
                    stdout=out,
                    stderr=subprocess.STDOUT,
                )
            deadline = time.monotonic() + 10 * FACTOR
            while time.monotonic() < deadline and self.proc.poll() is None:
                if _listening(self.port):
                    return self
                time.sleep(0.1)
            if self.proc.poll() is None:
                self.kill()
                pytest.fail(f"rdcd did not listen within {10 * FACTOR:.0f} s\n{self.output()}")
            this_run = self.out_log.read_text(errors="replace")[log_start:]
            if "Address already in use" not in this_run:
                pytest.fail(f"rdcd exited with {self.proc.returncode} at start\n{this_run[-4000:]}")
            # Another process took the port between the probe and rdcd's bind.
        pytest.fail(f"rdcd found its port taken 5 times\n{self.output()}")

    def output(self):
        try:
            return self.out_log.read_text(errors="replace")[-4000:]
        except OSError:
            return ""

    def check_alive(self, what):
        code = self.proc.poll()
        if code is not None:
            pytest.fail(f"rdcd exited with {code} during {what}\n{self.output()}")

    def rdci(self, *args, timeout=30):
        cmd = [self.rdci_path, *args, "-u", "--host", f"127.0.0.1:{self.port}"]
        try:
            r = subprocess.run(
                cmd, env=self.rdci_env, capture_output=True, text=True, timeout=timeout * FACTOR
            )
        except subprocess.TimeoutExpired:
            pytest.fail(f"rdci {' '.join(args)} timed out")
        self.check_alive("rdci " + " ".join(args))
        return r

    def ok(self, *args, timeout=30):
        r = self.rdci(*args, timeout=timeout)
        assert r.returncode == 0, f"rdci {' '.join(args)} -> {r.returncode}\n{r.stdout}{r.stderr}"
        return r.stdout

    def arm(self):
        self.armed.touch()

    def wait_for_failed_call(self, api, timeout):
        """Wait until a rule has failed `api` (calls from background threads come later)."""
        deadline = time.monotonic() + timeout * FACTOR
        while True:
            if any(a == api and failed for a, _, failed in self.calls()):
                return True
            if time.monotonic() > deadline or self.proc.poll() is not None:
                return False
            time.sleep(0.1)

    def stop(self):
        """SIGTERM rdcd and return its exit code."""
        if self.proc is None or self.proc.poll() is not None:
            return None if self.proc is None else self.proc.returncode
        self.proc.send_signal(signal.SIGTERM)
        try:
            return self.proc.wait(timeout=15 * FACTOR)
        except subprocess.TimeoutExpired:
            self.kill()
            pytest.fail(f"rdcd did not exit within {15 * FACTOR:.0f} s of SIGTERM\n{self.output()}")

    def kill(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def calls(self):
        """amdsmi_* calls the fake saw: (api, gpu, failed by a rule)."""
        if not self.call_log.exists():
            return []
        calls = []
        for line in self.call_log.read_text().splitlines():
            parts = line.split()
            if parts[2:] not in ([], ["!"]) or len(parts) < 2 or not parts[1].lstrip("-").isdigit():
                pytest.fail(f"malformed RDC_FAKE_LOG line: {line!r}")
            calls.append((parts[0], int(parts[1]), parts[2:] == ["!"]))
        return calls


@pytest.fixture
def rdcd(tools, tmp_path):
    """A fresh rdcd for one test; its teardown asserts a clean exit."""
    d = Rdcd(tools, tmp_path).start()
    yield d
    code = d.stop()
    assert code == 0, f"rdcd exited with {code} after SIGTERM\n{d.output()}"


def _created_id(stdout):
    m = re.search(r"group ID (\d+)", stdout)
    assert m, f"no group ID in: {stdout}"
    return m.group(1)


def _all_gpus_group(d, name="fake_all"):
    gid = _created_id(d.ok("group", "-c", name))
    d.ok("group", "-g", gid, "-a", ",".join(str(g) for g in range(NUM_GPUS)))
    return gid


def _field_names(d):
    names = {}
    for line in d.ok("dmon", "-l").splitlines():
        m = re.match(r"\s*(\d+)\s+RDC_(?:FI|EVNT)_(\w+)\s*:", line)
        if m:
            names[int(m.group(1))] = m.group(2)
    return names


def _dmon_rows(stdout):
    """Parse `rdci dmon` output into {gpu: {column: value}}, last sample wins."""
    header, rows = None, {}
    for line in stdout.splitlines():
        cols = line.split()
        if not cols:
            continue
        if cols[0] == "GPU":
            header = cols
        elif header and cols[0].isdigit():
            assert len(cols) == len(header), f"malformed dmon row {line!r} in:\n{stdout}"
            rows[int(cols[0])] = dict(zip(header[1:], cols[1:]))
    return rows


# ---------------------------------------------------------------------------
# The value table: a mix-up of GPUs or fields must change the value
# ---------------------------------------------------------------------------
def test_fake_readings_are_distinct():
    for gpu in range(8):
        row = [golden.reading(name, gpu) for name in golden.TABLE]
        assert len(set(row)) == len(row), f"GPU {gpu} has equal readings: {row}"
    for name in golden.TABLE:
        column = [golden.reading(name, gpu) for gpu in range(8)]
        assert len(set(column)) == len(column), f"{name} repeats across GPUs: {column}"


# ---------------------------------------------------------------------------
# The fake is what rdcd talks to, and RDC comes from the tested prefix
# ---------------------------------------------------------------------------
def test_fake_and_rdc_libraries_come_from_the_tested_prefix(rdcd):
    rdcd.ok("discovery", "-l")
    with open(f"/proc/{rdcd.proc.pid}/maps") as f:
        loaded = {line.split()[-1] for line in f if line.count(" ") >= 5}
    assert rdcd.fake in loaded, "the fake amd-smi is not mapped into rdcd"
    prefix = os.path.realpath(PREFIX)
    stray = [
        p
        for p in loaded
        if "librdc" in os.path.basename(p)
        and p != rdcd.fake
        and not os.path.realpath(p).startswith(prefix + os.sep)
    ]
    assert not stray, f"RDC libraries loaded from outside {prefix}: {stray}"
    assert rdcd.calls(), "RDC_FAKE_LOG is empty: amdsmi_* calls did not reach the fake"


# ---------------------------------------------------------------------------
# Values: every GPU reports its own readings, in RDC's documented units
# ---------------------------------------------------------------------------
def test_discovery_lists_every_fake_gpu(rdcd):
    out = rdcd.ok("discovery", "-l")
    assert f"{NUM_GPUS} GPUs found" in out
    for gpu in range(NUM_GPUS):
        assert re.search(rf"^{gpu}\s+{re.escape(golden.device_name(gpu))}\s*$", out, re.M), out


def test_discovery_lists_each_partition(tools, tmp_path):
    d = Rdcd(tools, tmp_path, gpus=2, RDC_FAKE_PARTITIONS=2).start()
    try:
        out = d.ok("discovery", "-l")
        assert "4 GPUs found" in out, out
        for index in range(4):
            name = golden.device_name(index // 2)
            assert re.search(rf"^{index}\s+{re.escape(name)}\s*$", out, re.M), out
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code}\n{d.output()}"


def test_dmon_reports_each_gpus_own_values(rdcd):
    names = _field_names(rdcd)
    gpus = ",".join(str(g) for g in range(NUM_GPUS))
    fields = ",".join(str(f) for f in DMON_FIELDS)
    out = rdcd.ok("dmon", "-e", fields, "-i", gpus, "-c", "1", "-d", "500")
    rows = _dmon_rows(out)
    assert sorted(rows) == list(range(NUM_GPUS)), out
    for gpu in range(NUM_GPUS):
        for field in DMON_FIELDS:
            got, want = rows[gpu][names[field]], str(golden.expected(field, gpu))
            assert got == want, (
                f"GPU {gpu} {names[field]}: rdci shows {got}, expected {want}\n{out}"
            )


def test_dmon_on_a_group_and_field_group(rdcd):
    names = _field_names(rdcd)
    gid = _all_gpus_group(rdcd)
    fgid = _created_id(rdcd.ok("fieldgroup", "-c", "fake_fg", "-f", "100,201,300"))
    out = rdcd.ok("dmon", "-g", gid, "-f", fgid, "-c", "1", "-d", "500")
    rows = _dmon_rows(out)
    assert sorted(rows) == list(range(NUM_GPUS)), out
    for gpu in range(NUM_GPUS):
        for field in (100, 201, 300):
            assert rows[gpu][names[field]] == str(golden.expected(field, gpu)), out
    rdcd.ok("fieldgroup", "-d", fgid)
    rdcd.ok("group", "-d", gid)
    assert "fake_all" not in rdcd.ok("group", "-l")


def _stat_line(out, label):
    m = re.search(rf"{re.escape(label)}\s*\|\s*Max: (\S+) Min: (\S+) Avg: (\S+)", out)
    assert m, f"no '{label}' line in:\n{out}"
    try:
        return tuple(float(v) for v in m.groups())
    except ValueError:
        pytest.fail(f"'{label}' has no numbers:\n{out}")


def test_job_stats_on_a_single_gpu(rdcd):
    gid = _created_id(rdcd.ok("group", "-c", "fake_job"))
    rdcd.ok("group", "-g", gid, "-a", "0")
    rdcd.ok("stats", "-s", "fake_job_1", "-g", gid)
    time.sleep(2 * FACTOR)
    out = rdcd.ok("stats", "-j", "fake_job_1")
    rdcd.ok("stats", "-x", "fake_job_1")
    rdcd.ok("stats", "-r", "fake_job_1")
    for label, reading in (
        ("Power Usage (Watts)", golden.reading("SOCKET_POWER_W", 0)),
        ("GPU Clock (MHz)", golden.reading("GFX_CLK_MHZ", 0)),
        ("Memory Clock (MHz)", golden.reading("MEM_CLK_MHZ", 0)),
        ("GPU Utilization (%)", golden.reading("GFX_ACTIVITY_PCT", 0)),
    ):
        assert _stat_line(out, label) == (reading, reading, reading), (label, out)


def test_policy_lifecycle(rdcd):
    gid = _all_gpus_group(rdcd)
    rdcd.ok("policy", "-g", gid, "-T", "95", "-A", "0")
    out = rdcd.ok("policy", "-g", gid, "--get")
    assert re.search(r"Temperature Limit\s*\|\s*95\s*\|\s*Notify", out), out
    rdcd.ok("policy", "-g", gid, "--clear")
    assert "Temperature Limit" not in rdcd.ok("policy", "-g", gid, "--get")


# ---------------------------------------------------------------------------
# Commands that must succeed (their output is checked where it is fixed)
# ---------------------------------------------------------------------------
def test_topology_and_link_commands(rdcd):
    for gpu in range(NUM_GPUS):
        rdcd.ok("topo", "-i", str(gpu))
    rdcd.ok("link", "-l")


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------
def test_rdcd_restarts_cleanly(tools, tmp_path):
    for run in (1, 2):
        d = Rdcd(tools, tmp_path).start()
        try:
            assert f"{NUM_GPUS} GPUs found" in d.ok("discovery", "-l")
        finally:
            code = d.stop()
        assert code == 0, f"run {run}: rdcd exited with {code}\n{d.output()}"


# ---------------------------------------------------------------------------
# Load: an idle rdcd must not poll amd-smi in a tight loop
# ---------------------------------------------------------------------------
IDLE_WINDOW_S = 5
# An idle rdcd makes about 5 calls: one event-notification wait per second.
# Without that pause the fake's 100 ms wait cap would allow 50.
IDLE_MAX_CALLS = 15


def test_idle_rdcd_call_rate(rdcd):
    rdcd.ok("discovery", "-l")
    before = len(rdcd.calls())
    time.sleep(IDLE_WINDOW_S)
    calls = rdcd.calls()[before:]
    assert len(calls) <= IDLE_MAX_CALLS, (
        f"idle rdcd made {len(calls)} amd-smi calls in {IDLE_WINDOW_S} s: "
        f"{sorted({api for api, _, _ in calls})}"
    )


def test_pytest_timeout_is_active(pytestconfig):
    """CI relies on pytest-timeout to name a hung test instead of killing the job."""
    if not REQUIRE:
        pytest.skip("only checked in CI (RDC_CI_REQUIRE_FAKE=1)")
    assert pytestconfig.pluginmanager.hasplugin("timeout"), "pytest-timeout is not installed"
    assert (pytestconfig.getoption("timeout") or 0) > 0, "pytest-timeout has no --timeout"


# ---------------------------------------------------------------------------
# Embedded mode, in a child process so a hang or crash cannot take pytest down
# ---------------------------------------------------------------------------
_EMBEDDED = r"""
import ctypes, json, sys
lib = ctypes.CDLL(sys.argv[1])
class Attr(ctypes.Structure):
    _fields_ = [("device_name", ctypes.c_char * 256), ("device_id", ctypes.c_uint64),
                ("num_of_compute_units", ctypes.c_uint32),
                ("target_graphics_version", ctypes.c_uint64)]
def call(name, *args):
    st = getattr(lib, name)(*args)
    if st != 0:
        print(json.dumps({"failed": name, "status": st})); sys.exit(1)
handle = ctypes.c_void_p()
call("rdc_init", ctypes.c_uint64(0))
call("rdc_start_embedded", 0, ctypes.byref(handle))
ids = (ctypes.c_uint32 * 128)()
count = ctypes.c_uint32(0)
call("rdc_device_get_all", handle, ids, ctypes.byref(count))
names = []
for i in range(count.value):
    attr = Attr()
    call("rdc_device_get_attributes", handle, ids[i], ctypes.byref(attr))
    names.append(attr.device_name.decode())
call("rdc_stop_embedded", handle)
call("rdc_shutdown")
print(json.dumps({"ids": list(ids[: count.value]), "names": names}))
"""


def test_embedded_start_read_stop(tools):
    bootstrap = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
    if bootstrap is None:
        _missing("librdc_bootstrap.so")
    env = dict(os.environ, LD_PRELOAD=tools[2], RDC_FAKE_GPUS=str(NUM_GPUS))
    try:
        r = subprocess.run(
            [sys.executable, "-c", _EMBEDDED, bootstrap],
            env=env,
            capture_output=True,
            text=True,
            timeout=60 * FACTOR,
        )
    except subprocess.TimeoutExpired:
        pytest.fail("embedded start/read/stop hung (child killed)")
    assert r.returncode == 0, f"embedded child exited {r.returncode}\n{r.stdout}{r.stderr[-3000:]}"
    result = json.loads(r.stdout.strip().splitlines()[-1])
    assert result["ids"] == list(range(NUM_GPUS))
    assert result["names"] == [golden.device_name(g) for g in range(NUM_GPUS)]


# ---------------------------------------------------------------------------
# Every amdsmi_* call RDC makes: implemented by the fake, reached, and failing
# ---------------------------------------------------------------------------
REQUIRED_BINARIES = ("rdcd", "rdci", "librdc.so", "librdc_bootstrap.so")
MIN_RDC_APIS = 40  # RDC references about 60; fewer means a binary was missed


def _nm(path, defined):
    flag = "--defined-only" if defined else "--undefined-only"
    out = subprocess.run(["nm", "-D", flag, path], capture_output=True, text=True).stdout
    return {line.split()[-1].split("@")[0] for line in out.splitlines() if " amdsmi_" in line}


def _rdc_binaries():
    paths = [os.path.join(PREFIX, "bin", b) for b in ("rdcd", "rdci")]
    for d in (os.path.join(PREFIX, "lib"), os.path.join(PREFIX, "lib", "rdc")):
        if os.path.isdir(d):
            paths += [
                os.path.join(d, f)
                for f in sorted(os.listdir(d))
                if f.startswith("librdc") and ".so" in f
            ]
    return [p for p in paths if os.path.isfile(p) and not os.path.islink(p)]


def _rdc_amdsmi_apis():
    if shutil.which("nm") is None:
        return []
    apis = set()
    for path in _rdc_binaries():
        apis |= _nm(path, defined=False)
    return sorted(apis)


RDC_APIS = _rdc_amdsmi_apis()


def test_fake_exports_every_amdsmi_call_rdc_makes(tools):
    if shutil.which("nm") is None:
        _missing("nm (binutils)")
    scanned = {os.path.basename(p).split(".so")[0] for p in _rdc_binaries()}
    missing_binaries = [b for b in REQUIRED_BINARIES if b.split(".so")[0] not in scanned]
    assert not missing_binaries, f"not scanned for amdsmi_* references: {missing_binaries}"
    assert len(RDC_APIS) >= MIN_RDC_APIS, f"only {len(RDC_APIS)} amdsmi_* references found"
    missing = sorted(set(RDC_APIS) - _nm(tools[2], defined=True))
    assert not missing, f"the fake does not implement: {missing}"


def _workload_commands(field_ids):
    """rdci commands that together reach the amdsmi_* calls RDC makes."""
    gpus = ",".join(str(g) for g in range(NUM_GPUS))
    half = len(field_ids) // 2
    dmon = ("-i", gpus, "-c", "1", "-d", "200")
    return [
        ("discovery", "-l"),
        ("dmon", "-e", ",".join(map(str, field_ids[:half])), *dmon),
        ("dmon", "-e", ",".join(map(str, field_ids[half:])), *dmon),
        ("topo", "-i", "0"),
        ("topo", "-i", "1"),
        ("link", "-l"),
        ("health", "-g", "{gid}", "-s", "a"),
        ("health", "-g", "{gid}", "-c"),
        ("health", "-g", "{gid}", "--clear"),
        ("stats", "-s", "cov", "-g", "{gid}"),
        ("stats", "-j", "cov"),
        ("stats", "-x", "cov"),
        ("stats", "-r", "cov"),
        ("policy", "-g", "{gid}", "-T", "95", "-A", "0"),
        ("policy", "-g", "{gid}", "--get"),
        ("policy", "-g", "{gid}", "--clear"),
        ("diag", "-g", "{gid}", "-r", "1"),
    ]


def _run(d, cmd, gid):
    d.rdci(*(arg.replace("{gid}", gid) for arg in cmd), timeout=60)


@pytest.fixture(scope="module")
def reach(tools, tmp_path_factory):
    """Run the whole workload once without faults.

    Returns (commands, {api: index of the last command that reached it},
    {api: GPUs it was called for while commands ran}, every API called). An API
    called only while rdcd starts or sets up the group maps to -1.
    """
    d = Rdcd(tools, tmp_path_factory.mktemp("reach")).start()
    last, at_start, in_commands = {}, {}, {}
    try:
        commands = _workload_commands(sorted(_field_names(d)))
        gid = _all_gpus_group(d, "work")
        for api, gpu, _ in d.calls():
            last[api] = -1
            at_start.setdefault(api, set()).add(gpu)
        for index, cmd in enumerate(commands):
            seen = len(d.calls())
            _run(d, cmd, gid)
            for api, gpu, _ in d.calls()[seen:]:
                last[api] = index
                in_commands.setdefault(api, set()).add(gpu)
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code} after the fault-free workload\n{d.output()}"
    # The GPUs a fault can target: those the API saw while armed.
    gpus = {api: in_commands.get(api) or at_start[api] for api in last}
    return commands, last, gpus, {api for api, _, _ in d.calls()}


@pytest.mark.parametrize("poison", [False, True], ids=["fail", "poison"])
@pytest.mark.parametrize("api", RDC_APIS or ["<none>"])
def test_rdcd_survives_a_failing_call(tools, tmp_path, reach, api, poison):
    """The call fails (with poison, it also fills its outputs with 0xA5).

    Only survival and a clean exit are checked here; the value and status RDC
    reports for a failed read are checked by the tests of the code that reads it.
    """
    if api == "<none>":
        _missing("nm (binutils)")
    commands, last, gpus, _ = reach
    if api in LIFECYCLE_APIS:
        pytest.skip(f"{api}: see LIFECYCLE_APIS")
    if api not in last:
        pytest.skip(f"{api} is not reached (see UNREACHED_APIS)")
    case = f"{api}@poison" if poison else api
    # Fail the call on a GPU it was seen on (prefer GPU 1), or everywhere if it
    # takes no GPU.
    seen = sorted(g for g in gpus[api] if g >= 0)
    rule = f"{api}@{1 if 1 in seen else seen[0]}" if seen else api
    d = Rdcd(tools, tmp_path, RDC_FAKE_FAIL=rule, RDC_FAKE_POISON=int(poison))
    if last[api] < 0:
        d.arm()  # called only while rdcd starts or sets up: fail it from the start
    fired = False
    try:
        d.start()
        out = d.rdci("group", "-c", "fault").stdout
        gid = re.search(r"group ID (\d+)", out).group(1) if "group ID" in out else "1"
        d.rdci("group", "-g", gid, "-a", ",".join(str(g) for g in range(NUM_GPUS)))
        d.arm()
        try:
            # Every command up to the last one that reached the API, so the
            # commands it depends on (stats -s before stats -j) run too.
            for cmd in commands[: last[api] + 1] or [("discovery", "-l")]:
                _run(d, cmd, gid)
            fired = d.wait_for_failed_call(api, timeout=5)
            d.check_alive(f"{rule} failing")
        except pytest.fail.Exception as e:
            if case not in DEFERRED_CRASHES or "rdcd exited" not in str(e):
                raise
            assert any(a == api and failed for a, _, failed in d.calls()), f"{rule} never fired"
            return
        if case in DEFERRED_CRASHES:
            pytest.fail(f"{case} no longer crashes rdcd; remove it from DEFERRED_CRASHES")
    finally:
        code = d.stop()
    assert fired, f"{rule} never fired: nothing called {api} once the fault was armed"
    assert code == 0, f"rdcd exited with {code} with {rule} failing\n{d.output()}"


def test_every_amdsmi_call_rdc_makes_is_reached(reach):
    if not RDC_APIS:
        _missing("nm (binutils)")
    last, called = reach[1], reach[3]
    shutdown_only = sorted(set(RDC_APIS) & called - set(last) - LIFECYCLE_APIS)
    assert not shutdown_only, f"called only while rdcd stops, so never failed: {shutdown_only}"
    deferred = {case.split("@")[0] for case in DEFERRED_CRASHES}
    assert not deferred & set(UNREACHED_APIS), "an API cannot be both deferred and unreached"
    not_reached = sorted(set(RDC_APIS) - called - set(UNREACHED_APIS))
    assert not not_reached, f"no workload reaches these, extend it or list them: {not_reached}"
    stale = sorted(set(UNREACHED_APIS) & called)
    assert not stale, f"reached now, remove from UNREACHED_APIS: {stale}"


# ---------------------------------------------------------------------------
# Packaging: the fake never ships in, or is referenced by, runtime files
# ---------------------------------------------------------------------------
def test_runtime_files_do_not_contain_the_fake():
    binaries = _rdc_binaries()
    if not binaries:
        _missing("RDC runtime binaries")
    for d in (
        os.path.join(PREFIX, "lib"),
        os.path.join(PREFIX, "lib", "rdc"),
        os.path.join(PREFIX, "bin"),
    ):
        if os.path.isdir(d):
            assert not [f for f in os.listdir(d) if "fake" in f], f"fake file in {d}"
    for path in binaries:
        with open(path, "rb") as f:
            assert b"RDC_FAKE_" not in f.read(), f"{path} references the fake"
