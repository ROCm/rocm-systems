"""rdci diag against the fake amd-smi: what the AMD SMI checks (compute processes,
node topology, GPU parameters) report for each fake configuration, and a
diagnostic run in rdcd and in rdci with a 2 MB stack.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import contextlib
import re
import resource
import subprocess
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (tools is a fixture)
    FACTOR,
    NUM_GPUS,
    Rdcd,
    _created_id,
    _free_port,
    _listening,
    tools,
)

PROCESSES = "No compute process"
TOPOLOGY = "Node topology check"
PARAMETERS = "GPU parameters check"
CHECKS = (PROCESSES, TOPOLOGY, PARAMETERS)
DETAILS = " =============== Diagnostic Details =================="
# The fake's limits: maximum, critical and emergency temperature of 3 sensors,
# and 4 voltage limits. Its readings are in range on GPUs 0 to 7.
LIMITS = 13
NO_LIMIT = "No temperature or voltage limit could be compared."
STACK_LIMIT = 2 << 20  # a diagnostic response is about 7 MB


def _small_stack():
    """preexec_fn: a 2 MB stack, which threads also take as their default size, and no core dump."""
    resource.setrlimit(
        resource.RLIMIT_STACK, (STACK_LIMIT, resource.getrlimit(resource.RLIMIT_STACK)[1])
    )
    # With a limit of 1, not 0, the kernel also skips a dump piped to a handler such as apport.
    # A hard limit of 0 cannot be raised to 1.
    if resource.getrlimit(resource.RLIMIT_CORE)[1] != 0:
        resource.setrlimit(resource.RLIMIT_CORE, (1, 1))


def _start_with_small_stack(d):
    """Rdcd.start() with rdcd under _small_stack.

    A copy of test_fake_gpu.py's Rdcd.start() with preexec_fn added (that file is
    not changed here); keep the two alike.
    """
    hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
    if hard != resource.RLIM_INFINITY and hard < STACK_LIMIT:
        pytest.skip(f"the stack's hard limit, {hard} bytes, is below {STACK_LIMIT}")
    for _ in range(5):
        d.port = _free_port()
        log_start = d.out_log.stat().st_size if d.out_log.exists() else 0
        with open(d.out_log, "a") as out:
            d.proc = subprocess.Popen(
                [d.rdcd_path, "-u", "-p", str(d.port)],
                env=d.env,
                stdout=out,
                stderr=subprocess.STDOUT,
                preexec_fn=_small_stack,
            )
        deadline = time.monotonic() + 10 * FACTOR
        while time.monotonic() < deadline and d.proc.poll() is None:
            if _listening(d.port):
                return d
            time.sleep(0.1)
        if d.proc.poll() is None:
            d.kill()
            pytest.fail(f"rdcd did not listen within {10 * FACTOR:.0f} s\n{d.output()}")
        this_run = d.out_log.read_text(errors="replace")[log_start:]
        if "Address already in use" not in this_run:
            pytest.fail(f"rdcd exited with {d.proc.returncode} at start\n{this_run[-4000:]}")
        # Another process took the port between the probe and rdcd's bind.
    pytest.fail(f"rdcd found its port taken 5 times\n{d.output()}")


@contextlib.contextmanager
def _rdcd(tools, tmp_path, gpus=NUM_GPUS, small_stack=False, **fake_env):
    """A running rdcd with this fake configuration; it must exit cleanly at the end."""
    d = Rdcd(tools, tmp_path, gpus, **fake_env)
    # The compute process check leaves out rdcd's own process, and the fake's processes are
    # 1000 + i (RDC_FAKE_PROCS of them, one more per list call with RDC_FAKE_PROCS_GROW; 64 is
    # far more list calls than a case makes): start rdcd again while its process ID is one of them.
    procs = int(fake_env.get("RDC_FAKE_PROCS", 0))
    fake_pids = range(1000, 1000 + procs + 64) if procs else range(0)
    while True:
        if small_stack:
            _start_with_small_stack(d)
        else:
            d.start()
        if d.proc.pid not in fake_pids:
            break
        code = d.stop()
        assert code == 0, f"rdcd exited with {code} after SIGTERM\n{d.output()}"
    try:
        yield d
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code} after SIGTERM\n{d.output()}"


def _group(d, name, gpus):
    gid = _created_id(d.ok("group", "-c", name))
    if gpus:
        d.ok("group", "-g", gid, "-a", ",".join(map(str, gpus)))
    return gid


def _parse(out):
    """rdci diag's output: ({check: status}, the detail lines)."""
    head, found, details = out.partition(DETAILS + "\n")
    assert found, f"no diagnostic details in:\n{out}"
    status = {}
    for line in head.splitlines():
        m = re.fullmatch(r"(.+?):\s+(Pass|Skip|Warn|Fail)", line)
        if m:
            status[m.group(1)] = m.group(2)
    return status, details.splitlines()


def _diag(d, gid, *args):
    return _parse(d.ok("diag", "-g", gid, "-r", "1", *args, timeout=60))


def _info(check, text):
    """A check's info line as rdci prints it."""
    return f"{check + ':':<26}{text}"


def _in_range(gpus):
    return " ".join(f"GPU {g}: {LIMITS} limits in range." for g in gpus)


def _device_lines(details, check):
    """The " Device N" lines rdci prints under one check's info line, before the next check's."""
    lines = None
    for line in details:
        if line.startswith(check + ":"):
            lines = []
        elif lines is not None:
            if not line.startswith(" Device "):
                break
            lines.append(line)
    assert lines is not None, details
    return lines


def test_topology_pairs_the_groups_gpus_by_index(tools, tmp_path):
    with _rdcd(tools, tmp_path, gpus=3) as d:
        status, details = _diag(d, _group(d, "topology", [1, 2]))
    assert status[TOPOLOGY] == "Pass", status
    assert _info(TOPOLOGY, "1=>2 weight:15 2=>1 weight:15") in details, details


def test_topology_of_one_gpu_is_skipped(tools, tmp_path):
    with _rdcd(tools, tmp_path) as d:
        status, details = _diag(d, _group(d, "one", [2]))
    assert status[TOPOLOGY] == "Skip", status
    assert _info(TOPOLOGY, "No pair of available GPUs to check.") in details, details


def test_gpu_parameters_count_each_gpus_limits(tools, tmp_path):
    with _rdcd(tools, tmp_path) as d:
        status, details = _diag(d, _group(d, "parameters", range(NUM_GPUS)))
    assert status[PARAMETERS] == "Pass", status
    assert _info(PARAMETERS, _in_range(range(NUM_GPUS))) in details, details


@pytest.mark.parametrize(
    "unfilled, limits",
    [("amdsmi_get_temp_metric", 4), ("amdsmi_get_gpu_volt_metric", 9)],
    ids=["temperature", "voltage"],
)
def test_limits_left_unwritten_are_not_compared(tools, tmp_path, unfilled, limits):
    """A limit AMD SMI leaves at 0 is not compared (13 limits without the fault)."""
    with _rdcd(tools, tmp_path, gpus=2, RDC_FAKE_UNFILLED=unfilled) as d:
        gid = _group(d, "unfilled", [0, 1])
        d.arm()
        status, details = _diag(d, gid)
    assert status[PARAMETERS] == "Pass", status
    text = f"GPU 0: {limits} limits in range. GPU 1: {limits} limits in range."
    assert _info(PARAMETERS, text) in details, details


def test_a_gpu_with_nothing_written_is_skipped(tools, tmp_path):
    unfilled = "amdsmi_get_temp_metric,amdsmi_get_gpu_volt_metric"
    with _rdcd(tools, tmp_path, gpus=2, RDC_FAKE_UNFILLED=unfilled) as d:
        gid = _group(d, "unfilled", [0, 1])
        d.arm()
        status, details = _diag(d, gid)
    assert status[PARAMETERS] == "Skip", status
    text = "GPU 0: no limit could be compared. GPU 1: no limit could be compared."
    assert _info(PARAMETERS, text) in details, details
    assert f" Device 0 {NO_LIMIT}" in details, details
    assert f" Device 1 {NO_LIMIT}" in details, details


def test_a_gpu_that_cannot_be_read_is_skipped(tools, tmp_path):
    fail = "amdsmi_get_temp_metric@0,amdsmi_get_gpu_volt_metric@0"
    with _rdcd(tools, tmp_path, gpus=2, RDC_FAKE_FAIL=fail) as d:
        gid = _group(d, "unreadable", [0, 1])
        d.arm()
        status, details = _diag(d, gid)
    assert status[PARAMETERS] == "Skip", status
    text = f"GPU 0: no limit could be compared. GPU 1: {LIMITS} limits in range."
    assert _info(PARAMETERS, text) in details, details
    assert f" Device 0 {NO_LIMIT}" in details, details


def test_processes_that_start_while_the_list_is_read_are_named(tools, tmp_path):
    """RDC_FAKE_PROCS_GROW adds a process between the count call and the fill call."""
    with _rdcd(tools, tmp_path, RDC_FAKE_PROCS=2, RDC_FAKE_PROCS_GROW=1) as d:
        status, details = _diag(d, _group(d, "processes", range(NUM_GPUS)))
    assert status[PROCESSES] == "Fail", details
    info = [line for line in details if line.startswith(PROCESSES + ":")]
    assert len(info) == 1, details
    pids = [int(p) for p in re.findall(r"Process: (\d+),", info[0])]
    assert len(pids) >= 3, info  # 2, and 1 more than the count call found
    # The fake runs process 1000 + i on GPU i % RDC_FAKE_GPUS. Only this check's lines: the ROCr
    # checks print a line per GPU too.
    processes = _device_lines(details, PROCESSES)
    for gpu in range(NUM_GPUS):
        on_gpu = " ".join(str(p) for p in pids if (p - 1000) % NUM_GPUS == gpu)
        lines = [line for line in processes if line.startswith(f" Device {gpu} ")]
        assert lines == ([f" Device {gpu} Running process: {on_gpu}"] if on_gpu else []), details


def test_processes_whose_gpus_cannot_be_read_are_skipped(tools, tmp_path):
    fail = "amdsmi_get_gpu_compute_process_gpus"
    with _rdcd(tools, tmp_path, RDC_FAKE_PROCS=2, RDC_FAKE_FAIL=fail) as d:
        gid = _group(d, "unreadable", range(NUM_GPUS))
        d.arm()
        status, details = _diag(d, gid)
    assert status[PROCESSES] == "Skip", status
    assert _info(PROCESSES, "Could not read the GPUs of 2 processes.") in details, details


def test_each_check_of_an_empty_group_is_skipped(tools, tmp_path):
    with _rdcd(tools, tmp_path) as d:
        status, details = _diag(d, _group(d, "empty", []))
    for check in CHECKS:
        assert status[check] == "Skip", status
        assert _info(check, "The group has no GPU.") in details, details


def test_a_custom_run_logs_no_error(tools, tmp_path):
    with _rdcd(tools, tmp_path) as d:
        gid = _group(d, "custom", range(NUM_GPUS))
        for _ in range(3):
            d.ok("diag", "-g", gid, "-r", "1", "-c", "custom", timeout=60)
        log = d.out_log.read_text(errors="replace")
    errors = [line for line in log.splitlines() if " ERROR " in line]
    assert not [line for line in errors if "Custom diagnostic run" in line], errors


def test_rdcd_runs_a_diagnostic_with_a_2mb_stack(tools, tmp_path):
    """rdcd does not hold the diagnostic response on its request thread's stack."""
    with _rdcd(tools, tmp_path, small_stack=True) as d:
        d.ok("discovery", "-l")
        status, _ = _diag(d, _group(d, "stack", range(NUM_GPUS)))
    assert set(CHECKS) <= set(status), status


def test_rdci_runs_a_diagnostic_with_a_2mb_stack(tools, tmp_path):
    """Neither rdcd nor rdci holds the diagnostic response on its stack."""
    with _rdcd(tools, tmp_path, small_stack=True) as d:
        gid = _group(d, "stack", range(NUM_GPUS))
        cmd = [d.rdci_path, "diag", "-g", gid, "-r", "1", "-u", "--host", f"127.0.0.1:{d.port}"]
        try:
            r = subprocess.run(
                cmd,
                env=d.rdci_env,
                capture_output=True,
                text=True,
                timeout=60 * FACTOR,
                preexec_fn=_small_stack,
            )
        except subprocess.TimeoutExpired:
            pytest.fail("rdci diag with a 2 MB stack timed out")
        d.check_alive("rdci diag with a 2 MB stack")
    assert r.returncode == 0, f"rdci diag with a 2 MB stack -> {r.returncode}\n{r.stdout}{r.stderr}"
    status, _ = _parse(r.stdout)
    assert set(CHECKS) <= set(status), status
