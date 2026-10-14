"""Job statistics against the fake amd-smi: every GPU of the job, by its ID.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import json
import re
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import FACTOR, Rdcd, _created_id, golden, tools  # noqa: F401 (fixture)

RDC_ST_NOT_FOUND = 6  # rdc_status_t in include/rdc/rdc.h; rdci exits with it
RDC_ST_MAX_LIMIT = 10
RDC_FI_GPU_MEMORY_USAGE = 501
RDC_FI_GPU_MEMORY_TOTAL = 502
PARTITION_G0_1 = (1 << 27) | (1 << 11)  # partition 1 of GPU 0, see RdcEntityCodec.h


def _memory_utilization(gpu):
    # RDC divides the memory used by 1% of the total memory, in integers.
    total = golden.expected(RDC_FI_GPU_MEMORY_TOTAL, gpu)
    return golden.expected(RDC_FI_GPU_MEMORY_USAGE, gpu) // (total // 100)


# rdci row -> reading of fake GPU g. Readings are constant: max == min == average.
STATS = {
    "Power Usage (Watts)": lambda g: golden.reading("SOCKET_POWER_W", g),
    "GPU Clock (MHz)": lambda g: golden.reading("GFX_CLK_MHZ", g),
    "Memory Clock (MHz)": lambda g: golden.reading("MEM_CLK_MHZ", g),
    "GPU Utilization (%)": lambda g: golden.reading("GFX_ACTIVITY_PCT", g),
    "Memory Utilization (%)": _memory_utilization,
    "GPU Temperature (Celsius)": lambda g: golden.reading("TEMP_EDGE_C", g),
    "PCIe Total Bandwidth (megabytes)": lambda g: golden.reading("PCIE_BANDWIDTH_MBPS", g),
}


def _sections(out):
    """Split `rdci stats -j -v` output into {"Summary": text, "GPU <id>": text}."""
    sections, name = {}, None
    for line in out.splitlines():
        m = re.match(r"\|\s+(Summary|GPU \S+)\s*$", line)
        if m:
            name = m.group(1)
            sections[name] = ""
        elif name:
            sections[name] += line + "\n"
    return sections


def _stat(text, label):
    """(max, min, average, SD) of a row, or None for a row without samples (N/A)."""
    row = (
        rf"\|[ \t]*{re.escape(label)}[ \t]*\|[ \t]*(N/A|Max: (\S+) Min: (\S+) Avg: (\S+) SD: (\S+))"
    )
    m = re.search(row, text)
    assert m, f"no '{label}' row in:\n{text}"
    return None if m.group(1) == "N/A" else tuple(float(v) for v in m.groups()[1:])


def _memory_used(text):
    m = re.search(r"Max GPU Memory Used \(bytes\)\s*\|\s*(\d+)", text)
    assert m, f"no memory used row in:\n{text}"
    return int(m.group(1))


def _missing(out, rows):
    sections = _sections(out)
    return [
        (name, label)
        for name, label in rows
        if name not in sections or _stat(sections[name], label) is None
    ]


def _run_job(d, entities, rows, job="job_all_gpus"):
    """Run a job on `entities` until each (section, row) in `rows` has samples.

    Returns rdci's verbose output while the job runs and after it stops, and
    the parsed JSON output.
    """
    gid = _created_id(d.ok("group", "-c", job))
    d.ok("group", "-g", gid, "-a", ",".join(str(e) for e in entities))
    d.ok("stats", "-s", job, "-g", gid)
    deadline = time.monotonic() + 10 * FACTOR
    running = d.ok("stats", "-j", job, "-v")
    while _missing(running, rows) and time.monotonic() < deadline:
        time.sleep(0.2)
        running = d.ok("stats", "-j", job, "-v")
    missing = _missing(running, rows)
    assert not missing, f"no samples after {10 * FACTOR} s for {missing}:\n{running}"
    d.ok("stats", "-x", job)
    stopped = d.ok("stats", "-j", job, "-v")
    data = json.loads(d.ok("stats", "-j", job, "-v", "--json"))
    d.ok("stats", "-r", job)
    return running, stopped, data


def _rows(gpus):
    return [(f"GPU {g}", label) for g in gpus for label in STATS]


def _gpu_keys(data):
    return {key for key in data if key.startswith("gpu_")}


def _check(out, gpus):
    """Each GPU's section shows its own readings; the summary covers all of them."""
    sections = _sections(out)
    assert sorted(sections) == sorted(["Summary"] + [f"GPU {g}" for g in gpus]), out
    for label, reading in STATS.items():
        values = [reading(g) for g in gpus]
        for g, value in zip(gpus, values):
            assert _stat(sections[f"GPU {g}"], label) == (value, value, value, 0), (g, label, out)
        # The summary average is the integer mean of the GPU averages.
        summary = (max(values), min(values), sum(values) // len(values), 0)
        assert _stat(sections["Summary"], label) == summary, (label, out)
    used = [golden.expected(RDC_FI_GPU_MEMORY_USAGE, g) for g in gpus]
    for g, value in zip(gpus, used):
        assert _memory_used(sections[f"GPU {g}"]) == value, (g, out)
    assert _memory_used(sections["Summary"]) == max(used), out


@pytest.fixture
def make_rdcd(tools, tmp_path):
    started = []

    def make(**fake_env):
        d = Rdcd(tools, tmp_path, **fake_env).start()
        started.append(d)
        return d

    yield make
    for d in started:
        code = d.stop()
        assert code == 0, f"rdcd exited with {code} after SIGTERM\n{d.output()}"


def test_job_stats_cover_every_gpu_of_the_group(make_rdcd):
    d = make_rdcd()
    running, stopped, data = _run_job(d, [1, 3], _rows([1, 3]))
    _check(running, [1, 3])
    _check(stopped, [1, 3])
    assert _gpu_keys(data) == {"gpu_1", "gpu_3"}
    for g in (1, 3):
        power = STATS["Power Usage (Watts)"](g)
        assert data[f"gpu_{g}"]["power_usage_max"] == power, data


def test_job_stats_for_sixteen_gpus_past_the_per_job_array(make_rdcd):
    # gpus[] holds RDC_MAX_NUM_GPUS_PER_JOB (16) entries, and GPU IDs are not slot numbers.
    # These GPUs have 472 to 592 GiB of memory: 1% of that does not fit in 32 bits.
    gpus = list(range(51, 67))
    d = make_rdcd(gpus=67)
    running, stopped, data = _run_job(d, gpus, _rows(gpus))
    _check(running, gpus)
    _check(stopped, gpus)
    assert _gpu_keys(data) == {f"gpu_{g}" for g in gpus}
    assert data["gpu_66"]["power_usage_max"] == STATS["Power Usage (Watts)"](66), data
    # At least one summary mean is not a whole number, so _check tells integer means apart.
    assert any(sum(map(reading, gpus)) % len(gpus) for reading in STATS.values())


def test_job_stats_for_partitions(make_rdcd):
    # With 2 partitions per GPU, RDC GPU 2 is fake GPU 1; partition g0.1 is on fake GPU 0.
    d = make_rdcd(gpus=2, RDC_FAKE_PARTITIONS=2)
    clock = golden.reading("GFX_CLK_MHZ", 0)
    # RDC samples only the clocks of a partition; it has no total memory reading.
    partition_rows = [("GPU g0.1", "GPU Clock (MHz)")]
    running, stopped, data = _run_job(d, [PARTITION_G0_1], partition_rows, job="partition_job")
    for out in (running, stopped):
        sections = _sections(out)
        assert sorted(sections) == ["GPU g0.1", "Summary"], out
        for text in sections.values():
            assert _stat(text, "GPU Clock (MHz)") == (clock, clock, clock, 0), out
            assert _stat(text, "Memory Utilization (%)") is None, out
    assert _gpu_keys(data) == {"gpu_g0.1"}
    memory_keys = ("max", "min", "avg", "stanard_deviation")  # published key spellings
    for key in (f"memory_utilization_{k}" for k in memory_keys):
        assert data["gpu_g0.1"][key] is None and data["job_summary"][key] is None, data

    rows = partition_rows + [("GPU 2", label) for label in STATS]
    running, stopped, data = _run_job(d, [2, PARTITION_G0_1], rows, job="mixed_job")
    for out in (running, stopped):
        sections = _sections(out)
        assert sorted(sections) == ["GPU 2", "GPU g0.1", "Summary"], out
        for label, reading in STATS.items():
            value = reading(1)
            assert _stat(sections["GPU 2"], label) == (value, value, value, 0), (label, out)
        assert _stat(sections["GPU g0.1"], "GPU Clock (MHz)") == (clock, clock, clock, 0), out
        # Each summary average is over the GPUs with samples.
        clocks = [golden.reading("GFX_CLK_MHZ", 1), clock]
        summary = (max(clocks), min(clocks), sum(clocks) // len(clocks), 0)
        assert _stat(sections["Summary"], "GPU Clock (MHz)") == summary, out
        for label in ("Power Usage (Watts)", "Memory Utilization (%)"):
            value = STATS[label](1)
            assert _stat(sections["Summary"], label) == (value, value, value, 0), (label, out)
    assert _gpu_keys(data) == {"gpu_2", "gpu_g0.1"}


def test_job_start_rejects_more_gpus_than_a_job_holds(make_rdcd):
    d = make_rdcd(gpus=17)
    gid = _created_id(d.ok("group", "-c", "too_many"))
    d.ok("group", "-g", gid, "-a", ",".join(str(g) for g in range(17)))
    r = d.rdci("stats", "-s", "job_too_many", "-g", gid)
    assert r.returncode == RDC_ST_MAX_LIMIT, (
        f"want {RDC_ST_MAX_LIMIT}, got {r.returncode}\n{r.stdout}{r.stderr}"
    )
    r = d.rdci("stats", "-j", "job_too_many")
    assert r.returncode == RDC_ST_NOT_FOUND, (
        f"want {RDC_ST_NOT_FOUND}, got {r.returncode}\n{r.stdout}{r.stderr}"
    )
