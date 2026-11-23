"""Watches on fake GPUs: a job stops even after another watched GPU group was
destroyed, and two jobs on GPU groups that share a GPU both get its samples.

Runs like test_fake_gpu.py (same environment variables and helpers).
"""

import re
import time

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (fixture)
    FACTOR,
    Rdcd,
    _created_id,
    tools,
)

NO_SAMPLE = str(2**64 - 1)  # the minimum rdci shows for a statistic without samples


def _group(d, name, gpus):
    gid = _created_id(d.ok("group", "-c", name))
    d.ok("group", "-g", gid, "-a", ",".join(str(g) for g in gpus))
    return gid


def _gpu_sections(stdout):
    """{GPU index: lines} of rdci stats -j -v."""
    sections, gpu = {}, None
    for line in stdout.splitlines():
        m = re.match(r"\|\s+GPU (\d+)\s*$", line)
        if m:
            gpu = int(m.group(1))
            sections[gpu] = []
        elif gpu is not None:
            sections[gpu].append(line)
    return sections


def test_job_stops_after_another_watched_group_is_destroyed(tools, tmp_path):
    d = Rdcd(tools, tmp_path).start()
    try:
        kept = _group(d, "kept", [0])
        gone = _group(d, "gone", [1])
        d.ok("stats", "-s", "kept_job", "-g", kept)
        d.ok("health", "-g", gone, "-s", "p")
        d.ok("group", "-d", gone)
        d.ok("stats", "-x", "kept_job")
    finally:
        d.stop()


def test_jobs_on_groups_sharing_a_gpu_both_get_its_samples(tools, tmp_path):
    d = Rdcd(tools, tmp_path).start()
    try:
        d.ok("stats", "-s", "job_a", "-g", _group(d, "a", [0]))
        d.ok("stats", "-s", "job_b", "-g", _group(d, "b", [0, 1]))
        # rdci starts jobs with a one-second update frequency.
        time.sleep(3 * FACTOR)
        for job in ("job_a", "job_b"):
            gpu0 = _gpu_sections(d.ok("stats", "-j", job, "-v"))[0]
            power = next(line for line in gpu0 if "Power Usage" in line)
            assert NO_SAMPLE not in power, f"{job}: {power}"
    finally:
        d.stop()
