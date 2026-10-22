"""GPU event notifications, against the fake amd-smi: RDC reads every event amd-smi wrote and no
more, skips one it cannot attribute instead of dropping the others, gives each event its own
timestamp, and keeps the events that each watch and each job asked for on a GPU, whatever the
other watches and jobs on that GPU do.

Runs like test_fake_gpu.py (same environment variables and helpers). Events are delivered by the
fake once RDC_FAKE_ARM exists (Rdcd.arm()); with RDC_FAKE_EVENT_MASKS=1 an event of a GPU waits
until RDC has enabled it on that GPU, as the driver sends only enabled events.
"""

import contextlib
import math
import re
import subprocess
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (tools is a fixture)
    FACTOR,
    Rdcd,
    _created_id,
    tools,
)
from test_fake_gpu_job_limits import JOB_FIELD_GROUP, RDC_ST_OK, Client

MASK = "amdsmi_set_gpu_event_notification_mask"
INIT = "amdsmi_init_gpu_event_notification"
STOP = "amdsmi_stop_gpu_event_notification"
# rdci dmon's event line: "<gpu>\t   **Event: <label>\t"<message>"" (with only event fields, dmon
# prints the GPU indexes of its rows without line ends, so the line need not start with it).
EVENT_LINE = re.compile(r'(\d+)\t\s*\*\*Event: (\S+)\t"([^"]*)"')
PROCESS_LINE = re.compile(r"PID: (\d+), Start: (\d+), End: (\d+)")


def _samples(base):
    # dmon samples every 250 ms; an event reaches RDC's cache up to ~1.1 s after the fake can
    # deliver it (the rest of the fake's 100 ms wait in a listen, then RDC's 1 s pause before the
    # next listen), more on slow builds.
    return max(base, math.ceil(base * FACTOR))


class Dmon:
    """`rdci dmon` in the background, so that a test can act while it watches."""

    def __init__(self, d, fields, gpus, samples):
        self.timeout = samples * 0.25 + 10 * FACTOR
        cmd = [d.rdci_path, "dmon", "-e", fields, "-i", gpus, "-c", str(samples), "-d", "250"]
        cmd += ["-u", "--host", f"127.0.0.1:{d.port}"]
        self.proc = subprocess.Popen(
            cmd, env=d.rdci_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )

    def events(self):
        """Wait for dmon to end; the events it printed, as (gpu, label, message)."""
        try:
            out, _ = self.proc.communicate(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            out, _ = self.proc.communicate()
            pytest.fail(f"rdci dmon did not end within {self.timeout:.0f} s\n{out[-3000:]}")
        assert self.proc.returncode == 0, f"rdci dmon -> {self.proc.returncode}\n{out[-3000:]}"
        return sorted((int(g), label, msg) for g, label, msg in EVENT_LINE.findall(out))


def _wait_for_call(d, api, gpu, since, timeout=10):
    """Wait until the fake logged `api` for `gpu` after its first `since` calls."""
    deadline = time.monotonic() + timeout * FACTOR
    while not any(a == api and g == gpu for a, g, _ in d.calls()[since:]):
        d.check_alive(f"waiting for {api} on GPU {gpu}")
        if time.monotonic() > deadline:
            pytest.fail(f"no {api} for GPU {gpu} within {timeout * FACTOR:.0f} s")
        time.sleep(0.05)


def _watch_events(d, fields, gpus, base_samples=24, before_arm=None):
    """`rdci dmon -e <fields> -i <gpus>`, arming the fake's events while it watches.

    dmon's first read, right after its watch, discards what the cache holds already, and an armed
    event can reach the cache at once; so the events are armed only after the watch wrote the
    first GPU's mask and dmon had time for that read (an RPC later). before_arm() runs in
    between, while dmon watches.
    """
    since = len(d.calls())
    dmon = Dmon(d, fields, gpus, _samples(base_samples))
    _wait_for_call(d, MASK, int(gpus.split(",")[0]), since)
    time.sleep(1.5 * FACTOR)
    if before_arm is not None:
        before_arm()
    d.arm()
    return dmon.events()


def _job_process(d, job):
    """Poll `rdci stats -j <job> -v` until it tracks a process that ended: (pid, start, end)."""
    deadline = time.monotonic() + 10 * FACTOR
    while True:
        out = d.ok("stats", "-j", job, "-v")
        m = PROCESS_LINE.search(out)
        if m and int(m.group(3)) != 0:
            assert "Processes Tracked                | 1" in out, out
            return tuple(int(v) for v in m.groups())
        if time.monotonic() > deadline:
            pytest.fail(f"job {job} tracked no ended process within {10 * FACTOR:.0f} s\n{out}")
        time.sleep(0.25)


def _group(d, name, gpus):
    gid = _created_id(d.ok("group", "-c", name))
    d.ok("group", "-g", gid, "-a", gpus)
    return gid


@contextlib.contextmanager
def _rdcd(tools, tmp_path, **fake_env):
    """rdcd on two fake GPUs; it must exit cleanly at the end."""
    d = Rdcd(tools, tmp_path, gpus=2, **fake_env)
    try:
        yield d.start()
    finally:
        code = d.stop()
    assert code == 0, f"rdcd exited with {code} after SIGTERM\n{d.output()}"


def test_events_reach_rdc_when_the_library_reports_more_than_it_wrote(tools, tmp_path):
    # Hardening: amd-smi reports only the events it wrote; RDC must read no further.
    with _rdcd(tools, tmp_path, RDC_FAKE_EVENTS="1@0,2@1,count=65536") as d:
        assert _watch_events(d, "2000,2001", "0,1") == [
            (0, "VM_PAGE_FAULT", "fake event 1"),
            (1, "THERMAL_THROT", "fake event 2"),
        ]


def test_an_event_from_a_gpu_rdc_does_not_list_drops_no_other_event(tools, tmp_path):
    # The x event's handle is not one of RDC's GPUs; the events around it still count.
    with _rdcd(tools, tmp_path, RDC_FAKE_EVENTS="1@0,1@x,2@1") as d:
        assert _watch_events(d, "2000,2001", "0,1") == [
            (0, "VM_PAGE_FAULT", "fake event 1"),
            (1, "THERMAL_THROT", "fake event 2"),
        ]


def test_two_events_of_one_kind_in_one_read_are_both_shown(tools, tmp_path):
    # Read together, with one timestamp, the two made rdc_field_get_value_since() return the
    # first one forever, and dmon never ended.
    with _rdcd(tools, tmp_path, RDC_FAKE_EVENTS="1@0,1@0") as d:
        assert _watch_events(d, "2000", "0") == [(0, "VM_PAGE_FAULT", "fake event 1")] * 2


def test_a_job_keeps_its_process_events_after_a_watch_of_another_event(tools, tmp_path):
    with _rdcd(tools, tmp_path, RDC_FAKE_EVENT_MASKS="1", RDC_FAKE_EVENTS="12@0,13@0") as d:
        d.ok("stats", "-s", "job", "-g", _group(d, "job", "0"))
        # A watch of VM page faults on the job's GPU, from start to end; then a process starts
        # and ends on that GPU.
        d.ok("dmon", "-e", "2000", "-i", "0", "-c", "2", "-d", "100")
        d.arm()
        pid, start, end = _job_process(d, "job")
        # Read in one go, the start and the end still get their own times.
        assert pid == 1000 and 0 < start < end, (pid, start, end)
        d.ok("stats", "-x", "job")


def test_a_watch_keeps_its_events_while_a_job_starts_and_stops(tools, tmp_path):
    with _rdcd(tools, tmp_path, RDC_FAKE_EVENT_MASKS="1", RDC_FAKE_EVENTS="1@0") as d:
        gid = _group(d, "job", "0")

        def job_starts_and_stops():
            d.ok("stats", "-s", "job", "-g", gid)
            d.ok("stats", "-x", "job")

        # The job's calls run while dmon watches, so dmon samples longer.
        events = _watch_events(d, "2000", "0", base_samples=40, before_arm=job_starts_and_stops)
        assert events == [(0, "VM_PAGE_FAULT", "fake event 1")]


def test_the_last_job_on_a_gpu_stops_its_process_events(tools, tmp_path):
    with _rdcd(tools, tmp_path, RDC_FAKE_EVENT_MASKS="1", RDC_FAKE_EVENTS="12@0,13@0") as d:
        d.ok("stats", "-s", "job_a", "-g", _group(d, "job_a", "0"))
        d.ok("stats", "-s", "job_b", "-g", _group(d, "job_b", "0"))
        since = len(d.calls())
        d.ok("stats", "-x", "job_a")
        calls = [(a, g) for a, g, _ in d.calls()[since:]]
        assert (STOP, 0) not in calls, "job_b still runs on GPU 0"
        d.arm()
        pid, start, end = _job_process(d, "job_b")
        assert pid == 1000 and 0 < start < end, (pid, start, end)
        since = len(d.calls())
        d.ok("stats", "-x", "job_b")
        calls = [(a, g) for a, g, _ in d.calls()[since:]]
        assert (STOP, 0) in calls, "no job runs on GPU 0 any more"


def test_a_job_stop_that_fails_still_stops_its_process_events(tools, tmp_path):
    # A client unwatched the job fields, and rdcd drops an ended watch within about a second,
    # so the stop fails at the job's unwatch; it released the job's listeners before that.
    with _rdcd(tools, tmp_path) as d:
        gid = _group(d, "job", "0")
        d.ok("stats", "-s", "job", "-g", gid)
        client = Client(d)
        try:
            deadline = time.monotonic() + 10 * FACTOR
            while client.unwatch(int(gid), JOB_FIELD_GROUP) == RDC_ST_OK:
                assert time.monotonic() < deadline, "rdcd kept the job's watch"
                time.sleep(0.1)
        finally:
            client.close()
        since = len(d.calls())
        r = d.rdci("stats", "-x", "job")
        assert r.returncode != 0, f"the stop of a job without its watch succeeded\n{r.stdout}"
        calls = [(a, g) for a, g, _ in d.calls()[since:]]
        assert (STOP, 0) in calls, calls


def test_a_watch_after_a_failed_stop_enables_its_events_again(tools, tmp_path):
    # The first watch's end writes GPU 0's mask 0 and then fails to stop its events. The next
    # watch of the same event must enable it again; RDC used to keep the old mask and skip it.
    # Armed before rdcd starts: the one failure goes to the first stop, at the first watch's end.
    (tmp_path / "armed").touch()
    with _rdcd(tools, tmp_path, RDC_FAKE_FAIL_ONCE=f"{STOP}@0") as d:
        d.ok("dmon", "-e", "2000", "-i", "0", "-c", "1", "-d", "100")
        assert d.wait_for_failed_call(STOP, timeout=5), "the stop never failed"
        since = len(d.calls())
        d.ok("dmon", "-e", "2000", "-i", "0", "-c", "1", "-d", "100")
        calls = [(a, g) for a, g, _ in d.calls()[since:]]
        assert (INIT, 0) in calls and (MASK, 0) in calls, calls
