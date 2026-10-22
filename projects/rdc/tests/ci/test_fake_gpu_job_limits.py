"""rdcd keeps at most 256 jobs: starting another removes the job that stopped
first or, if none has stopped, a running job whose GPU group was destroyed or
runs a later job. A job whose start fails leaves nothing behind, and removing a
job frees its ID. On fake GPUs, through the C API.

Runs like test_fake_gpu.py (same environment variables and helpers). Jobs use a
one-hour update frequency, so rdcd reads each job field once.
"""

import ctypes
import os
import time

import pytest

# The helpers and the `tools` fixture come from test_fake_gpu.py, not conftest.py:
# tests/ci/conftest.py is not installed with the suite (share/rdc/rdctst_tests/fake_gpu).
from test_fake_gpu import (  # noqa: F401 (fixture)
    FACTOR,
    PREFIX,
    Rdcd,
    _first_file,
    _missing,
    tools,
)

RDC_ST_OK = 0
RDC_ST_NOT_FOUND = 6
RDC_ST_CONFLICT = 7
RDC_ST_ALREADY_EXIST = 9
RDC_ST_GROUP_NOT_FOUND = 17
RDC_GROUP_EMPTY = 1
JOB_FIELD_GROUP = 0  # JOB_FIELD_ID, the field group rdcd watches for jobs
MAX_JOBS = 256  # RdcCacheManagerImpl::kMaxJobs
UPDATE_FREQ_US = 3600 * 1000 * 1000
JOB_INFO_SIZE = 1 << 16  # more than sizeof(rdc_job_info_t)
SET_MASK = "amdsmi_set_gpu_event_notification_mask"


class Client:
    """A C API connection to one rdcd. Each call returns its rdc_status_t."""

    def __init__(self, d):
        bootstrap = _first_file(os.path.join(PREFIX, "lib", "librdc_bootstrap.so"))
        if not bootstrap:
            _missing("librdc_bootstrap.so")
        lib = ctypes.CDLL(bootstrap)
        handle, job, u32 = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32
        lib.rdc_init.argtypes = [ctypes.c_uint64]
        lib.rdc_connect.argtypes = [ctypes.c_char_p, ctypes.POINTER(handle)] + [ctypes.c_char_p] * 3
        lib.rdc_disconnect.argtypes = [handle]
        lib.rdc_group_gpu_create.argtypes = [handle, ctypes.c_int, ctypes.c_char_p]
        lib.rdc_group_gpu_create.argtypes += [ctypes.POINTER(u32)]
        lib.rdc_group_gpu_add.argtypes = [handle, u32, u32]
        lib.rdc_group_gpu_destroy.argtypes = [handle, u32]
        lib.rdc_job_start_stats.argtypes = [handle, u32, job, ctypes.c_uint64]
        lib.rdc_job_get_stats.argtypes = [handle, job, ctypes.c_void_p]
        lib.rdc_job_stop_stats.argtypes = [handle, job]
        lib.rdc_job_remove.argtypes = [handle, job]
        lib.rdc_job_remove_all.argtypes = [handle]
        lib.rdc_field_unwatch.argtypes = [handle, u32, u32]
        initialized = lib.rdc_init(0)
        assert initialized == RDC_ST_OK
        self.lib = lib
        self.handle = handle()
        host = f"127.0.0.1:{d.port}".encode()
        connected = lib.rdc_connect(host, ctypes.byref(self.handle), None, None, None)
        assert connected == RDC_ST_OK
        self.info = ctypes.create_string_buffer(JOB_INFO_SIZE)
        self.groups = 0

    def close(self):
        disconnected = self.lib.rdc_disconnect(self.handle)
        assert disconnected == RDC_ST_OK

    def group(self, gpu=0):
        """A new group of one GPU."""
        gid = ctypes.c_uint32()
        self.groups += 1
        name = f"jobs{self.groups}".encode()
        created = self.lib.rdc_group_gpu_create(
            self.handle, RDC_GROUP_EMPTY, name, ctypes.byref(gid)
        )
        assert created == RDC_ST_OK
        added = self.lib.rdc_group_gpu_add(self.handle, gid.value, gpu)
        assert added == RDC_ST_OK
        return gid.value

    def destroy(self, gid):
        destroyed = self.lib.rdc_group_gpu_destroy(self.handle, gid)
        assert destroyed == RDC_ST_OK

    @staticmethod
    def _job(name):
        return ctypes.create_string_buffer(name.encode(), 64)

    def start(self, gid, name):
        return self.lib.rdc_job_start_stats(self.handle, gid, self._job(name), UPDATE_FREQ_US)

    def get(self, name):
        return self.lib.rdc_job_get_stats(self.handle, self._job(name), self.info)

    def stop(self, name):
        return self.lib.rdc_job_stop_stats(self.handle, self._job(name))

    def remove(self, name):
        return self.lib.rdc_job_remove(self.handle, self._job(name))

    def remove_all(self):
        return self.lib.rdc_job_remove_all(self.handle)

    def unwatch(self, gid, field_group):
        return self.lib.rdc_field_unwatch(self.handle, gid, field_group)


@pytest.fixture
def connect(tools, tmp_path):
    """connect(gpus=1, **fake_env) -> (rdcd, Client); the teardown asserts a
    clean exit."""
    started = []

    def _connect(gpus=1, **fake_env):
        d = Rdcd(tools, tmp_path, gpus=gpus, **fake_env).start()
        started.append(d)
        c = Client(d)
        started.append(c)
        return d, c

    yield _connect
    for item in reversed(started):
        if isinstance(item, Client):
            item.close()
        else:
            code = item.stop()
            assert code == 0, f"rdcd exited with {code} after SIGTERM\n{item.output()}"


def test_a_job_at_the_limit_removes_the_job_that_stopped_first(connect):
    d, c = connect()
    gid = c.group()
    for i in range(MAX_JOBS):
        assert c.start(gid, f"job{i}") == RDC_ST_OK, i
        assert c.stop(f"job{i}") == RDC_ST_OK, i
    assert c.start(gid, "new") == RDC_ST_OK, d.output()
    assert c.get("job0") == RDC_ST_NOT_FOUND
    assert [i for i in range(1, MAX_JOBS) if c.get(f"job{i}") != RDC_ST_OK] == []
    assert c.get("new") == RDC_ST_OK
    # A restarted job counts from its new start: job1 is now the oldest.
    assert c.stop("new") == RDC_ST_OK
    assert c.start(gid, "job2") == RDC_ST_OK
    assert c.stop("job2") == RDC_ST_OK
    assert c.start(gid, "newer") == RDC_ST_OK
    assert (c.get("job1"), c.get("job2"), c.get("job3")) == (RDC_ST_NOT_FOUND, RDC_ST_OK, RDC_ST_OK)


def test_a_job_that_fails_to_start_removes_no_job(connect):
    d, c = connect(gpus=2, RDC_FAKE_FAIL=SET_MASK)
    gid = c.group()
    for i in range(MAX_JOBS):
        assert c.start(gid, f"job{i}") == RDC_ST_OK, i
        assert c.stop(f"job{i}") == RDC_ST_OK, i
    assert c.start(gid, "running") == RDC_ST_OK
    assert c.get("job0") == RDC_ST_NOT_FOUND  # (the old code fails here: no limit)
    # Only a job that starts makes room: this start fails after it entered the
    # job, on the event mask of GPU 1 (GPU 0 has its mask already).
    d.arm()
    assert c.start(c.group(gpu=1), "failed") not in (
        RDC_ST_OK,
        RDC_ST_CONFLICT,
        RDC_ST_ALREADY_EXIST,
    )
    assert d.wait_for_failed_call(SET_MASK, 0), "the start did not fail on the event mask"
    assert c.get("job1") == RDC_ST_OK


def test_a_job_at_the_limit_removes_a_job_whose_group_was_destroyed(connect):
    d, c = connect()
    # A group runs one job at a time and rdcd keeps at most 64 groups, but a
    # group can be destroyed while its job runs: such jobs fill the limit.
    live = c.group()
    assert c.start(live, "live") == RDC_ST_OK  # started first; its group stays
    for i in range(MAX_JOBS - 1):
        gid = c.group()
        assert c.start(gid, f"job{i}") == RDC_ST_OK, i
        c.destroy(gid)
    assert c.start(c.group(), "new") == RDC_ST_OK, d.output()
    assert c.get("job0") == RDC_ST_NOT_FOUND
    # Not the job started first, whose group stays, nor the others.
    assert (c.get("live"), c.get("job1"), c.get("new")) == (RDC_ST_OK,) * 3


def _replace_jobs(c, gid):
    """Runs 256 jobs on one group, job0 to job255: once a client unwatches a
    group's job fields, another job can start on the group, and the group's
    earlier jobs share the later job's watch."""
    for i in range(MAX_JOBS):
        assert c.start(gid, f"job{i}") == RDC_ST_OK, i
        if i < MAX_JOBS - 1:
            assert c.unwatch(gid, JOB_FIELD_GROUP) == RDC_ST_OK, i


def test_a_job_at_the_limit_removes_a_job_that_a_later_job_replaced(connect):
    d, c = connect()
    gid = c.group()
    _replace_jobs(c, gid)
    assert c.start(c.group(), "new") == RDC_ST_OK, d.output()
    assert c.get("job0") == RDC_ST_NOT_FOUND
    # The last job kept its watch. rdcd drops ended watches at most once a
    # second; had the start ended the shared watch, the job could not stop.
    time.sleep(3 * FACTOR)
    assert c.stop(f"job{MAX_JOBS - 1}") == RDC_ST_OK


def test_a_conflicting_start_at_the_limit_stops_no_job(connect):
    d, c = connect()
    gid = c.group()
    _replace_jobs(c, gid)
    # The last job watches the group's job fields, so the start fails before
    # it makes room.
    assert c.start(gid, "new") == RDC_ST_CONFLICT
    assert c.stop("job0") == RDC_ST_OK  # still running


def _run_jobs_on_destroyed_groups(c):
    """Runs 256 jobs, job0 to job255, each on a group destroyed after its start."""
    for i in range(MAX_JOBS):
        gid = c.group()
        assert c.start(gid, f"job{i}") == RDC_ST_OK, i
        c.destroy(gid)


@pytest.mark.parametrize("orphans", ["replaced", "destroyed"])
def test_a_start_that_fails_at_the_limit_keeps_the_job_it_stopped(connect, orphans):
    d, c = connect(gpus=2, RDC_FAKE_FAIL=SET_MASK)
    if orphans == "replaced":
        _replace_jobs(c, c.group())
    else:
        _run_jobs_on_destroyed_groups(c)
    # The start fails after it made room, on the event mask of GPU 1.
    d.arm()
    assert c.start(c.group(gpu=1), "new") not in (RDC_ST_OK, RDC_ST_CONFLICT, RDC_ST_ALREADY_EXIST)
    assert d.wait_for_failed_call(SET_MASK, 0), "the start did not fail on the event mask"
    # The start stopped job0 to make room, then failed: job0 stays, stopped.
    assert c.get("job0") == RDC_ST_OK
    assert c.stop("job0") == RDC_ST_NOT_FOUND


def test_a_job_that_fails_to_start_leaves_nothing_behind(connect):
    d, c = connect()
    gid = c.group()
    assert c.start(gid, "first") == RDC_ST_OK
    # rdcd watches a group's job fields for one job at a time.
    assert c.start(gid, "second") == RDC_ST_CONFLICT
    assert c.get("second") == RDC_ST_NOT_FOUND
    # Stopping the job that failed must not stop the first job's watch.
    assert c.stop("second") == RDC_ST_NOT_FOUND
    assert c.start(gid, "third") == RDC_ST_CONFLICT
    assert c.start(c.group(), "second") == RDC_ST_OK  # not RDC_ST_ALREADY_EXIST


def test_a_stopped_job_whose_restart_fails_when_it_begins_stays(connect):
    d, c = connect()
    busy = c.group()
    assert c.start(busy, "running") == RDC_ST_OK
    assert c.start(c.group(), "old") == RDC_ST_OK
    assert c.stop("old") == RDC_ST_OK
    # The restart fails before RDC enters the job: on a group whose job fields
    # are watched, or on a group that is gone.
    assert c.start(busy, "old") == RDC_ST_CONFLICT
    assert c.get("old") == RDC_ST_OK
    gone = c.group()
    c.destroy(gone)
    assert c.start(gone, "old") == RDC_ST_GROUP_NOT_FOUND
    assert c.get("old") == RDC_ST_OK


def test_a_job_that_fails_after_its_watch_starts_leaves_nothing_behind(connect):
    # A job start sets the GPU's event mask after it starts watching the job's
    # fields; when that fails, the job and its watch must go.
    d, c = connect(RDC_FAKE_FAIL=SET_MASK)
    gid = c.group()
    d.arm()
    first = c.start(gid, "job")
    assert first not in (RDC_ST_OK, RDC_ST_CONFLICT, RDC_ST_ALREADY_EXIST), d.output()
    assert d.wait_for_failed_call(SET_MASK, 0), "the start did not fail on the event mask"
    assert c.get("job") == RDC_ST_NOT_FOUND
    assert c.start(gid, "job") == first  # not RDC_ST_CONFLICT or RDC_ST_ALREADY_EXIST


@pytest.mark.parametrize("remove", ["one", "all"])
def test_removing_a_job_that_cannot_stop_frees_its_id(connect, remove):
    d, c = connect()
    gid = c.group()
    assert c.start(gid, "job") == RDC_ST_OK
    # Unwatching the job field group ends the job's watch, which rdcd then drops
    # within about a second; a job without a watch cannot stop.
    deadline = time.monotonic() + 10 * FACTOR
    while c.unwatch(gid, JOB_FIELD_GROUP) == RDC_ST_OK:
        assert time.monotonic() < deadline, "rdcd kept the job's watch"
        time.sleep(0.1)
    assert c.stop("job") == RDC_ST_NOT_FOUND
    assert (c.remove("job") if remove == "one" else c.remove_all()) == RDC_ST_OK
    assert c.get("job") == RDC_ST_NOT_FOUND
    assert c.start(c.group(), "job") == RDC_ST_OK  # not RDC_ST_ALREADY_EXIST
