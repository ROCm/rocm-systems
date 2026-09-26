# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""
Unit tests for GPU-specific test counter selection.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

import pytest
from conftest import RocprofsysTest, _validate_rocpd_candidates
from rocprofsys import GPUInfo, TestResult as RocprofsysTestResult, ValidationResult
from rocprofsys import cache

pytestmark = [pytest.mark.pytest_impl]


# ---------------------------------------------------------------------------
# Helpers / fixtures for the persistent-cache tests
# ---------------------------------------------------------------------------
@dataclass
class _Sample:
    """A non-JSON-native return type, used to exercise to_cache/from_cache."""

    name: str
    tags: set


def _sample_to_cache(sample: _Sample) -> dict:
    return {"name": sample.name, "tags": sorted(sample.tags)}


def _sample_from_cache(data: dict) -> _Sample:
    return _Sample(name=data["name"], tags=set(data["tags"]))


@dataclass
class _DataclassSample:
    """A plain dataclass: the codec round-trips it automatically, no converters."""

    name: str
    tags: set
    path: Path = Path(".")


def _cache_concurrent_writer(
    cache_path: str, proc_idx: int, n_keys: int, barrier
) -> None:
    """Worker: hammer a shared ``PersistentCache`` with a distinct key range.

    Module-level so forked worker processes can reference it. The barrier releases
    all writers together so their read-modify-write cycles overlap and actually
    exercise the file lock.
    """
    store = cache.PersistentCache(Path(cache_path))
    barrier.wait()
    for k in range(n_keys):
        store.set(f"p{proc_idx}.key{k}", proc_idx * 1000 + k)


@pytest.fixture
def cache_store(tmp_path, monkeypatch):
    """An isolated ``PersistentCache`` wired in as the process-wide shared cache.

    Backed by a file under pytest's ``tmp_path`` and injected by monkeypatching
    ``cache.get_shared_cache``, so the decorator/property tests never touch the
    real cache file that a live test session uses.
    """
    store = cache.PersistentCache(tmp_path / "syscache.json")
    monkeypatch.setattr(cache, "get_shared_cache", lambda: store)
    return store


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@pytest.mark.class_name("gpu-info")
class TestGPUInfo(RocprofsysTest):
    def test_gfx1250_uses_gfx1250_counter_set(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx1250"],
            device_count=1,
            categories={"instinct"},
        )

        assert (
            gpu_info.rocm_events_for_test
            == "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TX_VCA_VCA_BUSY"
        )
        assert gpu_info.counter_names == [
            "GRBM_COUNT",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "TX_VCA_VCA_BUSY",
        ]
        assert gpu_info.expected_counter_files == [
            "rocprof-device-[0-9]-GRBM_COUNT*.txt",
            "rocprof-device-[0-9]-SQ_WAVES*.txt",
            "rocprof-device-[0-9]-SQ_INSTS_VALU*.txt",
            "rocprof-device-[0-9]-TX_VCA_VCA_BUSY*.txt",
        ]

    def test_mi300_and_later_keep_ta_ta_busy(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx942"],
            device_count=1,
            categories={"instinct"},
        )

        assert (
            gpu_info.rocm_events_for_test
            == "GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY"
        )
        assert gpu_info.counter_names == [
            "GRBM_COUNT",
            "SQ_WAVES",
            "SQ_INSTS_VALU",
            "TA_TA_BUSY",
        ]

    def test_non_mi300_non_gfx1250_keep_single_counter(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx1201"],
            device_count=1,
            categories={"radeon"},
        )

        assert gpu_info.rocm_events_for_test == "SQ_WAVES"
        assert gpu_info.counter_names == ["SQ_WAVES"]

    def test_unsupported_perf_counter_archs_flags_matches(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx1102", "gfx942"],
            device_count=2,
            categories={"radeon", "instinct"},
        )

        assert gpu_info.unsupported_perf_counter_archs == {"gfx1102"}

    def test_unsupported_perf_counter_archs_empty_when_supported(self):
        gpu_info = GPUInfo(
            available=True,
            architectures=["gfx942"],
            device_count=1,
            categories={"instinct"},
        )

        assert gpu_info.unsupported_perf_counter_archs == set()


@pytest.mark.class_name("test-result")
class TestTestResult(RocprofsysTest):
    def test_rocpd_files_prefers_default_database(self, tmp_path):
        default_db = tmp_path / "rocpd.db"
        rank_db = tmp_path / "rocpd-2-0.db"
        default_db.touch()
        rank_db.touch()

        result = RocprofsysTestResult(0, "", tmp_path, [], {})

        assert result.rocpd_files == [default_db]

    def test_rocpd_files_returns_sorted_rank_databases(self, tmp_path):
        higher_pid_db = tmp_path / "rocpd-66607-0.db"
        lower_pid_db = tmp_path / "rocpd-66606-0.db"
        higher_pid_db.touch()
        lower_pid_db.touch()

        result = RocprofsysTestResult(0, "", tmp_path, [], {})

        assert result.rocpd_files == [lower_pid_db, higher_pid_db]

    def test_rocpd_candidates_accepts_later_valid_candidate(
        self,
        tmp_path: Path,
    ) -> None:
        invalid_db = tmp_path / "rocpd-66606-0.db"
        valid_db = tmp_path / "rocpd-66607-0.db"
        invalid_db.touch()
        valid_db.touch()
        calls: list[Path] = []

        def validate_rocpd_database(db_path: Path) -> ValidationResult:
            calls.append(db_path)
            if db_path == valid_db:
                return ValidationResult(
                    True,
                    "valid candidate",
                    stdout="rocpd validated",
                    command=f"validate {db_path.name}",
                )
            return ValidationResult(
                False,
                "missing GPU counter rows",
                stdout="validation failed",
                command=f"validate {db_path.name}",
            )

        passing_output, failures, global_failure = _validate_rocpd_candidates(
            [invalid_db, valid_db],
            validate_rocpd_database,
            pass_regex=[r"rocpd validated"],
        )

        assert calls == [invalid_db, valid_db]
        assert passing_output == f"Command: validate {valid_db.name}\n\nvalid candidate"
        assert global_failure is None
        assert failures == [
            f"Command: validate {invalid_db.name}\n\nmissing GPU counter rows"
        ]

    def test_rocpd_candidates_reports_all_candidate_failures(
        self,
        tmp_path: Path,
    ) -> None:
        first_db = tmp_path / "rocpd-66606-0.db"
        second_db = tmp_path / "rocpd-7-0.db"
        first_db.touch()
        second_db.touch()

        def validate_rocpd_database(db_path: Path) -> ValidationResult:
            return ValidationResult(
                False,
                f"{db_path.name} is missing GPU counter rows",
                stdout="validation failed",
                command=f"validate {db_path.name}",
            )

        passing_output, failures, global_failure = _validate_rocpd_candidates(
            [first_db, second_db],
            validate_rocpd_database,
        )

        assert passing_output is None
        assert global_failure is None
        message = "\n\n--- Next ROCpd candidate ---\n\n".join(failures)
        assert "rocpd-66606-0.db is missing GPU counter rows" in message
        assert "rocpd-7-0.db is missing GPU counter rows" in message
        assert "--- Next ROCpd candidate ---" in message

    def test_rocpd_candidates_fail_regex_is_global(
        self,
        tmp_path: Path,
    ) -> None:
        invalid_db = tmp_path / "rocpd-66606-0.db"
        valid_db = tmp_path / "rocpd-66607-0.db"
        invalid_db.touch()
        valid_db.touch()
        calls: list[Path] = []

        def validate_rocpd_database(db_path: Path) -> ValidationResult:
            calls.append(db_path)
            if db_path == invalid_db:
                return ValidationResult(
                    False,
                    "invalid candidate",
                    stdout="validation failed with FORBIDDEN marker",
                    command=f"validate {db_path.name}",
                )
            return ValidationResult(
                True,
                "valid candidate",
                stdout="rocpd validated",
                command=f"validate {db_path.name}",
            )

        passing_output, failures, global_failure = _validate_rocpd_candidates(
            [invalid_db, valid_db],
            validate_rocpd_database,
            fail_regex=[r"FORBIDDEN"],
        )

        assert calls == [invalid_db]
        assert passing_output is None
        assert failures == []
        assert global_failure is not None
        assert "Fail regex found: FORBIDDEN" in global_failure
        assert f"validate {invalid_db.name}" in global_failure


@pytest.mark.class_name("cache")
class TestCache(RocprofsysTest):
    """Unit tests for the persistent cross-process cache (``rocprofsys.cache``).

    One test per component to keep the count small; each walks that component's
    behaviours in a single flow. The decorator/property tests use the
    ``cache_store`` fixture so they never touch the real shared cache file.
    """

    def test_codec_round_trips_supported_types_and_rejects_others(self) -> None:
        """The tagged JSON codec preserves every supported type and rejects the rest."""
        payload = {
            "none": None,
            "bools": [True, False],
            "ints": [0, 42, -7],
            "float": 3.14,
            "strings": ["", "hello"],
            "path": Path("/opt/rocm"),
            "tuples": [(), (7, 1, 1)],
            "sets": [set(), {"instinct", "apu"}],
            "mixed_list": [1, "a", Path("/x")],
            1: "non-string keys survive the tagged encoding",
            "nested": {"paths": [Path("/a")], "categories": {"instinct"}},
        }
        assert cache.decode(cache.encode(payload)) == payload

        with pytest.raises(cache.SerializationError):  # unsupported type
            cache.encode(object())
        with pytest.raises(cache.DeserializationError):  # unknown tag on decode
            cache.decode({"__type__": "bogus", "value": 1})

    def test_persistent_cache_store(self, tmp_path) -> None:
        """get/set/miss, cross-instance persistence, clear, typed values, resilience."""
        path = tmp_path / "c.json"
        store = cache.PersistentCache(path)

        assert store.get("missing") == (False, None)  # miss
        store.set("k", {"a": 1})
        store.set("path", Path("/opt/rocm"))  # non-JSON-native types
        store.set("tuple", (7, 1, 1))
        store.set("set", {"instinct", "apu"})
        # An unserializable value is a programming error: set() raises instead of
        # silently skipping, and nothing is stored under the key.
        with pytest.raises(cache.SerializationError):
            store.set("bad", object())
        assert store.get("bad") == (False, None)

        # A second instance (i.e. another process) reads what the first wrote.
        reader = cache.PersistentCache(path)
        assert reader.get("k") == (True, {"a": 1})
        assert reader.get("path") == (True, Path("/opt/rocm"))
        assert reader.get("tuple") == (True, (7, 1, 1))
        assert reader.get("set") == (True, {"instinct", "apu"})

        store.clear()  # wipes memory + disk
        assert cache.PersistentCache(path).get("k") == (False, None)

        # A corrupt file loads as empty but stays usable.
        path.write_text("{ not valid json")
        corrupt = cache.PersistentCache(path)
        assert corrupt.get("k") == (False, None)
        corrupt.set("k", 1)
        assert corrupt.get("k") == (True, 1)

    def test_persistent_cache_decorator(self, cache_store, monkeypatch) -> None:
        """@persistent_cache: compute-once, arg keying, converters, method=True, unserializable-raises, fail-open."""
        fn_calls: list[int] = []

        @cache.persistent_cache("test.fn")
        def fn(x: int) -> int:
            fn_calls.append(x)
            return x * 2

        assert fn(21) == 42  # miss -> compute + store
        assert fn(21) == 42  # hit -> served from cache
        fn(3)  # distinct arg -> recompute (keyed on args)
        assert fn_calls == [21, 3]

        # to_cache/from_cache round-trip a non-JSON-native return type.
        make_calls: list[str] = []

        @cache.persistent_cache(
            "test.sample", to_cache=_sample_to_cache, from_cache=_sample_from_cache
        )
        def make(name: str) -> _Sample:
            make_calls.append(name)
            return _Sample(name=name, tags={"a", "b"})

        first, second = make("v"), make("v")  # second is rebuilt via from_cache
        assert make_calls == ["v"]
        assert isinstance(second, _Sample) and second == first

        # method=True drops self from the key, so distinct instances share.
        meth_calls: list[tuple[str, str]] = []

        class Probe:
            def __init__(self, tag: str) -> None:
                self.tag = tag

            @cache.persistent_cache("test.meth", method=True)
            def check(self, target: str) -> str:
                meth_calls.append((self.tag, target))
                return f"result:{target}"

        assert Probe("A").check("x") == "result:x"  # computed by A
        assert Probe("B").check("x") == "result:x"  # different self -> still a hit
        assert meth_calls == [("A", "x")]

        # A non-serializable *result* is a bug: it propagates out of the wrapped
        # call (loud) rather than being silently left uncached.
        @cache.persistent_cache("test.unserializable")
        def bad() -> object:
            return object()

        with pytest.raises(cache.SerializationError):
            bad()

        # Fail-open: with no shared cache the wrapper still computes, never raises.
        monkeypatch.setattr(cache, "get_shared_cache", lambda: None)
        disabled_calls: list[int] = []

        @cache.persistent_cache("test.disabled")
        def disabled(x: int) -> int:
            disabled_calls.append(x)
            return x + 1

        assert disabled(1) == 2
        assert disabled(1) == 2
        assert disabled_calls == [1, 1]  # recomputed each time, no store

    def test_persistent_cached_property(self, cache_store) -> None:
        """Cross-instance sharing, per-class key namespacing, and loud on unserializable."""
        calls: list[int] = []

        class Caps:
            def __init__(self, n: int) -> None:
                self._n = n

            @cache.persistent_cached_property
            def value(self) -> int:
                calls.append(self._n)
                return 100

        assert Caps(1).value == 100  # miss -> compute + store
        assert Caps(2).value == 100  # fresh instance -> served from shared file
        assert calls == [1]

        # Same attribute name on a different class must not collide in the store.
        class Other:
            @cache.persistent_cached_property
            def value(self) -> str:
                return "other"

        assert Other().value == "other"

        # A property whose computed value can't be serialized raises on access
        # (loud), instead of silently caching nothing.
        class BadProp:
            @cache.persistent_cached_property
            def value(self) -> object:
                return object()

        with pytest.raises(cache.SerializationError):
            BadProp().value

    def test_dataclass_round_trips(self, cache_store) -> None:
        """Dataclasses (de)serialize automatically, with no to_cache/from_cache.

        Covers a direct codec round-trip (nested set + Path fields), reuse through
        @persistent_cache, and rejection of a field of an unsupported type.
        """
        sample = _DataclassSample(
            name="mi300", tags={"instinct", "apu"}, path=Path("/opt/rocm")
        )
        assert cache.decode(cache.encode(sample)) == sample  # direct round-trip

        calls: list[str] = []

        @cache.persistent_cache("test.dataclass")
        def make(name: str) -> _DataclassSample:
            calls.append(name)
            return _DataclassSample(name=name, tags={"x"})

        first, second = make("a"), make("a")  # second served from cache
        assert calls == ["a"]
        assert isinstance(second, _DataclassSample) and second == first

        # A dataclass carrying a field of an unsupported type cannot be encoded.
        @dataclass
        class _BadField:
            x: object

        with pytest.raises(cache.SerializationError):
            cache.encode(_BadField(object()))

    def test_concurrent_writers_preserve_all_entries(self, tmp_path) -> None:
        """Parallel writers to distinct keys must not clobber each other.

        Regression guard for the file-lock design: the exclusive lock is held on a
        stable sidecar file, so overlapping writers each preserve the others'
        entries across the atomic tmp+replace. Locking the data file itself would
        drop most writes, since ``os.replace`` swaps the locked inode.
        """
        import multiprocessing as mp

        cache_path = tmp_path / "syscache.tmp"
        cache.PersistentCache(cache_path).set("seed", 1)  # create the file first

        n_proc, n_keys = 6, 40
        ctx = mp.get_context("fork")  # the test suite is Linux-only
        barrier = ctx.Barrier(n_proc)
        procs = [
            ctx.Process(
                target=_cache_concurrent_writer,
                args=(str(cache_path), i, n_keys, barrier),
            )
            for i in range(n_proc)
        ]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
            assert p.exitcode == 0

        final = cache.PersistentCache(cache_path)
        lost = [
            f"p{i}.key{k}"
            for i in range(n_proc)
            for k in range(n_keys)
            if not final.get(f"p{i}.key{k}")[0]
        ]
        assert not lost, f"{len(lost)} of {n_proc * n_keys} concurrent writes were lost"
