"""Unit tests for accl_report.py warmup filtering and edge cases."""
import json
import os
import sys
import tempfile

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "plugins", "profiler", "accl")
)
from accl_report import parse_jsonl, fmt_size  # noqa: E402


def _make_record(sn, rank=0, n_ranks=8, exec_us=100.0,
                 msg_size=1048576, coll="AllReduce"):
    return json.dumps({
        "header": {"rank": rank, "n_ranks": n_ranks},
        "coll_perf": {
            "coll": coll,
            "coll_sn": sn,
            "coll_msg_size_bytes": msg_size,
            "coll_algo": "Ring",
            "coll_proto": "Simple",
            "coll_n_channels": 4,
            "coll_exec_time_us": exec_us,
            "coll_algobw_gbs": 1.0,
            "coll_busbw_gbs": 1.5,
            "coll_timing_source": "cpu_wallclock",
            "decomposition": {},
        },
    })


def _write_jsonl(lines):
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False
    )
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


def test_warmup_does_not_multiply_by_nranks():
    """seqNumber is per-comm, identical across ranks — warmup should not scale."""
    path = _write_jsonl([_make_record(sn=i, n_ranks=32) for i in range(0, 20)])
    try:
        records = parse_jsonl(path, warmup=5)
        # Should drop sn 0-4, keep sn 5-19 → 15 records
        assert len(records) == 15
        assert all(r.sn >= 5 for r in records)
    finally:
        os.unlink(path)


def test_warmup_zero_keeps_all():
    path = _write_jsonl([_make_record(sn=i) for i in range(0, 6)])
    try:
        records = parse_jsonl(path, warmup=0)
        assert len(records) == 6
    finally:
        os.unlink(path)


def test_warmup_default_drops_first_five():
    path = _write_jsonl([_make_record(sn=i) for i in range(0, 11)])
    try:
        records = parse_jsonl(path, warmup=5)
        assert len(records) == 6
        assert records[0].sn == 5
    finally:
        os.unlink(path)


def test_empty_file_returns_empty():
    path = _write_jsonl(["", "not json", "{}"])
    try:
        records = parse_jsonl(path, warmup=0)
        assert len(records) == 0
    finally:
        os.unlink(path)


def test_zero_records_after_warmup_warns(capsys):
    path = _write_jsonl([_make_record(sn=i) for i in range(0, 4)])
    try:
        records = parse_jsonl(path, warmup=10)
        assert len(records) == 0
        captured = capsys.readouterr()
        assert "WARNING" in captured.err
    finally:
        os.unlink(path)


def test_fmt_size_exact_boundaries():
    assert fmt_size(1024) == "1K"
    assert fmt_size(1024 * 1024) == "1M"
    assert fmt_size(1024 * 1024 * 1024) == "1G"


def test_fmt_size_fractional():
    assert fmt_size(1536 * 1024) == "1.5M"
    assert fmt_size(512) == "512B"
    assert fmt_size(2560 * 1024) == "2.5M"


def test_fmt_size_small():
    assert fmt_size(0) == "0B"
    assert fmt_size(1) == "1B"


# --- warmup is per (rank, coll, msg_size), not per run --------------------
# coll_sn is comm->seqNumber[func]: monotonic across the whole run per
# function. A bare `sn < warmup` threshold therefore only trims the first
# message size of a sweep; every later size keeps its own warmup iterations.


def _sweep(sizes, warm, timed, rank=0, coll="AllReduce", sn0=0):
    """One rank's sweep: `warm` slow + `timed` fast iterations per size."""
    lines, sn = [], sn0
    for size in sizes:
        for i in range(warm + timed):
            lines.append(_make_record(
                sn=sn, rank=rank, coll=coll, msg_size=size,
                exec_us=300.0 if i < warm else 100.0))
            sn += 1
    return lines


def test_warmup_dropped_for_every_message_size():
    sizes = [1024, 4096, 16384, 65536]
    path = _write_jsonl(_sweep(sizes, warm=5, timed=20))
    try:
        records = parse_jsonl(path, warmup=5)
        assert len(records) == len(sizes) * 20
        for size in sizes:
            kept = [r for r in records if r.msg_size == size]
            assert len(kept) == 20, f"{size}B kept {len(kept)}, expected 20"
            # 300us marks a warmup iteration; none may survive.
            assert all(r.exec_time_us == 100.0 for r in kept), \
                f"{size}B still contains warmup iterations"
    finally:
        os.unlink(path)


def test_warmup_groups_are_per_rank():
    """Each rank runs its own warmup, so N must be dropped per rank, not once
    globally. Rank 1's counter is offset here because nothing aligns the two
    sequences when per-rank files are concatenated."""
    lines = _sweep([1024], warm=3, timed=4, rank=0, sn0=0)
    lines += _sweep([1024], warm=3, timed=4, rank=1, sn0=100)
    path = _write_jsonl(lines)
    try:
        records = parse_jsonl(path, warmup=3)
        assert len(records) == 8
        for rank in (0, 1):
            kept = [r for r in records if r.rank == rank]
            assert len(kept) == 4, f"rank {rank} kept {len(kept)}, expected 4"
            assert all(r.exec_time_us == 100.0 for r in kept)
    finally:
        os.unlink(path)


def test_warmup_groups_are_per_collective():
    """seqNumber is per function, so AllReduce and Broadcast are independent
    sequences; each needs its own warmup dropped at each size."""
    sizes = [1024, 4096]
    lines = _sweep(sizes, warm=2, timed=3, coll="AllReduce")
    lines += _sweep(sizes, warm=2, timed=3, coll="Broadcast")
    path = _write_jsonl(lines)
    try:
        records = parse_jsonl(path, warmup=2)
        assert len(records) == 2 * len(sizes) * 3
        for coll in ("AllReduce", "Broadcast"):
            for size in sizes:
                kept = [r for r in records
                        if r.coll == coll and r.msg_size == size]
                assert len(kept) == 3, f"{coll} {size}B kept {len(kept)}"
                assert all(r.exec_time_us == 100.0 for r in kept)
    finally:
        os.unlink(path)


def test_warmup_selects_lowest_sn_not_file_order():
    """Records may land out of order; "first N" must mean lowest coll_sn."""
    order = [4, 0, 3, 1, 2]
    path = _write_jsonl([_make_record(sn=sn) for sn in order])
    try:
        records = parse_jsonl(path, warmup=2)
        assert sorted(r.sn for r in records) == [2, 3, 4]
    finally:
        os.unlink(path)


def test_short_group_is_dropped_with_warning(capsys):
    """A size with fewer records than warmup vanishes — say so on stderr."""
    lines = _sweep([1024], warm=5, timed=20)
    lines += [_make_record(sn=999, msg_size=2048)]
    path = _write_jsonl(lines)
    try:
        records = parse_jsonl(path, warmup=5)
        assert all(r.msg_size == 1024 for r in records)
        assert len(records) == 20
        err = capsys.readouterr().err
        assert "WARNING" in err and "2K" in err
    finally:
        os.unlink(path)
