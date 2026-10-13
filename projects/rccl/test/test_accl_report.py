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


def _make_record(sn, rank=0, n_ranks=8, exec_us=100.0):
    return json.dumps({
        "header": {"rank": rank, "n_ranks": n_ranks},
        "coll_perf": {
            "coll": "AllReduce",
            "coll_sn": sn,
            "coll_msg_size_bytes": 1048576,
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
