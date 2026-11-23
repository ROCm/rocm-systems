"""Unit tests for net_tel_report.py (ACCL network-counter report)."""
import json
import os
import re
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "plugins", "profiler", "accl")
)
from net_tel_report import (  # noqa: E402
    build_trace, classify_compare, classify_group_single, classify_node, classify_single,
    fmt_count, fmt_delta, load, main, per_size, totals, whole_run,
)

S = 1048576


def _snap(sid, t, rdma, ethtool=None, dev="rdma0"):
    return {"id": sid, "read_start_us": t, "read_end_us": t + 10, "stable": True,
            "devices": [{"name": dev, "ok": True, "read_start_us": t, "read_end_us": t + 10,
                         "rdma": rdma, "ethtool": ethtool}]}


def _rec(sn, rank=0, size=S, busbw=10.0, rdma=None, ethtool=None, base=None, coordinated=False,
         overlapped=False, unavailable=False, n_ranks=1, dev="rdma0"):
    """One window. `base` = absolute counters before it; deltas `rdma`/`ethtool`
    are added for the after snapshot."""
    delta = {"tx_rdma_ucast_bytes": size, "rx_rdma_ucast_bytes": size, **(rdma or {})}
    base = base or {}
    after = {k: base.get(k, 0) + delta.get(k, 0) for k in set(base) | set(delta)}
    eth_after = {k: v for k, v in (ethtool or {}).items()}
    return json.dumps({
        "header": {"rank": rank, "n_ranks": n_ranks},
        "coll_perf": {"coll": "AllReduce", "coll_sn": sn, "coll_msg_size_bytes": size,
                      "coll_busbw_gbs": busbw, "coll_exec_time_us": 100.0},
        "net_counters": {
            "overlapped": overlapped, "unavailable": unavailable, "coordinated": coordinated,
            "completion_verified": True, "before_ts_us": 1000 * sn, "after_ts_us": 1000 * sn + 600,
            "collective_start_cpu_us": 1000 * sn + 100, "collective_complete_cpu_us": 1000 * sn + 500,
            "devices": [{"name": dev, "rdma": None if unavailable else delta, "ethtool": ethtool}],
            "snapshot_before": _snap(2 * sn, 1000 * sn, dict(base), dev=dev),
            "snapshot_after": _snap(2 * sn + 1, 1000 * sn + 590, after, eth_after or None, dev=dev),
        },
    })


def _chain(n, rdma=None, size=S, **kw):
    """n windows whose snapshots chain: totals grow by `rdma` each window."""
    lines = []
    acc = {k: 0 for k in ["tx_rdma_ucast_bytes", "rx_rdma_ucast_bytes", *(rdma or {})]}
    for sn in range(n):
        lines.append(_rec(sn, size=size, rdma=rdma, base=dict(acc), **kw))
        for k, v in {"tx_rdma_ucast_bytes": size, "rx_rdma_ucast_bytes": size, **(rdma or {})}.items():
            acc[k] = acc.get(k, 0) + v
    return lines


def _write(tmp_path, lines, rank=0, host="nodeA", comm="0xabc", sub=""):
    d = tmp_path / sub if sub else tmp_path
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"accl_profiler_rank{rank}_{host}_pid1{rank}_{comm}.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


# --- Rules ---------------------------------------------------------------------
@pytest.mark.parametrize("total,expected", [(0, "CLEAN"), (999, "NEGLIGIBLE"), (1000, "ACTIVE")])
def test_single_counter_default_threshold(total, expected):
    assert classify_single(total, "tx_rdma_ack_timeout") == expected


def test_single_counter_pfc_thresholds():
    assert classify_single(4999, "frames_rx_pripause") == "NEGLIGIBLE"
    assert classify_single(5000, "frames_rx_pripause") == "ACTIVE"
    # tx PFC scales with node count.
    assert classify_single(150, "frames_tx_pripause", nodes=2) == "NEGLIGIBLE"
    assert classify_single(200, "frames_tx_pripause", nodes=2) == "ACTIVE"


def test_group_single_uses_flat_1000():
    assert classify_group_single(0) == "CLEAN"
    assert classify_group_single(999) == "NEGLIGIBLE"
    assert classify_group_single(40000) == "ACTIVE"


@pytest.mark.parametrize("run,base,expected", [
    (0, 0, "CLEAN"),
    (500, 0, "IGNORED"),          # below threshold, one side zero
    (500, 400, "IGNORED"),        # below threshold, > 10% apart
    (500, 480, "NEGLIGIBLE"),     # below threshold, close
    (2000, 0, "REGRESSION"),
    (0, 2000, "IMPROVED"),
    (2005, 2000, "NO CHANGE"),    # < 1%
    (2100, 2000, "NO CHANGE"),    # 5%
    (2200, 2000, "REGRESSION"),   # +10%
    (1800, 2000, "IMPROVED"),     # -10%
])
def test_compare_rules(run, base, expected):
    assert classify_compare(run, base, "tx_rdma_ack_timeout") == expected


def test_compare_group_thresholds():
    assert classify_compare(40000, 0, "__group_PFC__") == "IGNORED"
    assert classify_compare(60000, 0, "__group_PFC__") == "REGRESSION"
    assert classify_compare(9000, 0, "__group_CNP__") == "IGNORED"
    assert classify_compare(1000, 0, "__group_Errors__") == "REGRESSION"


def test_node_classes():
    assert classify_node(60, 100) == "HOT"
    assert classify_node(50, 100) == "MODERATE"
    assert classify_node(10, 100) == "COLD"
    assert classify_node(0, 0) == "COLD"


def test_formats():
    assert fmt_count(999) == "999"
    assert fmt_count(1000) == "1K"
    assert fmt_count(1505) == "1.51K"   # half up
    assert fmt_count(2500000) == "2.50M"
    assert fmt_count(3 * 10**9) == "3G"
    assert fmt_delta(0, 0) == "-"
    assert fmt_delta(5, 0) == "+inf"
    assert fmt_delta(110, 100) == "+10.0%"


# --- Measurements --------------------------------------------------------------
def test_per_size_is_average_per_iteration_after_warmup(tmp_path):
    lines = _chain(8, rdma={"resp_rx_outouf_seq": 3})
    runs = load([_write(tmp_path, lines)], warmup=2, label="r")
    row = per_size(runs[0])[("AllReduce", S)]
    assert (row.iters_used, row.iters_total) == (6, 6)
    assert row.per_iter["resp_rx_outouf_seq"] == 3
    assert row.per_iter["tx_rdma_ucast_bytes"] == S


def test_per_size_skips_unusable_windows(tmp_path):
    lines = [_rec(0, rdma={"resp_rx_outof_buf": 9}, overlapped=True),
             _rec(1, unavailable=True), _rec(2, rdma={"resp_rx_outof_buf": 1})]
    row = per_size(load([_write(tmp_path, lines)], warmup=0, label="r")[0])[("AllReduce", S)]
    assert (row.iters_used, row.iters_total) == (1, 3)
    assert row.per_iter["resp_rx_outof_buf"] == 1


def test_shared_nic_counted_once(tmp_path):
    # Two ranks on one host read the same NIC: count it once (lowest rank).
    a = _write(tmp_path, _chain(3, rdma={"rx_rdma_cnp_pkts": 5}, n_ranks=2), rank=0)
    b = _write(tmp_path, [l.replace('"rank": 0', '"rank": 1') for l in
                          _chain(3, rdma={"rx_rdma_cnp_pkts": 5}, n_ranks=2)], rank=1)
    run = load([a, b], warmup=0, label="r")[0]
    assert per_size(run)[("AllReduce", S)].per_iter["rx_rdma_cnp_pkts"] == 5
    assert totals(whole_run(run))["rx_rdma_cnp_pkts"] == 15


def test_whole_run_from_first_and_last_snapshot(tmp_path):
    run = load([_write(tmp_path, _chain(4, rdma={"tx_rdma_ack_timeout": 2}))], warmup=0, label="r")[0]
    t = totals(whole_run(run))
    assert t["tx_rdma_ack_timeout"] == 8
    assert t["tx_rdma_ucast_bytes"] == 4 * S


def test_warmup_drops_same_sns_on_every_rank(tmp_path):
    a = _write(tmp_path, [_rec(sn, n_ranks=2) for sn in range(5, 10)], rank=0)
    b = _write(tmp_path, [_rec(sn, rank=1, n_ranks=2) for sn in range(5, 10)],
               rank=1, host="nodeB")
    run = load([a, b], warmup=2, label="r")[0]
    assert sorted({w.sn for w in run.windows}) == [7, 8, 9]


# --- Report --------------------------------------------------------------------
def test_single_run_report(tmp_path, capsys):
    _write(tmp_path, _chain(6, rdma={"rx_rdma_cnp_pkts": 2000}))
    assert main([str(tmp_path), "--warmup", "1"]) == 0
    out = capsys.readouterr().out
    assert "Performance : 10.00 GB/s (avg busbw)" in out
    assert "CNP:         ACTIVE" in out
    assert "Errors:      CLEAN" in out
    assert "OVERALL VERDICT" not in out
    assert "CNP at 1M: rx_rdma_cnp_pkts 2K (per iteration)" in out
    assert "per-size counts approximate" in out
    assert "(no PFC frames)" in out


def test_strict_run_is_exact(tmp_path, capsys):
    _write(tmp_path, _chain(3, coordinated=True))
    main([str(tmp_path), "--warmup", "0"])
    out = capsys.readouterr().out
    assert "Counter accuracy: exact per collective" in out
    assert "Counts near size boundaries" not in out


def test_compare_regression(tmp_path, capsys):
    _write(tmp_path, _chain(4, rdma={"resp_rx_outouf_seq": 3000}), sub="run")
    _write(tmp_path, _chain(4), sub="base")
    main([str(tmp_path / "run"), "--baseline", str(tmp_path / "base"), "--warmup", "0"])
    out = capsys.readouterr().out
    assert "Errors:      ACTIVE / REGRESSION" in out
    assert "OVERALL VERDICT: REGRESSION DETECTED" in out
    assert "Errors at 1M: resp_rx_outouf_seq 3K (per iteration)" in out


def test_compare_no_regression(tmp_path, capsys):
    _write(tmp_path, _chain(4, rdma={"rx_rdma_cnp_pkts": 10}), sub="run")
    _write(tmp_path, _chain(4, rdma={"rx_rdma_cnp_pkts": 10}), sub="base")
    main([str(tmp_path / "run"), "--baseline", str(tmp_path / "base"), "--warmup", "0"])
    out = capsys.readouterr().out
    assert "Performance : NO CHANGE" in out
    assert "OVERALL VERDICT: NO REGRESSION" in out


def test_compare_size_mismatch_is_invalid(tmp_path, capsys):
    _write(tmp_path, _chain(3), sub="run")
    _write(tmp_path, _chain(3, size=4 * S), sub="base")
    main([str(tmp_path / "run"), "--baseline", str(tmp_path / "base"), "--warmup", "0"])
    cap = capsys.readouterr()
    assert "OVERALL VERDICT: INVALID COMPARISON" in cap.out
    assert "different message sizes" in cap.err


def test_compare_mode_mismatch_not_comparable(tmp_path, capsys):
    _write(tmp_path, _chain(3), sub="run")
    _write(tmp_path, _chain(3, coordinated=True), sub="base")
    main([str(tmp_path / "run"), "--baseline", str(tmp_path / "base"), "--warmup", "0"])
    cap = capsys.readouterr()
    assert "Performance : NOT COMPARABLE" in cap.out
    assert "accuracy modes differ" in cap.err


def test_pfc_priority_pause_time(tmp_path, capsys):
    lines = []
    for sn in range(2):
        eth = {"frames_rx_pripause": 100 * (sn + 1), "frames_rx_pri_3": 100 * (sn + 1),
               "rx_pripause_3_1us_count": 59 * (sn + 1)}
        rec = json.loads(_rec(sn))
        before = {"frames_rx_pripause": 100 * sn, "frames_rx_pri_3": 100 * sn,
                  "rx_pripause_3_1us_count": 59 * sn}
        rec["net_counters"]["snapshot_before"]["devices"][0]["ethtool"] = before
        rec["net_counters"]["snapshot_after"]["devices"][0]["ethtool"] = eth
        lines.append(json.dumps(rec))
    _write(tmp_path, lines)
    main([str(tmp_path), "--warmup", "0"])
    out = capsys.readouterr().out
    # Span: first read start 0 -> last read end 1600 us; 118 us paused on priority 3.
    assert re.search(r"frames_rx_pripause +200 +NEGLIGIBLE", out)
    assert re.search(r"nodeA +rdma0 +3 +200 +0 +7.375% +0.000%", out)


def test_detail_and_sizes_filter(tmp_path, capsys):
    _write(tmp_path, _chain(3) + _chain(3, size=4 * S))
    main([str(tmp_path), "--warmup", "0", "--detail", "--sizes", "4M"])
    out = capsys.readouterr().out
    assert "=== Detail: nodeA" in out
    table = out.split("=== Per-Size Counters")[1].split("\n\n")[0]
    assert "4M" in table and " 1M " not in table


def test_no_net_counters_exits_1(tmp_path):
    (tmp_path / "accl_profiler_rank0_h_pid1_0x1.jsonl").write_text(
        json.dumps({"header": {"rank": 0}, "coll_perf": {"coll": "AllReduce"}}) + "\n")
    assert main([str(tmp_path)]) == 1


def test_missing_path_exits_2(tmp_path):
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / "nope")])
    assert e.value.code == 2


def test_trace(tmp_path):
    _write(tmp_path, _chain(2, rdma={"resp_rx_outof_buf": 1}))
    out = tmp_path / "t.json"
    assert main([str(tmp_path), "--warmup", "0", "--trace", str(out)]) == 0
    ev = json.loads(out.read_text())["traceEvents"]
    windows = [e for e in ev if e.get("ph") == "X" and e["tid"] % 10 == 2]
    assert len(windows) == 2
    assert windows[0]["args"]["drops"] == {"OOB (resp_rx_outof_buf)": 1}
    assert windows[0]["args"]["window"] == "suspect"
    assert any(e.get("ph") == "C" and e["name"] == "rank 0 drops" and e["args"] == {"OOB": 1}
               for e in ev)


def test_trace_marks_exact_and_approximate(tmp_path):
    _write(tmp_path, _chain(1, coordinated=True), sub="s")
    _write(tmp_path, _chain(1), sub="p")
    for sub, quality, color in (("s", "exact", "good"), ("p", "approximate", "yellow")):
        out = tmp_path / f"{sub}.json"
        main([str(tmp_path / sub), "--warmup", "0", "--trace", str(out)])
        w = [e for e in json.loads(out.read_text())["traceEvents"]
             if e.get("ph") == "X" and e["tid"] % 10 == 2][0]
        assert (w["args"]["window"], w["cname"]) == (quality, color)
    assert build_trace([]) ["traceEvents"] == []
