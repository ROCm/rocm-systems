"""Unit tests for net_tel_report.py (ACCL network-counter report)."""
import json
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "plugins", "profiler", "accl")
)
from net_tel_report import (  # noqa: E402
    load, verdicts, correct_skew, build_trace, main,
)

S = 1048576


def _rec(sn, rank=0, size=S, exec_us=100.0, overlapped=False, unavailable=False,
         rdma=None, ethtool=None, launch_wait_us=1500, net=True):
    rec = {
        "header": {"rank": rank, "n_ranks": 2},
        "coll_perf": {"coll": "AllReduce", "coll_sn": sn, "coll_msg_size_bytes": size,
                      "coll_exec_time_us": exec_us, "decomposition": {}},
    }
    if net:
        counters = {"tx_rdma_ucast_bytes": size, "tx_rdma_ucast_pkts": 256,
                    "rx_rdma_ucast_bytes": size, "rx_rdma_ucast_pkts": 256}
        counters.update(rdma or {})
        rec["net_counters"] = {
            "overlapped": overlapped, "unavailable": unavailable,
            "before_ts_us": 1000 * sn, "after_ts_us": 1000 * sn + 600,
            "after_read_us": 1400, "launch_wait_us": launch_wait_us,
            "devices": [{"name": "rdma0", "rdma": None if unavailable else counters,
                         "ethtool": ethtool}],
            "sw": None,
        }
    return json.dumps(rec)


def _write(tmp_path, lines, rank=0, host="nodeA", comm="0xabc"):
    p = tmp_path / f"accl_profiler_rank{rank}_{host}_pid1{rank}_{comm}.jsonl"
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _verdict(tmp_path, lines):
    recs, _ = load([_write(tmp_path, lines)], warmup=0)
    return verdicts(recs)[("AllReduce", S)]


def test_clean_with_enough_usable_records(tmp_path):
    v = _verdict(tmp_path, [_rec(i) for i in range(4)])
    assert v.verdict == "CLEAN"
    assert v.n_usable == 4
    assert v.tx_bytes_per_coll == S


def test_loss_on_receiver_pools_ranks(tmp_path):
    a = _write(tmp_path, [_rec(i, rank=0) for i in range(3)], rank=0)
    b = _write(tmp_path, [_rec(i, rank=1, rdma={"resp_rx_outouf_seq": 4})
                          for i in range(3)], rank=1)
    recs, _ = load([a, b], warmup=0)
    v = verdicts(recs)[("AllReduce", S)]
    assert v.verdict == "NETWORK_SUSPECT"
    assert v.reason == "loss/retransmit"
    assert v.drivers["resp_rx_outouf_seq"] == (12, 3, [1])


def test_congestion_only_is_named(tmp_path):
    v = _verdict(tmp_path, [_rec(i, rdma={"rx_rdma_cnp_pkts": 2}) for i in range(3)])
    assert v.verdict == "NETWORK_SUSPECT"
    assert "congestion" in v.reason


def test_loss_wins_over_congestion(tmp_path):
    v = _verdict(tmp_path, [_rec(i, rdma={"rx_rdma_cnp_pkts": 2, "req_rx_pkt_seq_err": 1})
                            for i in range(3)])
    assert v.reason == "loss/retransmit"


def test_overlapped_and_unavailable_are_excluded(tmp_path):
    lines = [_rec(i) for i in range(3)]
    lines += [_rec(10, overlapped=True, rdma={"tx_rdma_retx_pkts": 9})]
    lines += [_rec(11, unavailable=True)]
    v = _verdict(tmp_path, lines)
    assert v.verdict == "CLEAN"
    assert (v.n_usable, v.n_overlapped, v.n_unavailable) == (3, 1, 1)


def test_inconclusive_when_too_few_or_mostly_overlapped(tmp_path):
    assert _verdict(tmp_path, [_rec(i) for i in range(2)]).verdict == "INCONCLUSIVE"
    lines = [_rec(i) for i in range(3)] + [_rec(10 + i, overlapped=True) for i in range(4)]
    v = _verdict(tmp_path, lines)
    assert v.verdict == "INCONCLUSIVE"
    assert "4 overlapped" in v.reason


def test_half_usable_is_not_inconclusive(tmp_path):
    lines = [_rec(i) for i in range(3)] + [_rec(10 + i, overlapped=True) for i in range(3)]
    assert _verdict(tmp_path, lines).verdict == "CLEAN"


def test_unknown_driver_is_inconclusive(tmp_path):
    lines = []
    for i in range(3):
        r = json.loads(_rec(i))
        r["net_counters"]["devices"][0]["rdma"] = {"out_of_sequence": 0}
        lines.append(json.dumps(r))
    assert _verdict(tmp_path, lines).verdict == "INCONCLUSIVE"


def test_no_traffic_is_inconclusive(tmp_path):
    v = _verdict(tmp_path, [_rec(i, rdma={"tx_rdma_ucast_bytes": 0, "rx_rdma_ucast_bytes": 0})
                            for i in range(3)])
    assert v.verdict == "INCONCLUSIVE"
    assert "no NIC traffic" in v.reason


def test_stale_ethtool_never_drives_a_verdict(tmp_path):
    eth = {"stale": True, "frames_rx_pripause": 1000}
    v = _verdict(tmp_path, [_rec(i, ethtool=eth) for i in range(3)])
    assert v.verdict == "CLEAN"
    assert v.pause_frames == 3000


def test_warmup_drops_lowest_sn_per_size(tmp_path):
    recs, _ = load([_write(tmp_path, [_rec(i) for i in (4, 3, 2, 1, 0)])], warmup=2)
    assert sorted(r.sn for r in recs) == [2, 3, 4]


def test_filename_gives_host_and_comm(tmp_path):
    recs, _ = load([_write(tmp_path, [_rec(0, rank=3)], rank=3, host="pit2-p03-g63", comm="0xdead")],
                   warmup=0)
    assert (recs[0].rank, recs[0].host, recs[0].comm) == (3, "pit2-p03-g63", "0xdead")


def test_skew_correction_uses_the_slowest_rank(tmp_path):
    # Rank 0's read ended 600 us before rank 1's: its kernel waited that long.
    a = _write(tmp_path, [_rec(0, rank=0, exec_us=900.0, launch_wait_us=1000)], rank=0)
    b = _write(tmp_path, [_rec(0, rank=1, exec_us=310.0, launch_wait_us=1600)], rank=1)
    recs, _ = load([a, b], warmup=0)
    correct_skew(recs)
    got = {r.rank: r.corrected_us for r in recs}
    assert got == {0: 300.0, 1: 310.0}


def test_skew_correction_stays_within_a_comm(tmp_path):
    a = _write(tmp_path, [_rec(0, rank=0, exec_us=500.0, launch_wait_us=1000)],
               rank=0, comm="0x1")
    b = _write(tmp_path, [_rec(0, rank=1, exec_us=500.0, launch_wait_us=5000)],
               rank=1, comm="0x2")
    recs, _ = load([a, b], warmup=0)
    correct_skew(recs)
    assert all(r.corrected_us == 500.0 for r in recs)


def test_trace_has_windows_reads_and_counters(tmp_path):
    a = _write(tmp_path, [_rec(0), _rec(1, rdma={"tx_rdma_retx_pkts": 2})], rank=0)
    recs, _ = load([a], warmup=0)
    correct_skew(recs)
    t = build_trace(recs)
    ev = t["traceEvents"]
    windows = [e for e in ev if e["ph"] == "X" and e["cat"] != "read"]
    assert [w["cat"] for w in windows] == ["window", "suspect"]
    assert windows[1]["args"]["tx_rdma_retx_pkts"] == 2
    assert windows[0]["ts"] == 0 and windows[0]["dur"] == 600
    assert sum(1 for e in ev if e.get("cat") == "read") == 2
    assert any(e["ph"] == "M" and e["args"]["name"] == "nodeA" for e in ev)
    assert all(e["ph"] != "C" or "tx_MB" in e["args"] for e in ev)


def test_main_prints_report_and_writes_trace(tmp_path, capsys):
    path = _write(tmp_path, [_rec(i) for i in range(4)] +
                  [json.dumps({"summary": {"complete": True, "net_counters": {
                      "records": 4, "overlapped": 0, "unavailable": 0, "reads": 4,
                      "read_us_mean": 1400.0, "read_us_max": 1500}}})])
    trace = tmp_path / "t.json"
    assert main([path, "--warmup", "0", "--trace", str(trace)]) == 0
    out = capsys.readouterr().out
    assert "CLEAN" in out
    assert "EXEC TIME" in out
    assert "Counter reads: 4, mean 1400 us" in out
    assert json.loads(trace.read_text())["traceEvents"]


def test_main_without_net_counters_fails(tmp_path, capsys):
    path = _write(tmp_path, [_rec(i, net=False) for i in range(3)])
    assert main([path, "--warmup", "0"]) == 1
    assert "ACCL_NET_COUNTERS=1" in capsys.readouterr().out
