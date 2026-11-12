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
         rdma=None, ethtool=None, launch_wait_us=1500, net=True, n_ranks=2):
    rec = {
        "header": {"rank": rank, "n_ranks": n_ranks},
        "coll_perf": {"coll": "AllReduce", "coll_sn": sn, "coll_msg_size_bytes": size,
                      "coll_exec_time_us": exec_us, "decomposition": {}},
    }
    if net:
        counters = {"tx_rdma_ucast_bytes": size, "tx_rdma_ucast_pkts": 256,
                    "rx_rdma_ucast_bytes": size, "rx_rdma_ucast_pkts": 256}
        counters.update(rdma or {})
        rec["net_counters"] = {
            "overlapped": overlapped, "unavailable": unavailable,
            "completion_verified": True, "coordinated": True,
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


KEY = ("0xabc", "AllReduce", S)


def _verdict(tmp_path, lines):
    # Single-rank fixtures must declare a single-rank communicator.
    lines = [json.dumps({**json.loads(line), 'header': {**json.loads(line)['header'], 'n_ranks': 1}}) for line in lines]
    recs, _ = load([_write(tmp_path, lines)], warmup=0)
    return verdicts(recs)[KEY]


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
    v = verdicts(recs)[KEY]
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
    lines += [_rec(10, overlapped=True, rdma={"resp_rx_outouf_seq": 9})]
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
    v = _verdict(tmp_path, lines)
    assert v.verdict == "INCONCLUSIVE"
    assert "not ionic" in v.reason


def test_no_traffic_is_inconclusive(tmp_path):
    v = _verdict(tmp_path, [_rec(i, rdma={"tx_rdma_ucast_bytes": 0, "rx_rdma_ucast_bytes": 0})
                            for i in range(3)])
    assert v.verdict == "INCONCLUSIVE"
    assert "no NIC traffic" in v.reason


def test_stale_ethtool_never_drives_a_verdict(tmp_path):
    eth = {"stale": True, "frames_rx_pri_3": 1000}
    v = _verdict(tmp_path, [_rec(i, ethtool=eth) for i in range(3)])
    assert v.verdict == "CLEAN"
    assert (v.pause_rx, v.pause_tx) == (3000, 0)


def test_never_usable_rank_is_inconclusive(tmp_path):
    a = _write(tmp_path, [_rec(i, rank=0) for i in range(15)], rank=0)
    b = _write(tmp_path, [_rec(i, rank=1, unavailable=True) for i in range(15)], rank=1)
    recs, _ = load([a, b], warmup=0)
    v = verdicts(recs)[KEY]
    assert v.verdict == "INCONCLUSIVE"
    assert "rank(s) 1" in v.reason


def test_unlisted_counters_are_shown_not_judged(tmp_path):
    v = _verdict(tmp_path, [_rec(i, rdma={"req_rx_dup_response": 5,
                                          "tx_rdma_ccl_cts_pkts": 9}) for i in range(3)])
    assert v.verdict == "INCONCLUSIVE"
    assert v.other == {"req_rx_dup_response": (15, 3, [0])}


def test_records_without_net_counters_are_counted(tmp_path):
    v = _verdict(tmp_path, [_rec(i) for i in range(3)] + [_rec(9, net=False)])
    assert v.n_no_net == 1
    assert v.n_records == 3


def test_comms_are_judged_separately(tmp_path):
    a = _write(tmp_path, [_rec(i, n_ranks=1) for i in range(3)], comm="0x1")
    b = _write(tmp_path, [_rec(i, n_ranks=1, rdma={"tx_rdma_ack_timeout": 1}) for i in range(3)], comm="0x2")
    recs, _ = load([a, b], warmup=0)
    vs = verdicts(recs)
    assert vs[("0x1", "AllReduce", S)].verdict == "CLEAN"
    assert vs[("0x2", "AllReduce", S)].verdict == "NETWORK_SUSPECT"


def test_warmup_drops_the_same_sns_on_every_rank(tmp_path):
    # Rank 1 is missing sn 0: dropping per file would desync the skew groups.
    a = _write(tmp_path, [_rec(i, rank=0) for i in range(4)], rank=0)
    b = _write(tmp_path, [_rec(i, rank=1) for i in range(1, 4)], rank=1)
    recs, _ = load([a, b], warmup=2)
    assert sorted((r.rank, r.sn) for r in recs) == [(0, 2), (0, 3), (1, 2), (1, 3)]


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


def test_skew_correction_skips_groups_with_a_timed_out_read(tmp_path):
    a = _write(tmp_path, [_rec(0, rank=0, exec_us=900.0, launch_wait_us=1000)], rank=0)
    b = _write(tmp_path, [_rec(0, rank=1, exec_us=310.0, launch_wait_us=1000000,
                               unavailable=True)], rank=1)
    recs, _ = load([a, b], warmup=0)
    correct_skew(recs)
    assert all(r.corrected_us is None for r in recs)


def test_skew_correction_skips_incomplete_groups(tmp_path):
    a = _write(tmp_path, [_rec(0, rank=0, exec_us=900.0, launch_wait_us=1000, n_ranks=3)], rank=0)
    b = _write(tmp_path, [_rec(0, rank=1, exec_us=310.0, launch_wait_us=1600, n_ranks=3)], rank=1)
    recs, _ = load([a, b], warmup=0)
    correct_skew(recs)
    assert all(r.corrected_us is None for r in recs)


def test_skew_correction_stays_within_a_comm(tmp_path):
    a = _write(tmp_path, [_rec(0, rank=0, exec_us=500.0, launch_wait_us=1000, n_ranks=1)],
               rank=0, comm="0x1")
    b = _write(tmp_path, [_rec(0, rank=1, exec_us=500.0, launch_wait_us=5000, n_ranks=1)],
               rank=1, comm="0x2")
    recs, _ = load([a, b], warmup=0)
    correct_skew(recs)
    assert all(r.corrected_us == 500.0 for r in recs)


def test_trace_has_windows_reads_and_counters(tmp_path):
    a = _write(tmp_path, [_rec(0, n_ranks=1), _rec(1, rdma={"resp_rx_outouf_seq": 2}, n_ranks=1)],
               rank=0)
    recs, _ = load([a], warmup=0)
    correct_skew(recs)
    t = build_trace(recs)
    ev = t["traceEvents"]
    windows = [e for e in ev if e["ph"] == "X" and e["cat"] != "read"]
    assert [w["cat"] for w in windows] == ["window", "suspect"]
    assert windows[1]["args"]["drops"] == {"OOS (resp_rx_outouf_seq)": 2}
    assert windows[0]["ts"] == 0 and windows[0]["dur"] == 600
    assert sum(1 for e in ev if e.get("cat") == "read") == 2
    assert any(e["ph"] == "M" and e["args"]["name"].startswith("nodeA") for e in ev)
    drops = [e["args"] for e in ev if e["ph"] == "C" and e["name"].endswith("drops")]
    assert drops == [{"OOS": 0}, {"OOS": 2}]
    assert windows[0]["args"]["exec_corrected_us"] == 100.0


def test_trace_keeps_comms_on_separate_threads(tmp_path):
    a = _write(tmp_path, [_rec(0, n_ranks=1)], comm="0x1")
    b = _write(tmp_path, [_rec(0, n_ranks=1)], comm="0x2")
    recs, _ = load([a, b], warmup=0)
    ev = build_trace(recs)["traceEvents"]
    tids = {e["tid"] for e in ev if e["ph"] == "X" and e.get("cat") != "read"}
    assert len(tids) == 2


def test_main_prints_report_and_writes_trace(tmp_path, capsys):
    path = _write(tmp_path, [_rec(i, n_ranks=1) for i in range(4)] +
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


def test_inputs_are_found_recursively(tmp_path):
    sub = tmp_path / "run" / "r1"
    sub.mkdir(parents=True)
    _write(sub, [_rec(0)])
    recs, _ = load([str(tmp_path / "run")], warmup=0)
    assert len(recs) == 1


def test_main_without_net_counters_fails(tmp_path, capsys):
    path = _write(tmp_path, [_rec(i, net=False) for i in range(3)])
    assert main([path, "--warmup", "0"]) == 1
    assert "ACCL_NET_COUNTERS=1" in capsys.readouterr().out


def test_skew_groups_by_sn(tmp_path):
    a = _write(tmp_path, [_rec(0, rank=0, exec_us=900.0, launch_wait_us=1000),
                          _rec(1, rank=0, exec_us=400.0, launch_wait_us=1000)], rank=0)
    b = _write(tmp_path, [_rec(0, rank=1, exec_us=310.0, launch_wait_us=1600),
                          _rec(1, rank=1, exec_us=400.0, launch_wait_us=1000)], rank=1)
    recs, _ = load([a, b], warmup=0)
    correct_skew(recs)
    got = {(r.rank, r.sn): r.corrected_us for r in recs}
    assert got == {(0, 0): 300.0, (1, 0): 310.0, (0, 1): 400.0, (1, 1): 400.0}


def test_skew_correction_clamps_at_zero(tmp_path):
    a = _write(tmp_path, [_rec(0, rank=0, exec_us=100.0, launch_wait_us=0)], rank=0)
    b = _write(tmp_path, [_rec(0, rank=1, exec_us=100.0, launch_wait_us=5000)], rank=1)
    recs, _ = load([a, b], warmup=0)
    correct_skew(recs)
    assert {r.rank: r.corrected_us for r in recs} == {0: 0.0, 1: 100.0}


def test_tx_per_coll_and_pause_use_usable_records_only(tmp_path):
    eth = {"stale": True, "frames_tx_pri_3": 4, "rx_pripause_3_1us_count": 99}
    lines = [_rec(i, ethtool=eth) for i in range(3)]
    lines.append(_rec(9, overlapped=True, rdma={"tx_rdma_ucast_bytes": 10 * S}, ethtool=eth))
    v = _verdict(tmp_path, lines)
    assert v.tx_bytes_per_coll == S
    assert (v.pause_rx, v.pause_tx) == (0, 12)


def test_devices_are_summed(tmp_path):
    r = json.loads(_rec(0))
    r["net_counters"]["devices"].append(
        {"name": "rdma1", "rdma": {"tx_rdma_ucast_bytes": S, "req_rx_pkt_seq_err": 1},
         "ethtool": None})
    lines = [json.dumps(r)] + [_rec(i) for i in range(1, 3)]
    v = _verdict(tmp_path, lines)
    assert v.drivers["req_rx_pkt_seq_err"] == (1, 1, [0])
    assert v.tx_bytes_per_coll == S * 4 / 3


def test_trace_marks_overlapped_windows(tmp_path):
    a = _write(tmp_path, [_rec(0, overlapped=True, n_ranks=1)])
    recs, _ = load([a], warmup=0)
    cats = [e["cat"] for e in build_trace(recs)["traceEvents"] if e["ph"] == "X"]
    assert cats == ["overlapped", "read"]


def test_incomplete_summary_warns(tmp_path, capsys):
    path = _write(tmp_path, [_rec(0, n_ranks=1),
                             json.dumps({"summary": {"complete": False, "net_counters": {
                                 "records": 1, "queued_at_finalize": 1}}})])
    main([path, "--warmup", "0"])
    captured = capsys.readouterr()
    assert "complete=false" in captured.err
    assert "1 not written by finalize" in captured.out


def test_bad_lines_and_foreign_files_are_tolerated(tmp_path):
    p = tmp_path / "other.jsonl"
    p.write_text("not json\n" + _rec(0, n_ranks=1) + "\n")
    recs, _ = load([str(p)], warmup=0)
    assert len(recs) == 1 and recs[0].host == "?"


def test_missing_rank_file_cannot_be_clean(tmp_path):
    path = _write(tmp_path, [_rec(i, rank=0, n_ranks=2) for i in range(3)])
    rs, _ = load([path], warmup=0)
    v = verdicts(rs)[KEY]
    assert v.verdict == 'INCONCLUSIVE'
    assert 'rank(s) 1' in v.reason


def test_rank_without_counters_cannot_be_clean(tmp_path):
    a = _write(tmp_path, [_rec(i, rank=0) for i in range(3)])
    b = _write(tmp_path, [_rec(i, rank=1, net=False) for i in range(3)], rank=1)
    rs, _ = load([a, b], warmup=0)
    assert verdicts(rs)[KEY].verdict == 'INCONCLUSIVE'


def test_raw_snapshots_are_deduplicated_and_activity_is_separate(tmp_path):
    lines = []
    for i in range(3):
        r = json.loads(_rec(i, n_ranks=1))
        n = r['net_counters']
        n.update(completion_verified=True, coordinated=True,
                 collective_start_cpu_us=1000*i+150, collective_complete_cpu_us=1000*i+500,
                 snapshot_before={'id': i, 'read_start_us': 1000*i, 'read_end_us': 1000*i+100},
                 snapshot_after={'id': i+1, 'read_start_us': 1000*(i+1), 'read_end_us': 1000*(i+1)+100})
        lines.append(json.dumps(r))
    rs, _ = load([_write(tmp_path, lines)], warmup=0)
    es = build_trace(rs)['traceEvents']
    reads = [e for e in es if e.get('cat') == 'read']
    activity = [e for e in es if e.get('cat') == 'collective']
    windows = [e for e in es if e.get('cat') == 'window']
    assert len(reads) == 4 and len(activity) == len(windows) == 3
    assert activity[0]['dur'] == 350 and windows[0]['dur'] == 900
    assert activity[0]['tid'] != windows[0]['tid'] != reads[0]['tid']
    assert activity[0]["args"]["coordinated"] is True


def test_discontinuous_counter_window_is_not_usable(tmp_path):
    lines = []
    for i in range(3):
        r = json.loads(_rec(i, n_ranks=1))
        r['net_counters']['counter_discontinuities'] = ['rdma0/resp_rx_outouf_seq']
        lines.append(json.dumps(r))
    rs, _ = load([_write(tmp_path, lines)], warmup=0)
    assert verdicts(rs)[KEY].verdict == 'INCONCLUSIVE'
    assert not any(e['ph'] == 'C' for e in build_trace(rs)['traceEvents'])


def test_uncoordinated_windows_cannot_be_clean(tmp_path):
    lines = []
    for i in range(3):
        r = json.loads(_rec(i, n_ranks=1))
        r['net_counters']['coordinated'] = False
        lines.append(json.dumps(r))
    rs, _ = load([_write(tmp_path, lines)], warmup=0)
    assert verdicts(rs)[KEY].verdict == 'INCONCLUSIVE'


def test_legacy_windows_without_completion_proof_cannot_be_clean(tmp_path):
    lines = []
    for i in range(3):
        r = json.loads(_rec(i, n_ranks=1))
        r['net_counters'].pop('completion_verified')
        r['net_counters'].pop('coordinated')
        lines.append(json.dumps(r))
    rs, _ = load([_write(tmp_path, lines)], warmup=0)
    assert verdicts(rs)[KEY].verdict == 'INCONCLUSIVE'


def _passive(lines):
    out = []
    for line in lines:
        r = json.loads(line)
        r["net_counters"]["coordinated"] = False
        out.append(json.dumps(r))
    return out


def test_passive_loss_is_still_suspect(tmp_path):
    v = _verdict(tmp_path, _passive([_rec(i, rdma={"resp_rx_outouf_seq": 2}) for i in range(3)]))
    assert v.verdict == "NETWORK_SUSPECT"
    assert v.reason == "loss/retransmit (uncoordinated windows)"


def test_passive_without_symptoms_cannot_be_clean(tmp_path):
    v = _verdict(tmp_path, _passive([_rec(i) for i in range(4)]))
    assert v.verdict == "INCONCLUSIVE"
    assert "ACCL_NET_STRICT=1" in v.reason


def test_trace_shows_only_selected_counters(tmp_path):
    eth = {"stale": True, "frames_rx_pri_3": 7, "rx_pripause_3_1us_count": 500}
    a = _write(tmp_path, [_rec(0, n_ranks=1, ethtool=eth,
                               rdma={"rx_rdma_cnp_pkts": 3, "req_rx_dup_response": 1})])
    recs, _ = load([a], warmup=0)
    ev = build_trace(recs)["traceEvents"]
    w = [e for e in ev if e["ph"] == "X" and e["cat"] != "read"][0]["args"]
    assert w["tx_bytes"] == S and w["rx_bytes"] == S
    assert w["ecn_cnp"] == {"CNP received (rx_rdma_cnp_pkts)": 3}
    assert w["pfc_frames (stale)"] == {"rx prio 3": 7}
    assert "drops" not in w
    assert not any("dup_response" in k for k in w)
    tracks = {e["name"] for e in ev if e["ph"] == "C"}
    assert tracks == {"rank 0 MB per collective", "rank 0 ECN/CNP", "rank 0 PFC frames (stale)"}
