#!/usr/bin/env python3
"""Tests for accl_fabric_report.py on synthetic accl + telemetry output.

Run: python3 test_accl_fabric_report.py
"""
import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import accl_fabric_report as fr  # noqa: E402

MB = 1 << 20
PRIO = 3
RCCL_BPUS = 10_000.0      # RCCL bytes per us (10 GB/s)
BUSY = (1_000_000, 1_400_000)


def accl_line(rank, host, pid, coll, size, sn, start, end, comm="0x1", n_ranks=2, cq_errors=0):
    cp = {"coll": coll, "coll_sn": sn, "coll_msg_size_bytes": size, "coll_algo": "RING",
          "coll_proto": "SIMPLE", "coll_n_channels": 4, "coll_exec_time_us": end - start,
          "coll_algobw_gbs": 1.0, "coll_busbw_gbs": 2.0, "coll_timing_source": "gpu_globaltimer",
          "decomposition": {"enqueue_to_kernel_us": 1, "gpu_kernel_avg_us": end - start,
                            "proxy_network_us": 10, "proxy_peer_wait_us": 10, "n_proxy_ops": 2},
          "host_ts": {"clock": "CLOCK_MONOTONIC", "enqueue_us": start - 5,
                      "exec_start_us": start, "exec_end_us": end},
          "net": {"source": "rccl_telemetry_sw", "window_start_us": start, "window_end_us": end,
                  "devices": [{"device_id": 0, "tx_bytes": size, "rx_bytes": size,
                               "completion_hist": [6, 2], "cq_errors": cq_errors}]}}
    return json.dumps({"header": {"rank": rank, "n_ranks": n_ranks, "host": host, "pid": pid,
                                  "comm_hash": comm}, "coll_perf": cp})


def cum(rate, t):
    """Integral up to t of a piecewise-constant rate given as {(a, b): per_us}."""
    return sum(max(0, min(t, b) - a) * r for (a, b), r in (rate or {}).items())


def telemetry(host, pid, roce="nic0", t0=900_000, t1=1_600_000, step=10_000, prio=PRIO,
              rx_pause=None, tx_pause=None, rx_frames=None, tx_frames=None, ecn=None, oos=None, rnr=None, oob=None,
              other_bpus=0.0, link_gbps=400, prio_bytes=True, time_counters=True,
              refresh_us=None, pause_prio=None, no_pfc_prio=None, cnp=None, qos=None, hw_type="x",
              pause_refresh_us=None, seq=None, ack_to=None, drop=()):
    """One telemetry file. Rates are {(a, b): per-us}; pause rates are a fraction
    of time. RCCL sends RCCL_BPUS; `other_bpus` is someone else's traffic on the
    same port. `refresh_us` makes the NIC's ethtool counters step that often.
    `drop` names columns this NIC does not give."""
    pp = prio if pause_prio is None else pause_prio
    samples = []
    t = t0
    while t <= t1:
        tn = t if refresh_us is None else (t // refresh_us) * refresh_us   # NIC-side value time
        z = [0] * 8
        rxp, txp, rxf, txf, pb = list(z), list(z), list(z), list(z), list(z)
        tp = tn if pause_refresh_us is None else (t // pause_refresh_us) * pause_refresh_us
        rxp[pp] = int(cum(rx_pause, tp))
        txp[pp] = int(cum(tx_pause, tn))
        rxf[pp] = int(cum(rx_frames, tn))
        txf[pp] = int(cum(tx_frames, tn))
        nic = (RCCL_BPUS + other_bpus) * tn
        pb[prio] = int(nic)
        if no_pfc_prio is not None:       # the NIC has no PFC counters for this priority
            for a in (rxp, txp, rxf, txf):
                a[no_pfc_prio] = -1
        s = {"ts_us": t, "device_id": 0, "roce_device": roce,
             "tx_bytes": RCCL_BPUS * t, "rx_bytes": RCCL_BPUS * t,
             "nic_tx_bytes": int(nic), "nic_rx_bytes": int(nic),
             "nic_tx_packets": int(nic / 4096), "nic_rx_packets": int(nic / 4096),
             "ecn_marked_pkts": int(cum(ecn, t)), "cnp_sent": int(cum(cnp, t)), "cnp_rcvd": 0,
             "oos_drop_count": int(cum(oos, t)), "seq_err_naks_rcvd": int(cum(seq, t)),
             "local_ack_timeout_err": int(cum(ack_to, t)),
             "out_of_buffer": int(cum(oob, t)), "rnr_retry_err": int(cum(rnr, t)),
             "pfc_rx_pause_us": rxp if time_counters else z, "pfc_tx_pause_us": txp if time_counters else z,
             "pfc_rx_frames": rxf, "pfc_tx_frames": txf}
        if prio_bytes:
            s["prio_tx_bytes"] = pb
            s["prio_rx_bytes"] = pb
        for k in drop:
            s.pop(k, None)
        samples.append(s)
        t += step
    return {"host_name": host, "process_id": str(pid), "clock": "CLOCK_MONOTONIC",
            "histogram_bucket_interval_ns": 30000, "histogram_max_buckets": 5,
            "sample_interval_ms": step // 1000,
            "devices": [{"device_id": 0, "roce_device": roce, "hw_type": hw_type, "link_gbps": link_gbps,
                         **({"qos": qos} if qos is not None else {})}],
            "hw_samples": samples}


def iterations(rank, host, pid, start=BUSY[0], n=20, dur=20_000, gap=0, size=64 * MB, comm="0x1",
               first_sn=0, n_ranks=2, cq_errors=0):
    return [accl_line(rank, host, pid, "AllReduce", size, first_sn + i, start + i * (dur + gap),
                      start + i * (dur + gap) + dur, comm, n_ranks, cq_errors) for i in range(n)]


class Run:
    """A directory holding one synthetic run, analysed with default thresholds."""

    def __init__(self, ranks, pfc_prio=None, baseline=None):
        d = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, d, True)
        self.dirname = d
        for rank, host, pid, lines, tel in ranks:
            if not any('"summary"' in l for l in lines):   # the plugin always ends with one
                lines = lines + [json.dumps({"summary": {"dropped_collectives": 0, "complete": True}})]
            with open(os.path.join(d, f"accl_profiler_rank{rank}_{host}_pid{pid}.jsonl"), "a") as f:
                f.write("\n".join(lines) + "\n")
            if tel is not None:
                with open(os.path.join(d, f"rccl_telemetry_{host}_0_{pid}.json"), "w") as f:
                    json.dump(tel, f)
        recs, _ = fr.load_accl(d, 0)
        self.tels = fr.load_telemetry(d)
        used = {k: fr.devices_used([r for r in recs if (r.host, r.pid) == k], t)
                for k, t in self.tels.items()}
        self.lane = fr.pick_lane(self.tels, used, pfc_prio)
        self.results = fr.analyze(recs, self.tels, self.lane, fr.Thresholds())
        self.drops = {k: fr.diagnose_drops(gs, self.lane, fr.Thresholds(), baseline)
                      for k, gs in fr._by_size(self.results).items()}
        self.summary = fr.job_summary(self.results, self.lane, None, self.drops)

    def drop(self):
        return next(iter(self.drops.values()))

    def rank(self, r):
        return next(g for g in self.results if g.rank == r)

    def cli(self, *extra):
        d = self.dirname
        return subprocess.run([sys.executable, os.path.join(HERE, "accl_fabric_report.py"), *extra,
                               "--accl", d, "--telemetry", d, "--warmup", "0"],
                              capture_output=True, text=True, check=True).stdout


def one(tel=None, lines=None, **kw):
    """Single rank 0 on hostA running 20 back-to-back 64MB AllReduces in BUSY."""
    return Run([(0, "hostA", 1, lines or iterations(0, "hostA", 1, n_ranks=1), tel or telemetry("hostA", 1, **kw))])


def assert_nested(tc, ev):
    """Slices on one track must nest, or Perfetto reports import errors; and no
    event may come before 0."""
    tc.assertFalse([e for e in ev if e.get("ts", 0) < 0], "negative timestamp")
    by = {}
    for e in ev:
        if e["ph"] == "X":
            by.setdefault((e["pid"], e["tid"]), []).append((e["ts"], e["ts"] + e["dur"]))
    for iv in by.values():
        stack = []
        for a, b in sorted(iv, key=lambda x: (x[0], -x[1])):
            while stack and stack[-1] <= a:
                stack.pop()
            tc.assertFalse(stack and b > stack[-1], "slice overlaps its neighbour without nesting")
            stack.append(b)


class TraceTest(unittest.TestCase):
    def test_overlapping_collectives_get_lanes(self):
        # Each collective starts 1 us before the previous one ends.
        lines = [accl_line(0, "hostA", 1, "AllReduce", 64 * MB, i, BUSY[0] + i * 1000,
                           BUSY[0] + i * 1000 + 1001) for i in range(20)]
        r = Run([(0, "hostA", 1, lines, telemetry("hostA", 1))])
        out = os.path.join(r.dirname, "t.json")
        r.cli("trace", "-o", out)
        with open(out) as f:
            ev = json.load(f)["traceEvents"]
        assert_nested(self, ev)
        self.assertEqual(len({e["tid"] for e in ev if e["ph"] == "X"}), 2)

    def test_slow_counter_rate_is_averaged_over_its_steps(self):
        # NIC bytes step once per 100 ms while sampled every 10 ms: rates must
        # be the real ~20 GB/s, not 0 then 10x spikes.
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1),
                  telemetry("hostA", 1, other_bpus=10_000, refresh_us=100_000))])
        out = os.path.join(r.dirname, "t.json")
        r.cli("trace", "-o", out)
        with open(out) as f:
            ev = json.load(f)["traceEvents"]
        nic = [e["args"]["tx"] for e in ev if e["ph"] == "C" and e["name"].startswith("nic0 NIC GB/s")]
        self.assertTrue(nic)
        self.assertTrue(all(15 <= v <= 25 for v in nic[1:-1]), nic[:10])


class BoundsTest(unittest.TestCase):
    def test_union_merges_back_to_back(self):
        self.assertEqual(fr.union([(0, 10), (10, 20), (30, 40), (5, 8)]), [(0, 20), (30, 40)])

    def test_duration_bounds_full_fill(self):
        s = fr.Series([0, 100, 200], [0.0, 10.0, 20.0])
        b = fr.bound(s, [(0, 200)], 0, True)
        self.assertEqual((b.lb, b.ub, b.fill), (20.0, 20.0, 1.0))

    def test_duration_bounds_partial_fill(self):
        # 30 us paused somewhere in a 100 us window; the collective ran 10 us of it.
        s = fr.Series([0, 100], [0.0, 30.0])
        b = fr.bound(s, [(40, 50)], 0, True)
        self.assertEqual((b.lb, b.ub), (0.0, 10.0))

    def test_event_bounds_need_full_fill(self):
        s = fr.Series([0, 100], [0.0, 7.0])
        self.assertEqual(fr.bound(s, [(0, 100)], 0, False).lb, 7.0)
        self.assertEqual(fr.bound(s, [(40, 50)], 0, False).lb, 0.0)

    def test_refresh_lag_keeps_earlier_burst_out_of_lower_bound(self):
        # 1000 frames just before the span reach the counter only in [100, 200]
        # because it refreshes up to 50 us late; only the 10 in [200, 300] are
        # surely inside. Seen on Thor2: a 64M burst charged to the 128M after it.
        s = fr.Series([0, 100, 200, 300, 400], [0.0, 0.0, 1000.0, 1010.0, 1010.0])
        b = fr.bound(s, [(100, 300)], 50, False)
        self.assertEqual((b.lb, b.ub), (10.0, 1010.0))

    def test_counter_that_has_not_refreshed_yet_is_not_zero(self):
        # Samples stop 20 us after the span; the counter shows events 50 us late,
        # so the span's last 30 us may not be in it yet.
        s = fr.Series([0, 100, 200, 320], [0.0, 0.0, 0.0, 0.0])
        self.assertEqual(fr.bound(s, [(100, 300)], 50, False).ub, fr.INF)
        self.assertEqual(fr.bound(s, [(100, 300)], 0, False).ub, 0.0)
        # However short the unseen tail is next to a long run.
        s = fr.Series([0, 150_000_000], [0.0, 0.0])
        self.assertEqual(fr.bound(s, [(1000, 149_900_000)], 1_000_000, False).ub, fr.INF)

    def test_back_to_back_collectives_form_one_stretch(self):
        # 10 us gaps between 990 us collectives; counter refreshes 1000 us late.
        spans = [(1000 + i * 1000, 1990 + i * 1000) for i in range(100)]
        s = fr.Series([0, 50_000, 100_000, 150_000], [0.0, 10.0, 600.0, 700.0])
        events = fr.bound(s, spans, 1000, False)
        self.assertEqual(events.lb, 590.0)                  # [49000, 100000] lies in the stretch
        dur = fr.bound(s, spans, 1000, True)
        self.assertAlmostEqual(dur.lb, 590.0 - 51 * 10, delta=10)   # gives up the gap time
        far = [(1000 + i * 10_000, 1990 + i * 10_000) for i in range(10)]
        self.assertEqual(fr.bound(s, far, 1000, False).lb, 0.0)    # 9 ms gaps stay apart

    def test_unsampled_span_is_unbounded(self):
        s = fr.Series([0, 100], [0.0, 7.0])
        self.assertEqual(fr.bound(s, [(50, 300)], 0, False).ub, fr.INF)

    def test_bad_read_dropped_reset_carried(self):
        dip = {"ts_us": [0, 10, 20, 30], "v": [100, 0, 300, 300]}
        self.assertEqual(fr._series(dip, "v").vals, [100.0, 300.0, 300.0])
        reset = {"ts_us": [0, 10, 20, 30], "v": [100, 150, 5, 25]}
        s = fr._series(reset, "v")
        self.assertEqual((s.vals, s.resets), ([100.0, 150.0, 155.0, 175.0], 1))

    def test_hist_quantile(self):
        self.assertEqual(fr.hist_quantile([6, 2], 0.5, 30000), 30.0)
        self.assertEqual(fr.hist_quantile([6, 2], 0.99, 30000), 60.0)
        self.assertIsNone(fr.hist_quantile([], 0.5, 30000))
        fr.HIST_NS, fr.HIST_MAX = 30000, 5
        self.assertEqual(fr.fmt_quantile(150.0), ">=120")


class VerdictTest(unittest.TestCase):
    def test_fabric_congestion(self):
        r = one(rx_pause={BUSY: 0.10})
        g = r.rank(0)
        self.assertEqual(g.verdict, "FABRIC-CONGESTION")
        self.assertAlmostEqual(g.nics[0].rx_pause[0], 10.0, delta=0.2)
        self.assertIn("FABRIC: switch paused sending on 1 rank(s), no NIC pushed back",
                      next(iter(r.summary.values())))

    def test_host_backpressure(self):
        g = one(tx_pause={BUSY: 0.08}).rank(0)
        self.assertEqual(g.verdict, "HOST-BACKPRESSURE")

    def test_clean(self):
        r = one()
        self.assertTrue(r.rank(0).verdict.startswith("CLEAN"))
        self.assertIn("PFC counters zero on every priority", next(iter(r.summary.values())))

    def test_no_pfc_on_rccls_priority(self):
        # mlx5 with PFC enabled only on other priorities: RCCL's has no pause counters.
        r = one(no_pfc_prio=PRIO)
        self.assertEqual((r.lane.prio, r.lane.no_pfc), (PRIO, True))
        self.assertIn(f"no PFC on priority {PRIO}", next(iter(r.summary.values())))
        self.assertIn(f"PFC is not enabled on priority {PRIO}", r.cli("report"))

    # --- the four false reports found in review --------------------------------

    def test_pause_on_another_priority_is_not_rccls(self):
        r = one(rx_pause={BUSY: 0.10}, pause_prio=0)      # storage on prio 0, RCCL on 3
        self.assertEqual((r.lane.prio, r.lane.source), (PRIO, "traffic"))
        self.assertIn(0, r.lane.other_pause)
        self.assertTrue(r.rank(0).verdict.startswith("CLEAN"))

    def test_lane_unknown_without_priority_bytes_or_settings(self):
        # AINIC: no per-priority byte counters, DCB settings not readable. Pause is
        # seen, but on a priority that may not be RCCL's, so it is not judged.
        r = one(rx_pause={BUSY: 0.10}, prio_bytes=False, qos={"dcb": "unsupported", "traffic_class": 0, "sl": 0})
        self.assertEqual((r.lane.prio, r.lane.source), (None, "unknown"))
        self.assertNotIn("FABRIC", r.rank(0).verdict)
        self.assertIn("not judged", next(iter(r.summary.values())))
        out = r.cli("report")
        self.assertIn("UNKNOWN", out)
        self.assertIn("does not expose its DCB settings", out)
        self.assertEqual(r.lane.other_pause, {})          # cannot be called "not RCCL's"
        self.assertIn("Pause was seen on priorities 3; whether any is RCCL's is not known", out)
        self.assertTrue(r.rank(0).verdict.startswith("CLEAN (pause not judged;"), r.rank(0).verdict)

    def test_receiver_backpressure_is_not_blamed_on_fabric(self):
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: 0.10})),
                 (1, "hostB", 2, iterations(1, "hostB", 2), telemetry("hostB", 2, tx_pause={BUSY: 0.10}))])
        s = next(iter(r.summary.values()))
        self.assertTrue(s.startswith("FABRIC+HOST?: the switch paused 1 sender rank(s)"), s)
        self.assertIn("rank 1 (hostB) paused the switch", s)
        self.assertIn("cannot split the two", s)
        self.assertEqual(r.rank(0).verdict, "FABRIC-CONGESTION(echo?)")

    def pair(self, rx, tx):
        return Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: rx})),
                    (1, "hostB", 2, iterations(1, "hostB", 2), telemetry("hostB", 2, tx_pause={BUSY: tx}))])

    def test_pushback_is_reported_never_subtracted(self):
        # The two shares are of different ranks' time and backpressure is not a
        # causal bound, so no "X% is the fabric" is ever claimed.
        for rx, tx in ((0.08, 0.04), (0.20, 0.04), (0.054, 0.012)):
            s = next(iter(self.pair(rx, tx).summary.values()))
            self.assertNotIn("is the fabric", s)
            self.assertIn("ms)", s)                    # absolute times, comparable across ranks
        # Receiver paused about as long as the senders: it can explain all of it.
        self.assertTrue(next(iter(self.pair(0.08, 0.04).summary.values())).startswith("FABRIC+HOST?"))
        # Receiver paused far less: only part of it can be backpressure.
        s = next(iter(self.pair(0.20, 0.04).summary.values()))
        self.assertTrue(s.startswith("FABRIC?"), s)
        self.assertIn("part of the senders' pause may be backpressure", s)

    def test_equal_pause_time_on_different_busy_times(self):
        # Sender paused 80 ms of 400 ms (20%), receiver 80 ms of 2 s (4%): the
        # same stop, so it can explain all of the senders' pause.
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: 0.2})),
                 (1, "hostB", 2, iterations(1, "hostB", 2, dur=100_000),
                  telemetry("hostB", 2, t1=3_200_000, tx_pause={BUSY: 0.2}))])
        s = next(iter(r.summary.values()))
        self.assertTrue(s.startswith("FABRIC+HOST?"), s)
        self.assertIn("80 ms", s)

    def test_possible_pushback_is_hedged(self):
        # Receiver's pause falls between its short collectives: provable 0, possible > 0.
        gaps = {(BUSY[0] + i * 10_000 + 100, BUSY[0] + (i + 1) * 10_000): 0.5 for i in range(40)}
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: 0.10})),
                 (1, "hostB", 2, iterations(1, "hostB", 2, n=40, dur=100, gap=9_900),
                  telemetry("hostB", 2, tx_pause=gaps))])
        s = next(iter(r.summary.values()))
        self.assertTrue(s.startswith("FABRIC?"), s)
        self.assertIn("rank 1 (hostB) may have paused the switch", s)

    def test_no_pushback_keeps_fabric(self):
        s = next(iter(self.pair(0.10, 0.0).summary.values()))
        self.assertIn("no NIC pushed back", s)

    def test_pause_between_short_collectives_is_not_charged(self):
        # 100 us collectives 10 ms apart; every pause falls in the gaps.
        gaps = {(BUSY[0] + i * 10_000 + 100, BUSY[0] + (i + 1) * 10_000): 0.5 for i in range(40)}
        r = one(lines=iterations(0, "hostA", 1, n=40, dur=100, gap=9_900), rx_pause=gaps)
        g = r.rank(0)
        self.assertEqual(g.nics[0].rx_pause[0], 0.0)           # nothing provable
        self.assertTrue(g.verdict.startswith("INCONCLUSIVE (possible FABRIC-CONGESTION"), g.verdict)

    def test_no_samples_during_run_is_inconclusive(self):
        r = one(t0=100_000, t1=900_000)
        self.assertTrue(r.rank(0).verdict.startswith("INCONCLUSIVE"), r.rank(0).verdict)
        self.assertTrue(next(iter(r.summary.values())).startswith("INCONCLUSIVE"))

    def test_no_telemetry_is_inconclusive(self):
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), None)])
        self.assertTrue(r.rank(0).verdict.startswith("INCONCLUSIVE"))

    # --- the other gaps ---------------------------------------------------------

    def test_link_shared_without_pause(self):
        # 400 Gb/s = 50,000 bytes/us; someone else sends 38,000, RCCL 10,000: 96% busy.
        g = one(other_bpus=38_000).rank(0)
        self.assertEqual(g.verdict, "LINK-SHARED")
        self.assertGreater(g.nics[0].link_pct, 90)

    def test_link_busy_with_rccl_only_is_fine(self):
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, link_gbps=90))])
        self.assertTrue(r.rank(0).verdict.startswith("CLEAN"))
        self.assertGreater(r.rank(0).nics[0].link_pct, 80)

    def test_loss_without_pause(self):
        r = one(oos={BUSY: 0.0001})
        self.assertEqual(r.rank(0).verdict, "OUT-OF-SEQ")
        self.assertTrue(next(iter(r.summary.values())).startswith("DROPS: FABRIC_LIKELY"))

    def test_receiver_not_ready_names_the_right_host(self):
        # Out of buffers counts on the receiver: this host.
        r = one(oob={BUSY: 0.0001})
        self.assertEqual(r.rank(0).verdict, "RX-NO-BUFFER")
        self.assertIn("receive buffers ran out at rank 0 (hostA nic0", r.drop().endpoints[0])
        self.assertIn("that host's own receiver", r.drop().endpoints[0])
        # RNR NAKs count on the sender: the not-ready host is one of its peers.
        r = one(rnr={BUSY: 0.0001})
        self.assertEqual(r.rank(0).verdict, "PEER-RNR")
        self.assertIn("the not-ready peer is not identified", r.drop().endpoints[1])

    def test_ecn_needs_a_share_of_packets(self):
        # RCCL alone is ~2.4 packets/us; 0.1 marks/us is ~4%, 1 mark is ~0%.
        self.assertEqual(one(ecn={BUSY: 0.1}).rank(0).verdict, "CONGESTION-MARKS")
        self.assertTrue(one(ecn={(BUSY[0], BUSY[0] + 1): 1.0}).rank(0).verdict.startswith("CLEAN"))

    def test_no_ecn_marks_on_slow_port_counters_is_clean(self):
        # Thor2: port counters step once a second, so a window can see 0 packets.
        # Zero marks are 0% whatever the denominator, not "possible".
        r = one(refresh_us=500_000, t1=2_200_000)
        g, dev = r.rank(0), r.tels[("hostA", 1)].devices[0]
        pkts = fr.bound(dev.series.get("nic_rx_packets"), g.spans, fr.refresh_us(dev, 10_000), False)
        self.assertEqual(pkts.lb, 0.0)              # the case: packets moved, but not provably inside
        self.assertGreater(pkts.ub, 0.0)
        self.assertEqual(g.nics[0].ecn_pct, (0.0, 0.0))
        self.assertNotIn("CONGESTION-MARKS", g.verdict)

    def test_cnps_are_a_rate_not_a_share(self):
        # 300 CNPs/s is about 0.01% of packets here, yet a flow slowed all the time.
        g = one(cnp={BUSY: 300e-6}).rank(0)
        self.assertEqual(g.verdict, "CONGESTION-MARKS")
        self.assertAlmostEqual(g.nics[0].cnp_ps[0], 300, delta=30)
        self.assertTrue(one(cnp={BUSY: 20e-6}).rank(0).verdict.startswith("CLEAN"))

    def test_frames_only_nic_bounds_pause_from_above(self):
        # 400 Gb/s: one frame pauses at most 65535 * 512 bit times = 83.9 us.
        few = one(rx_frames={BUSY: 500e-6}, time_counters=False, roce="bnxt_re0").rank(0)
        self.assertTrue(few.nics[0].frames_only)
        self.assertEqual(few.nics[0].rx_pause[0], 0.0)
        self.assertAlmostEqual(few.nics[0].rx_pause[1], 4.2, delta=0.2)
        self.assertTrue(few.verdict.startswith("CLEAN"))          # at most 4.2%: ruled out
        self.assertEqual(fr._fmt_bound(few.nics[0].rx_pause), "0-4.2%")
        self.assertEqual(fr._fmt_bound((0.0, 0.4)), "0-0.4%")     # never a midpoint
        many = one(rx_frames={BUSY: 5000e-6}, time_counters=False, roce="bnxt_re0").rank(0)
        self.assertAlmostEqual(many.nics[0].rx_pause[1], 42, delta=2)
        self.assertEqual(many.verdict, "INCONCLUSIVE (possible FABRIC-CONGESTION)")  # never proven

    def test_counted_frames_prove_pushback(self):
        # Receiver counts only frames: its share is a bound, but a frame proves it paused.
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: 0.08})),
                 (1, "hostB", 2, iterations(1, "hostB", 2),
                  telemetry("hostB", 2, tx_frames={BUSY: 500e-6}, time_counters=False))])
        s = next(iter(r.summary.values()))
        self.assertIn("rank 1 (hostB) paused the switch", s)
        self.assertNotIn("may have paused", s)

    def test_mixed_nics_keep_their_own_unit(self):
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: 0.1})),
                 (1, "hostB", 2, iterations(1, "hostB", 2),
                  telemetry("hostB", 2, rx_frames={BUSY: 5000e-6}, time_counters=False))])
        self.assertFalse(r.rank(0).nics[0].frames_only)
        self.assertEqual(r.rank(0).verdict, "FABRIC-CONGESTION")
        self.assertTrue(r.rank(1).nics[0].frames_only)
        self.assertIn("possible FABRIC-CONGESTION", r.rank(1).verdict)

    # --- RCCL's priority from the NIC's DCB settings ------------------------

    @staticmethod
    def _qos(app, pfc_on, tc=0):
        return {"dcb": "ok", "traffic_class": tc, "sl": 0, "traffic_class_mixed": False,
                "pfc_enabled": [1 if p in pfc_on else 0 for p in range(8)], "app": app}

    def test_priority_from_roce_port_entry(self):
        # Thor2 on ruby: dgram-port-prio 4791:3, PFC on 3. No byte counters needed.
        q = self._qos([{"selector": "dgram", "protocol": 4791, "priority": 3}], {3})
        r = one(rx_pause={BUSY: 0.10}, prio_bytes=False, qos=q, hw_type="thor2")
        self.assertEqual((r.lane.prio, r.lane.source, r.lane.no_pfc), (3, "config", False))
        self.assertEqual(r.rank(0).verdict, "FABRIC-CONGESTION")
        self.assertIn("NIC maps RoCE (UDP 4791)", r.cli("report"))

    def test_priority_from_dscp_map_with_pfc_off(self):
        # mlx5 on OCI: DSCP 0 -> priority 1, PFC only on 0 and 5.
        q = self._qos([{"selector": "dscp", "protocol": 0, "priority": 1},
                       {"selector": "dscp", "protocol": 26, "priority": 5}], {0, 5})
        r = one(prio=1, qos=q)
        self.assertEqual((r.lane.prio, r.lane.source, r.lane.no_pfc), (1, "config", True))
        out = r.cli("report")
        self.assertIn("PFC enabled on priorities [0, 5] (NIC settings)", out)
        self.assertIn("PFC is not enabled on priority 1", out)

    def test_roce_port_entry_alone_is_not_trusted_on_mlx5(self):
        q = self._qos([{"selector": "dgram", "protocol": 4791, "priority": 3}], {3})
        r = one(rx_pause={BUSY: 0.10}, prio_bytes=False, qos=q, hw_type="mlx5")
        self.assertEqual(r.lane.prio, None)
        self.assertIn("SL 0", r.lane.why)

    def test_settings_that_carried_nothing_lose_to_bytes(self):
        q = self._qos([{"selector": "dscp", "protocol": 0, "priority": 3}], {3})
        r = one(prio=1, qos=q)                     # every byte on 1, none on 3
        self.assertEqual((r.lane.prio, r.lane.source), (1, "traffic"))
        self.assertTrue(any("carried almost no bytes" in w for w in r.lane.warnings))

    def test_other_traffic_does_not_overrule_settings(self):
        # RCCL on 3 per settings, storage heavier on 0: settings stand, with a note.
        q = self._qos([{"selector": "dscp", "protocol": 0, "priority": 3}], {3})
        tel = telemetry("hostA", 1, qos=q)
        for smp in tel["hw_samples"]:
            b = smp["prio_tx_bytes"][3]
            smp["prio_tx_bytes"][0] = smp["prio_rx_bytes"][0] = 3 * b
        r = one(tel=tel)
        self.assertEqual((r.lane.prio, r.lane.source), (3, "config"))
        self.assertTrue(any("other traffic shares these NICs" in w for w in r.lane.warnings))

    def test_each_nic_uses_its_own_priority(self):
        # hostA's NIC puts RCCL on 3, hostB's on 1; each pauses on its own.
        qa = self._qos([{"selector": "dscp", "protocol": 0, "priority": 3}], {3})
        qb = self._qos([{"selector": "dscp", "protocol": 0, "priority": 1}], {1})
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1),
                  telemetry("hostA", 1, prio=3, rx_pause={BUSY: 0.10}, qos=qa)),
                 (1, "hostB", 2, iterations(1, "hostB", 2),
                  telemetry("hostB", 2, prio=1, rx_pause={BUSY: 0.10}, qos=qb))])
        self.assertEqual((r.lane.prio, r.lane.source), (None, "per-nic"))
        self.assertEqual(r.lane.prio_for("hostA", 1, 0), 3)
        self.assertEqual(r.lane.prio_for("hostB", 2, 0), 1)
        self.assertEqual(r.rank(0).verdict, "FABRIC-CONGESTION")
        self.assertEqual(r.rank(1).verdict, "FABRIC-CONGESTION")
        self.assertIn("PFC priority differs by NIC", r.cli("report"))

    def test_switching_priority_recomputes_job_fields(self):
        # NIC A: settings say 3, but heavy storage traffic on 0. NIC B: no DCB,
        # all traffic on 3. Every NIC says 3; the job byte majority says 0.
        qa = self._qos([{"selector": "dscp", "protocol": 0, "priority": 3}], {3})
        ta = telemetry("hostA", 1, prio=3, qos=qa)
        for smp in ta["hw_samples"]:
            b = smp["prio_tx_bytes"][3]
            smp["prio_tx_bytes"][0] = smp["prio_rx_bytes"][0] = 5 * b
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), ta),
                 (1, "hostB", 2, iterations(1, "hostB", 2),
                  telemetry("hostB", 2, prio=3, rx_pause={BUSY: 0.10}, qos={"dcb": "unsupported",
                                                                         "traffic_class": 0, "sl": 0}))])
        self.assertEqual(r.lane.prio, 3)
        self.assertNotIn(3, r.lane.other_pause)        # RCCL's own pause is not "other"
        self.assertLess(r.lane.share, 0.5)             # priority 3's share, not priority 0's

    def test_dscp_missing_from_map_is_not_guessed(self):
        q = self._qos([{"selector": "dscp", "protocol": 26, "priority": 3}], {3})
        r = one(rx_pause={BUSY: 0.10}, prio_bytes=False, qos=q)
        self.assertEqual(r.lane.prio, None)
        self.assertIn("DSCP 0 is not in the NIC's DSCP map", r.lane.why)

    def test_each_counter_family_has_its_own_lag(self):
        # Port bytes refresh every sample; the pause counter only every 150 ms.
        r = one(rx_pause={(900_000, 1_600_000): 0.10}, pause_refresh_us=150_000)
        dev = r.tels[("hostA", 1)].devices[0]
        self.assertEqual(fr.refresh_us(dev, 10_000, "nic_tx_bytes"), 10_000)
        self.assertEqual(fr.refresh_us(dev, 10_000, f"pfc_rx_pause_us[{PRIO}]"), 150_000)
        # A counter that never moved borrows the byte counters' lag, not its own.
        self.assertEqual(fr.refresh_us(dev, 10_000, "ecn_marked_pkts"), 10_000)

    def test_pooled_rx_bytes_are_not_read_as_other_traffic(self):
        # Second NIC of a merged connection: the port receives at full rate, but
        # RCCL charged those rx bytes to the first NIC. Without the flag that
        # reads as a link full of someone else's traffic.
        tel = telemetry("hostA", 1, other_bpus=40_000.0)
        for smp in tel["hw_samples"]:
            smp["rx_bytes"] = 0
            smp["nic_tx_bytes"] = int(RCCL_BPUS * smp["ts_us"])     # tx is RCCL's own, per NIC
        self.assertIn("LINK-SHARED", one(tel=tel).rank(0).verdict)
        tel["devices"][0]["rx_bytes_merged"] = True
        self.assertNotIn("LINK-SHARED", one(tel=tel).rank(0).verdict)

    def test_receive_that_opened_the_window_is_added_once(self):
        # Two NICs, 64 MiB received inside the window on each, and a 4096-byte
        # receive that opened the window before either NIC's delta started.
        lines = []
        for l in iterations(0, "hostA", 1):
            obj = json.loads(l)
            net = obj["coll_perf"]["net"]
            net["rx_bytes_before_window"] = 4096
            net["devices"].append(dict(net["devices"][0], device_id=1))
            lines.append(json.dumps(obj))
        g = one(lines=lines).rank(0)
        self.assertEqual(g.rx_per_coll_lb, 2 * 64 * MB + 4096)    # once per collective, not per NIC
        self.assertEqual(g.nics[0].sw_rx_per_coll, 64 * MB)       # per-NIC: within the window only

    def test_missing_counters_are_not_clean(self):
        # Only RCCL's software bytes: nothing about loss, marks, link or pause.
        tel = telemetry("hostA", 1)
        for smp in tel["hw_samples"]:
            for k in list(smp):
                if k not in ("ts_us", "device_id", "roce_device", "tx_bytes", "rx_bytes"):
                    del smp[k]
        g = one(tel=tel).rank(0)
        self.assertTrue(g.verdict.startswith("INCONCLUSIVE"), g.verdict)
        for what in ("delivery counters", "ECN/CNP counters", "link use", "pause counters"):
            self.assertIn(what, g.verdict)

    def test_incomplete_accl_run_is_not_clean(self):
        lines = iterations(0, "hostA", 1) + [json.dumps(
            {"summary": {"complete": False, "dropped_collectives": 200, "dropped_proxy_ops": 200}})]
        r = one(lines=lines)
        s = next(iter(r.summary.values()))
        self.assertTrue(s.startswith("INCONCLUSIVE: accl lost data"), s)
        self.assertIn("dropped_proxy_ops=200", s)

    def test_accl_file_without_summary_is_not_clean(self):
        d = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, d, True)
        with open(os.path.join(d, "accl_profiler_rank0_hostA_pid1.jsonl"), "w") as f:
            f.write("\n".join(iterations(0, "hostA", 1)) + "\n")          # process never finalized
        recs, _ = fr.load_accl(d, 0)
        self.assertIn("no end-of-run summary", recs[0].accl_incomplete)

    def test_record_without_exec_start_is_skipped(self):
        # One record with no kernel or network activity: exec_start_us 0.
        bad = json.loads(iterations(0, "hostA", 1, n=1)[0])
        bad["coll_perf"]["host_ts"]["exec_start_us"] = 0
        r = one(lines=iterations(0, "hostA", 1) + [json.dumps(bad)])
        recs, skipped = fr.load_accl(r.dirname, 0)
        self.assertEqual(skipped, 1)
        self.assertLess(r.rank(0).busy_us, 1_000_000)      # not stretched back to time 0

    def test_repeated_bad_reads_are_not_a_reset(self):
        # Thor2 sysfs has read 0 more than once in a row between good values.
        ser = fr._series({"ts_us": [1, 2, 3, 4, 5, 6], "x": [100, 200, 0, 0, 400, 500]}, "x")
        self.assertEqual((ser.total(), ser.resets), (400, 0))
        ser = fr._series({"ts_us": [1, 2, 3, 4], "x": [100, 200, 5, 30]}, "x")   # a real reset
        self.assertEqual((ser.total(), ser.resets), (130, 1))

    def test_slow_refresh_widens_windows(self):
        tel = telemetry("hostA", 1, refresh_us=100_000)
        r = one(tel=tel)
        dev = r.tels[("hostA", 1)].devices[0]
        self.assertGreaterEqual(fr.refresh_us(dev, 10_000), 90_000)

    def test_sequential_communicators_are_not_merged(self):
        # Same collective and size, two communicators one after the other; only
        # the first ran while the switch paused us.
        lines = iterations(0, "hostA", 1, comm="0x1") + \
            iterations(0, "hostA", 1, start=1_400_000, n=9, comm="0x2", first_sn=100)
        r = one(lines=lines, rx_pause={BUSY: 0.10})
        verdicts = sorted(r.summary.values())
        self.assertEqual(len(verdicts), 2, r.summary)
        self.assertTrue(verdicts[0].startswith("CLEAN"), verdicts)
        self.assertTrue(verdicts[1].startswith("FABRIC"), verdicts)
        self.assertIn("comm 0x1", r.cli("report"))

    def test_overlapping_communicators_warned(self):
        lines = iterations(0, "hostA", 1, comm="0x1") + iterations(0, "hostA", 1, comm="0x2", first_sn=100)
        r = one(lines=lines)
        recs, _ = fr.load_accl(r.dirname, 0)
        self.assertTrue(any("ran at the same time" in w for w in fr._comm_overlap_warnings(recs)))

    def test_cli_report_and_trace(self):
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: 0.1})),
                 (1, "hostB", 2, iterations(1, "hostB", 2), telemetry("hostB", 2))])
        out = r.cli("report")
        self.assertIn("AllReduce 64M: FABRIC: switch paused sending on 1 rank(s)", out)
        trace = os.path.join(r.dirname, "t.json")
        r.cli("trace", "-o", trace)
        with open(trace) as f:
            ev = json.load(f)["traceEvents"]
        colls = [e for e in ev if e["ph"] == "X" and e["name"] != "network"]
        nets = [e for e in ev if e["ph"] == "X" and e["name"] == "network"]
        self.assertEqual((len(colls), len(nets)), (40, 40))
        assert_nested(self, ev)
        pause = [e for e in ev if e["ph"] == "C" and "PFC pause %" in e["name"]]
        self.assertAlmostEqual(max(e["args"]["rx: switch paused us"] for e in pause), 10.0, delta=0.2)


class DropDiagnosisTest(unittest.TestCase):
    """Delivery symptoms against endpoint evidence: buckets and confidence."""
    EV = 0.0001          # 40 events over BUSY's 400 ms

    @staticmethod
    def two(a=None, b=None, la=None, lb=None):
        """Ranks 0 (hostA) and 1 (hostB) of one 2-rank communicator."""
        return Run([(0, "hostA", 1, la or iterations(0, "hostA", 1), telemetry("hostA", 1, **(a or {}))),
                    (1, "hostB", 2, lb or iterations(1, "hostB", 2), telemetry("hostB", 2, **(b or {})))])

    def test_timeout_only_with_unknown_endpoints_is_inconclusive(self):
        d = one(ack_to={BUSY: self.EV}, drop=("out_of_buffer", "rnr_retry_err")).drop()
        self.assertEqual(d.bucket, "INCONCLUSIVE")
        self.assertIn("only ACK timeouts", " ".join(d.reasons))
        self.assertIn("in window 40 at rank 0", d.signals[2])            # still shown
        self.assertIn("unknown at rank 0 (hostA nic0): no counter", d.endpoints[0])
        self.assertTrue(any("lost ACK on the return path" in c for c in d.competing))

    def test_timeout_only_with_healthy_endpoints_is_low(self):
        d = one(ack_to={BUSY: self.EV}).drop()
        self.assertEqual((d.bucket, d.confidence), ("FABRIC_LIKELY", "low"))

    def test_symptoms_with_healthy_endpoints(self):
        d = self.two(a={"oos": {BUSY: self.EV}}, b={"seq": {BUSY: self.EV}}).drop()
        self.assertEqual((d.bucket, d.confidence), ("FABRIC_LIKELY", "medium"))
        self.assertIn("healthy endpoints", d.reasons[0])
        for e in d.endpoints[:4]:
            self.assertNotIn("unknown", e)
        # ECN marks on the path corroborate the fabric.
        d = self.two(a={"oos": {BUSY: self.EV}, "ecn": {BUSY: 0.1}}).drop()
        self.assertEqual((d.bucket, d.confidence), ("FABRIC_LIKELY", "high"))

    def test_local_receive_buffer_exhaustion(self):
        r = self.two(b={"oob": {BUSY: self.EV}})
        d = r.drop()
        self.assertEqual((d.bucket, d.confidence), ("HOST_LIKELY", "medium"))
        self.assertIn("receive buffers ran out at rank 1 (hostB nic0", d.endpoints[0])
        self.assertIn("fine at the other 1 NIC(s)", d.endpoints[0])
        self.assertEqual(r.rank(1).verdict, "RX-NO-BUFFER")
        self.assertTrue(r.rank(0).verdict.startswith("CLEAN"))

    def test_sender_side_rnr_is_not_the_observers_fault(self):
        r = self.two(a={"rnr": {BUSY: self.EV}})
        d = r.drop()
        self.assertEqual(d.bucket, "HOST_LIKELY")
        self.assertIn("RNR NAKs seen at rank 0 (hostA nic0", d.endpoints[1])
        self.assertIn("seen by the sender; the not-ready peer is not identified", d.endpoints[1])
        # The observer's own receiver is fine.
        self.assertEqual(d.endpoints[0], "receive buffers: no receiver ran out of buffers")
        # Rank 0 only observed it: its own row and the summary do not blame it.
        self.assertEqual(r.rank(0).verdict, "PEER-RNR")
        self.assertNotIn("receive buffers ran out", " ".join(d.endpoints + d.reasons))
        s = next(iter(r.summary.values()))
        self.assertIn("from PEER-RNR on rank 0 (hostA)", s)        # named as what it saw
        self.assertNotIn("RX-NO-BUFFER", s)

    def test_endpoint_pressure_and_symptoms_together(self):
        r = self.two(a={"oos": {BUSY: self.EV}}, b={"oob": {BUSY: self.EV}})
        d = r.drop()
        self.assertEqual((d.bucket, d.confidence), ("HOST_LIKELY", "low"))
        self.assertTrue(any("loss in the fabric" in c for c in d.competing))
        self.assertIn("OUT-OF-SEQ", d.signals[0])                       # neither side hidden
        self.assertIn("receive buffers ran out", d.endpoints[0])
        # With fabric congestion as well: both, explicitly.
        d = self.two(a={"oos": {BUSY: self.EV}, "ecn": {BUSY: 0.1}}, b={"oob": {BUSY: self.EV}}).drop()
        self.assertEqual((d.bucket, d.confidence), ("MIXED", "medium"))

    def test_only_possibly_overlapping_events(self):
        # 100 us collectives 10 ms apart; every drop falls in the gaps.
        gaps = {(BUSY[0] + i * 10_000 + 100, BUSY[0] + (i + 1) * 10_000): 0.001 for i in range(40)}
        d = one(lines=iterations(0, "hostA", 1, n=40, dur=100, gap=9_900, n_ranks=1), oos=gaps).drop()
        self.assertEqual(d.bucket, "INCONCLUSIVE")
        self.assertIn("possibly in window 0-", d.signals[0])
        self.assertIn("only possibly fall inside", d.reasons[0])

    def test_missing_counter_is_not_covered_by_another(self):
        # No out-of-sequence column; SEQ-NAK and ACK-TIMEOUT read 0. Not NONE.
        d = one(drop=("oos_drop_count",)).drop()
        self.assertEqual(d.bucket, "INCONCLUSIVE")
        self.assertIn("OUT-OF-SEQ: rank 0 (hostA nic0): not measured", d.unknown)
        self.assertIn("none where measured", d.signals[0])

    def test_stale_samples_resets_and_incomplete_accl_lower_confidence(self):
        base = one(oos={BUSY: self.EV}).drop()
        self.assertEqual((base.bucket, base.confidence), ("FABRIC_LIKELY", "medium"))
        # A 100 ms hole in the samples inside the collectives.
        tel = telemetry("hostA", 1, oos={BUSY: self.EV})
        tel["hw_samples"] = [x for x in tel["hw_samples"] if not 1_100_000 < x["ts_us"] < 1_200_000]
        d = one(tel=tel).drop()
        self.assertEqual(d.confidence, "low")
        self.assertIn("sampler stalls cover", d.quality[0])
        # The accl file lost data, and a counter reset inside the collectives:
        # no confidence left.
        tel = telemetry("hostA", 1, oos={BUSY: self.EV})
        for x in tel["hw_samples"]:
            if x["ts_us"] >= 1_200_000:
                x["oos_drop_count"] -= 30
        lines = iterations(0, "hostA", 1, n_ranks=1) + [json.dumps(
            {"summary": {"complete": False, "dropped_proxy_ops": 3}})]
        d = one(tel=tel, lines=lines).drop()
        self.assertEqual((d.bucket, d.leaning), ("INCONCLUSIVE", "FABRIC_LIKELY"))
        self.assertTrue(any("accl lost data" in q for q in d.quality))
        self.assertTrue(any("reset" in q for q in d.quality))
        self.assertIn("in window", d.signals[0])                         # signals kept

    def test_rank_without_telemetry_leaves_endpoints_unknown(self):
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, oos={BUSY: self.EV})),
                 (1, "hostB", 2, iterations(1, "hostB", 2), None)])
        d = r.drop()
        self.assertEqual((d.bucket, d.confidence), ("FABRIC_LIKELY", "low"))
        self.assertIn("unknown at rank 1 (hostB): no telemetry", d.endpoints[0])

    def test_shared_nic_and_overlapping_communicators(self):
        d = one(oos={BUSY: self.EV}, other_bpus=20_000).drop()      # RCCL a third of the port
        self.assertEqual(d.confidence, "low")
        self.assertIn("was this job's RCCL", d.quality[0])
        lines = iterations(0, "hostA", 1, n_ranks=1) + iterations(0, "hostA", 1, comm="0x2", first_sn=100,
                                                                   n_ranks=1)
        r = one(lines=lines, oos={BUSY: self.EV})
        for d in r.drops.values():
            self.assertTrue(any("other collectives ran at the same time" in q for q in d.quality))
            self.assertEqual(d.confidence, "low")

    def test_clean_adequately_observed_run(self):
        r = self.two()
        d = r.drop()
        self.assertEqual((d.bucket, d.confidence), ("NONE", "high"))
        s = next(iter(r.summary.values()))
        self.assertTrue(s.startswith("CLEAN"), s)
        self.assertNotIn("also", s)

    def test_recovery_signals_are_not_added_up(self):
        r = one(oos={BUSY: self.EV}, seq={BUSY: self.EV}, ack_to={BUSY: self.EV})
        d = r.drop()
        self.assertEqual([x.split(",")[0] for x in d.signals[:3]], ["OUT-OF-SEQ", "SEQ-NAK", "ACK-TIMEOUT"])
        for x in d.signals[:3]:
            self.assertIn("in window 40 at", x)
        self.assertIn("not added up into a count of lost packets", d.signals[3])
        self.assertEqual(r.rank(0).verdict, "OUT-OF-SEQ+SEQ-NAK+ACK-TIMEOUT")
        out = r.cli("report")
        self.assertIn("40/40/40", out)
        self.assertFalse([x for x in d.signals if "120" in x])

    def test_no_finding_hidden_by_the_headline(self):
        # A host out of buffers leads; the rx pause is still named.
        s = next(iter(one(rx_pause={BUSY: 0.10}, oob={BUSY: self.EV}).summary.values()))
        self.assertTrue(s.startswith("DROPS: HOST_LIKELY (medium)"), s)
        self.assertIn("also: FABRIC-CONGESTION on rank 0 (hostA)", s)
        # Fabric-side drops come after the pause, and are named after it.
        s = next(iter(one(rx_pause={BUSY: 0.10}, oos={BUSY: self.EV}).summary.values()))
        self.assertTrue(s.startswith("FABRIC: switch paused"), s)
        self.assertIn("OUT-OF-SEQ on rank 0 (hostA)", s)
        self.assertIn("drops FABRIC_LIKELY", s)
        # Possible findings are named too.
        s = next(iter(one(ecn={BUSY: 0.1}, tx_frames={BUSY: 0.0006}, time_counters=False).summary.values()))
        self.assertIn("possible HOST-BACKPRESSURE on rank 0 (hostA)", s)

    def test_pause_alone_is_not_a_drop_diagnosis(self):
        r = one(tx_pause={BUSY: 0.10})
        self.assertEqual(r.drop().bucket, "NONE")
        self.assertIn("pause is flow control, not drops", " ".join(r.drop().reasons))
        self.assertNotIn("drops", next(iter(r.summary.values())))

    def test_small_proven_pause_does_not_block_a_diagnosis(self):
        # 0.5% pause is proven but under the 5% threshold: fine, and said so.
        d = one(tx_pause={BUSY: 0.005}, oos={BUSY: self.EV}).drop()
        self.assertEqual((d.bucket, d.confidence), ("FABRIC_LIKELY", "medium"))
        self.assertIn("under the threshold at rank 0 (hostA nic0)", d.endpoints[2])
        self.assertEqual(one(tx_pause={BUSY: 0.005}).drop().bucket, "NONE")

    def test_frames_only_pause_is_never_proven_pressure(self):
        # 0-5% from frames: possible, so health is not established; never a host cause.
        d = one(tx_frames={BUSY: 0.0006}, time_counters=False, oos={BUSY: self.EV}).drop()
        self.assertEqual((d.bucket, d.confidence), ("FABRIC_LIKELY", "low"))
        self.assertIn("possibly receiver backpressure", d.endpoints[2])

    def test_slow_drop_counter_that_never_published_is_not_none(self):
        # Counters step once a second and samples stop 200 ms after the
        # collectives: drops in them may not have shown yet.
        tel = telemetry("hostA", 1, refresh_us=1_000_000)
        for x in tel["hw_samples"]:
            x["oos_drop_count"] = int(cum({BUSY: self.EV}, x["ts_us"] // 1_000_000 * 1_000_000))
        d = one(tel=tel).drop()
        self.assertNotEqual(d.bucket, "NONE")
        self.assertIn("had not refreshed yet", " ".join(d.unknown))

    def test_completion_errors(self):
        r = one(lines=iterations(0, "hostA", 1, n_ranks=1, cq_errors=1))
        self.assertEqual(r.drop().bucket, "INCONCLUSIVE")
        self.assertIn("only completion errors", r.drop().reasons[0])
        self.assertIn("CQ-ERROR", r.rank(0).verdict)
        self.assertFalse(next(iter(r.summary.values())).startswith("CLEAN"))
        # With another symptom, the symptom decides and the errors stay listed.
        d = one(lines=iterations(0, "hostA", 1, n_ranks=1, cq_errors=1), oos={BUSY: self.EV}).drop()
        self.assertEqual(d.bucket, "FABRIC_LIKELY")
        self.assertTrue(any("local faults" in c for c in d.competing))
        # They need no telemetry.
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1, n_ranks=1, cq_errors=2), None)])
        self.assertIn("40 at rank 0 (hostA)", r.drop().signals[3])

    def test_rank_without_traffic_is_not_blind(self):
        # Rank 1 has telemetry but drove no NIC (intra-node only).
        idle = telemetry("hostB", 2)
        for x in idle["hw_samples"]:
            x["tx_bytes"] = x["rx_bytes"] = 0
        lines = [json.dumps(dict(json.loads(l), coll_perf=dict(json.loads(l)["coll_perf"], net={})))
                 for l in iterations(1, "hostB", 2)]
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1)), (1, "hostB", 2, lines, idle)])
        self.assertEqual(r.drop().bucket, "NONE")

    def test_missing_ranks_shared_processes_and_lag(self):
        # 2-rank communicator, one rank's accl files missing.
        d = one(lines=iterations(0, "hostA", 1), oos={BUSY: self.EV}).drop()
        self.assertIn("1 of 2 ranks: no accl records", d.endpoints[0])
        self.assertEqual((d.confidence, d.quality), ("low", []))      # from unknown health, no quality step
        self.assertIn("endpoint health not established", d.reasons[0])
        # Two processes of the job drive the same NIC.
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, oos={BUSY: self.EV})),
                 (1, "hostA", 2, iterations(1, "hostA", 2), telemetry("hostA", 2))])
        self.assertTrue(any("2 processes of this job drive" in q for q in r.drop().quality))
        # Counters 150 ms late against 100 ms of collectives.
        d = one(lines=iterations(0, "hostA", 1, n=5, n_ranks=1), oos={BUSY: self.EV}, refresh_us=150_000).drop()
        self.assertTrue(any("refresh ~" in q for q in d.quality), d.quality)

    def test_rule_branches(self):
        # Rx pause next to a receiver that pushed back is not fabric corroboration,
        # and the pushback is not a cause of drops: PFC kept that receiver lossless.
        d = self.two(a={"oos": {BUSY: self.EV}, "rx_pause": {BUSY: 0.10}}, b={"tx_pause": {BUSY: 0.10}}).drop()
        self.assertEqual((d.bucket, d.confidence), ("FABRIC_LIKELY", "low"))
        self.assertTrue(any("PFC kept it from dropping" in c for c in d.competing))
        # Nothing known about receivers at all: no fabric verdict.
        d = one(oos={BUSY: self.EV}, drop=("out_of_buffer", "rnr_retry_err")).drop()
        self.assertEqual((d.bucket, d.leaning), ("INCONCLUSIVE", "FABRIC_LIKELY"))
        # Receiver not ready, with a symptom only possibly in the window: lower.
        gaps = {(BUSY[0] + i * 10_000 + 100, BUSY[0] + (i + 1) * 10_000): 0.001 for i in range(40)}
        d = self.two(a={"oos": gaps}, b={"oob": {BUSY: self.EV}},
                     la=iterations(0, "hostA", 1, n=40, dur=100, gap=9_900)).drop()
        self.assertEqual((d.bucket, d.confidence), ("HOST_LIKELY", "low"))
        # Attribution issues do not lower NONE: a count that stayed 0 is 0.
        d = one(other_bpus=20_000).drop()
        self.assertEqual((d.bucket, d.confidence), ("NONE", "high"))
        # ACK timeouts only, with healthy endpoints and ECN marks: medium.
        d = one(ack_to={BUSY: self.EV}, ecn={BUSY: 0.1}).drop()
        self.assertEqual((d.bucket, d.confidence), ("FABRIC_LIKELY", "medium"))
        # Nothing seen, but backpressure unknown (PFC off on RCCL's priority): medium.
        d = one(no_pfc_prio=PRIO).drop()
        self.assertEqual((d.bucket, d.confidence), ("NONE", "medium"))
        # NIC progress needs bytes both ways: a receive port that stood still
        # during the collectives (moving before and after) is unknown.
        tel = telemetry("hostA", 1, oos={BUSY: self.EV})
        for x in tel["hw_samples"]:
            x["nic_rx_bytes"] = int(4e4 * (min(x["ts_us"], BUSY[0]) + max(0, x["ts_us"] - BUSY[1])))
        d = one(tel=tel).drop()
        self.assertEqual((d.bucket, d.confidence), ("FABRIC_LIKELY", "low"))
        self.assertIn("NIC progress: unknown", d.endpoints[3])
        # Out-of-buffer only possibly inside the collectives: not a host verdict.
        gaps = {(BUSY[0] + i * 10_000 + 100, BUSY[0] + (i + 1) * 10_000): 0.001 for i in range(40)}
        d = one(lines=iterations(0, "hostA", 1, n=40, dur=100, gap=9_900, n_ranks=1), oob=gaps).drop()
        self.assertEqual(d.bucket, "INCONCLUSIVE")
        # A completion error does not weaken a receiver that was not ready.
        d = self.two(la=iterations(0, "hostA", 1, cq_errors=1), b={"oob": {BUSY: self.EV}}).drop()
        self.assertEqual((d.bucket, d.confidence), ("HOST_LIKELY", "medium"))

    def test_nics_named_by_accl_but_not_sampled_are_blind(self):
        tel = telemetry("hostB", 2)
        tel["hw_samples"] = []
        r = self.two()
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1)),
                 (1, "hostB", 2, iterations(1, "hostB", 2), tel)])
        self.assertNotEqual(r.drop().bucket, "NONE")
        self.assertIn("rank 1 (hostB)", " ".join(r.drop().unknown))

    def test_ack_column_from_retransmits_is_labelled(self):
        tel = telemetry("hostA", 1, ack_to={BUSY: self.EV})
        tel["devices"][0]["hw_counters"] = {"local_ack_timeout_err": -1, "to_retransmits": 0}
        self.assertIn("counts retransmits after a timeout", one(tel=tel).drop().signals[2])
        self.assertNotIn("retransmits", one(ack_to={BUSY: self.EV}).drop().signals[2])

    def test_missing_delivery_column_is_not_clean(self):
        r = one(drop=("local_ack_timeout_err",))
        self.assertIn("not measured: delivery counters (ACK-TIMEOUT)", r.rank(0).verdict)
        self.assertFalse(next(iter(r.summary.values())).startswith("CLEAN"))
        r = one(drop=("out_of_buffer",))
        self.assertIn("receiver-not-ready counters (RX-NO-BUFFER)", r.rank(0).verdict)

    def test_report_shows_every_part_and_baseline(self):
        base = one(lines=[l.replace('"coll_busbw_gbs": 2.0', '"coll_busbw_gbs": 4.0')
                          for l in iterations(0, "hostA", 1, n_ranks=1)])
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1, n_ranks=1), telemetry("hostA", 1, ack_to={BUSY: self.EV},
                                                                               drop=("out_of_buffer",)))])
        out = r.cli("report", "--baseline", base.dirname)
        diag = out[out.index("Drop diagnosis"):out.index("Findings:")]
        for part in ("signals", "endpoints", "not measured  RX-NO-BUFFER: rank 0 (hostA nic0): no counter",
                     "why", "could also be", "slowness"):
            self.assertIn(part, diag)
        self.assertIn("AllReduce 64M: INCONCLUSIVE", diag)
        self.assertIn("busBW 2.0 GB/s against 4.0 GB/s in the baseline run (-50%)", diag)
        self.assertIn("does not by itself show the collective was slow", diag)
        # A baseline with another rank count is not a match.
        other = one(lines=iterations(0, "hostA", 1, n_ranks=16))
        self.assertIn("no matched baseline", r.cli("report", "--baseline", other.dirname))


if __name__ == "__main__":
    unittest.main()
