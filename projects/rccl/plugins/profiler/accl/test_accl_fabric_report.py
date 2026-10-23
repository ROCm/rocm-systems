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


def accl_line(rank, host, pid, coll, size, sn, start, end, comm="0x1"):
    cp = {"coll": coll, "coll_sn": sn, "coll_msg_size_bytes": size, "coll_algo": "RING",
          "coll_proto": "SIMPLE", "coll_n_channels": 4, "coll_exec_time_us": end - start,
          "coll_algobw_gbs": 1.0, "coll_busbw_gbs": 2.0, "coll_timing_source": "gpu_globaltimer",
          "decomposition": {"enqueue_to_kernel_us": 1, "gpu_kernel_avg_us": end - start,
                            "proxy_network_us": 10, "proxy_peer_wait_us": 10, "n_proxy_ops": 2},
          "host_ts": {"clock": "CLOCK_MONOTONIC", "enqueue_us": start - 5,
                      "exec_start_us": start, "exec_end_us": end},
          "net": {"source": "rccl_telemetry_sw", "window_start_us": start, "window_end_us": end,
                  "devices": [{"device_id": 0, "tx_bytes": size, "rx_bytes": size,
                               "completion_hist": [6, 2]}]}}
    return json.dumps({"header": {"rank": rank, "n_ranks": 2, "host": host, "pid": pid,
                                  "comm_hash": comm}, "coll_perf": cp})


def cum(rate, t):
    """Integral up to t of a piecewise-constant rate given as {(a, b): per_us}."""
    return sum(max(0, min(t, b) - a) * r for (a, b), r in (rate or {}).items())


def telemetry(host, pid, roce="nic0", t0=900_000, t1=1_600_000, step=10_000, prio=PRIO,
              rx_pause=None, tx_pause=None, rx_frames=None, ecn=None, oos=None, rnr=None,
              other_bpus=0.0, link_gbps=400, prio_bytes=True, time_counters=True,
              refresh_us=None, pause_prio=None, no_pfc_prio=None, cnp=None):
    """One telemetry file. Rates are {(a, b): per-us}; pause rates are a fraction
    of time. RCCL sends RCCL_BPUS; `other_bpus` is someone else's traffic on the
    same port. `refresh_us` makes the NIC's ethtool counters step that often."""
    pp = prio if pause_prio is None else pause_prio
    samples = []
    t = t0
    while t <= t1:
        tn = t if refresh_us is None else (t // refresh_us) * refresh_us   # NIC-side value time
        z = [0] * 8
        rxp, txp, rxf, pb = list(z), list(z), list(z), list(z)
        rxp[pp] = int(cum(rx_pause, tn))
        txp[pp] = int(cum(tx_pause, tn))
        rxf[pp] = int(cum(rx_frames, tn))
        nic = (RCCL_BPUS + other_bpus) * tn
        pb[prio] = int(nic)
        if no_pfc_prio is not None:       # the NIC has no PFC counters for this priority
            for a in (rxp, txp, rxf):
                a[no_pfc_prio] = -1
        s = {"ts_us": t, "device_id": 0, "roce_device": roce,
             "tx_bytes": RCCL_BPUS * t, "rx_bytes": RCCL_BPUS * t,
             "nic_tx_bytes": int(nic), "nic_rx_bytes": int(nic),
             "nic_tx_packets": int(nic / 4096), "nic_rx_packets": int(nic / 4096),
             "ecn_marked_pkts": int(cum(ecn, t)), "cnp_sent": int(cum(cnp, t)), "cnp_rcvd": 0,
             "oos_drop_count": int(cum(oos, t)), "seq_err_naks_rcvd": 0, "local_ack_timeout_err": -1,
             "out_of_buffer": -1, "rnr_retry_err": int(cum(rnr, t)),
             "pfc_rx_pause_us": rxp if time_counters else z, "pfc_tx_pause_us": txp if time_counters else z,
             "pfc_rx_frames": rxf, "pfc_tx_frames": [-1 if no_pfc_prio == i else 0 for i in range(8)]}
        if prio_bytes:
            s["prio_tx_bytes"] = pb
            s["prio_rx_bytes"] = pb
        samples.append(s)
        t += step
    return {"host_name": host, "process_id": str(pid), "clock": "CLOCK_MONOTONIC",
            "histogram_bucket_interval_ns": 30000, "histogram_max_buckets": 5,
            "sample_interval_ms": step // 1000,
            "devices": [{"device_id": 0, "roce_device": roce, "hw_type": "x", "link_gbps": link_gbps}],
            "hw_samples": samples}


def iterations(rank, host, pid, start=BUSY[0], n=20, dur=20_000, gap=0, size=64 * MB, comm="0x1",
               first_sn=0):
    return [accl_line(rank, host, pid, "AllReduce", size, first_sn + i, start + i * (dur + gap),
                      start + i * (dur + gap) + dur, comm) for i in range(n)]


class Run:
    """A directory holding one synthetic run, analysed with default thresholds."""

    def __init__(self, ranks, pfc_prio=None):
        d = tempfile.mkdtemp()
        atexit.register(shutil.rmtree, d, True)
        self.dirname = d
        for rank, host, pid, lines, tel in ranks:
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
        self.summary = fr.job_summary(self.results, self.lane)

    def rank(self, r):
        return next(g for g in self.results if g.rank == r)

    def cli(self, *extra):
        d = self.dirname
        return subprocess.run([sys.executable, os.path.join(HERE, "accl_fabric_report.py"), *extra,
                               "--accl", d, "--telemetry", d, "--warmup", "0"],
                              capture_output=True, text=True, check=True).stdout


def one(tel=None, lines=None, **kw):
    """Single rank 0 on hostA running 20 back-to-back 64MB AllReduces in BUSY."""
    return Run([(0, "hostA", 1, lines or iterations(0, "hostA", 1), tel or telemetry("hostA", 1, **kw))])


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
        s = fr.Series([0, 100, 200, 300], [0.0, 0.0, 1000.0, 1010.0])
        b = fr.bound(s, [(100, 300)], 50, False)
        self.assertEqual((b.lb, b.ub), (10.0, 1010.0))

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
        self.assertIn("PFC is not enabled on it", r.cli("report"))

    # --- the four false reports found in review --------------------------------

    def test_pause_on_another_priority_is_not_rccls(self):
        r = one(rx_pause={BUSY: 0.10}, pause_prio=0)      # storage on prio 0, RCCL on 3
        self.assertEqual((r.lane.prio, r.lane.source), (PRIO, "traffic"))
        self.assertIn(0, r.lane.other_pause)
        self.assertTrue(r.rank(0).verdict.startswith("CLEAN"))

    def test_lane_unconfirmed_without_priority_bytes(self):
        r = one(rx_pause={BUSY: 0.10}, prio_bytes=False)
        self.assertEqual(r.lane.source, "most-paused")
        self.assertIn("unconfirmed", next(iter(r.summary.values())))
        self.assertIn("UNCONFIRMED", r.cli("report"))

    def test_receiver_backpressure_is_not_blamed_on_fabric(self):
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: 0.10})),
                 (1, "hostB", 2, iterations(1, "hostB", 2), telemetry("hostB", 2, tx_pause={BUSY: 0.10}))])
        s = next(iter(r.summary.values()))
        self.assertTrue(s.startswith("HOST: rank 1 (hostB) paused the switch"), s)
        self.assertIn("likely backpressure", s)
        self.assertEqual(r.rank(0).verdict, "FABRIC-CONGESTION(echo?)")

    def pair(self, rx, tx):
        return Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: rx})),
                    (1, "hostB", 2, iterations(1, "hostB", 2), telemetry("hostB", 2, tx_pause={BUSY: tx}))])

    def test_pushback_under_threshold_still_counts(self):
        # 4% pushback is under the 5% threshold but explains the senders' 8%, minus 4%.
        s = next(iter(self.pair(0.08, 0.04).summary.values()))
        self.assertTrue(s.startswith("HOST: rank 1 (hostB) paused the switch (up to 4.0%)"), s)
        # With 10% on the senders, 6% is left over: more than the threshold.
        s = next(iter(self.pair(0.10, 0.04).summary.values()))
        self.assertTrue(s.startswith("FABRIC+HOST"), s)
        self.assertNotIn("no NIC pushed back", s)

    def test_fabric_beyond_what_pushback_explains(self):
        s = next(iter(self.pair(0.20, 0.04).summary.values()))
        self.assertTrue(s.startswith("FABRIC+HOST: switch paused senders at least 20.0%"), s)
        self.assertIn("roughly 16.0% or more is the fabric", s)

    def test_possible_pushback_is_hedged(self):
        # Receiver's pause falls between its short collectives: provable 0, possible > 0.
        gaps = {(BUSY[0] + i * 10_000 + 100, BUSY[0] + (i + 1) * 10_000): 0.5 for i in range(40)}
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: 0.10})),
                 (1, "hostB", 2, iterations(1, "hostB", 2, n=40, dur=100, gap=9_900),
                  telemetry("hostB", 2, tx_pause=gaps))])
        s = next(iter(r.summary.values()))
        self.assertTrue(s.startswith("HOST?: rank 1 (hostB) may have paused the switch"), s)

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
        self.assertEqual(r.rank(0).verdict, "FABRIC-LOSS")
        self.assertTrue(next(iter(r.summary.values())).startswith("FABRIC: drops"))

    def test_receiver_not_ready(self):
        self.assertEqual(one(rnr={BUSY: 0.0001}).rank(0).verdict, "HOST-RNR")

    def test_ecn_needs_a_share_of_packets(self):
        # RCCL alone is ~2.4 packets/us; 0.1 marks/us is ~4%, 1 mark is ~0%.
        self.assertEqual(one(ecn={BUSY: 0.1}).rank(0).verdict, "CONGESTION-MARKS")
        self.assertTrue(one(ecn={(BUSY[0], BUSY[0] + 1): 1.0}).rank(0).verdict.startswith("CLEAN"))

    def test_cnps_are_a_rate_not_a_share(self):
        # 300 CNPs/s is about 0.01% of packets here, yet a flow slowed all the time.
        g = one(cnp={BUSY: 300e-6}).rank(0)
        self.assertEqual(g.verdict, "CONGESTION-MARKS")
        self.assertAlmostEqual(g.nics[0].cnp_ps[0], 300, delta=30)
        self.assertTrue(one(cnp={BUSY: 20e-6}).rank(0).verdict.startswith("CLEAN"))

    def test_frames_only_nic_uses_frames(self):
        r = one(rx_frames={BUSY: 500e-6}, time_counters=False, roce="bnxt_re0")
        g = r.rank(0)
        self.assertEqual(g.nics[0].unit, "/s")
        self.assertAlmostEqual(g.nics[0].rx_pause[0], 500, delta=10)
        self.assertEqual(g.verdict, "FABRIC-CONGESTION")

    def test_mixed_nics_keep_their_own_unit(self):
        r = Run([(0, "hostA", 1, iterations(0, "hostA", 1), telemetry("hostA", 1, rx_pause={BUSY: 0.1})),
                 (1, "hostB", 2, iterations(1, "hostB", 2),
                  telemetry("hostB", 2, rx_frames={BUSY: 500e-6}, time_counters=False))])
        self.assertEqual(r.rank(0).nics[0].unit, "%")
        self.assertEqual(r.rank(1).nics[0].unit, "/s")
        self.assertEqual(r.rank(1).verdict, "FABRIC-CONGESTION")

    def test_slow_refresh_widens_windows(self):
        tel = telemetry("hostA", 1, refresh_us=100_000)
        r = one(tel=tel)
        dev = r.tels[("hostA", 1)].devices[0]
        self.assertGreaterEqual(fr.refresh_us(dev, 10_000), 90_000)

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


if __name__ == "__main__":
    unittest.main()
