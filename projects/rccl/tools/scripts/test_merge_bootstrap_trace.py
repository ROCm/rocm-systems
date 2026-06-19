#!/usr/bin/env python3

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).with_name("merge_bootstrap_trace.py")


def load_merge_module():
    spec = importlib.util.spec_from_file_location("merge_bootstrap_trace", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = load_merge_module()


def make_event(t_ns, rank, phase, md, dur_us, nbytes):
    return struct.pack(MOD.EVT_FMT, t_ns, rank, phase, md, dur_us, nbytes)


def make_net(t_ns, rank, phase, md, rtt, rttvar, rto, cwnd, retrans, lost, unacked, ca):
    return struct.pack(MOD.NET_FMT, t_ns, rank, phase, md, rtt, rttvar, rto,
                       cwnd, retrans, lost, unacked, ca, 0)


def make_v2_dump(rank, is_root, events, nets, ev_ovf=0, net_ovf=0):
    hdr = struct.pack(MOD.HDR_V2_FMT, MOD.MAGIC, 2, MOD.ENDIAN_MARK, rank,
                      is_root, len(events), len(nets), ev_ovf, net_ovf, 0)
    return hdr + b"".join(events) + b"".join(nets)


def make_v1_dump(rank, is_root, events):
    hdr = struct.pack(MOD.HDR_V1_FMT, MOD.MAGIC, 1, rank, is_root, len(events), 0)
    return hdr + b"".join(events)


class MergeBootstrapTraceTest(unittest.TestCase):
    def test_collects_deploy_trace_text_events(self):
        with tempfile.TemporaryDirectory() as td:
            trace_dir = Path(td)
            (trace_dir / "rank0.log").write_text(
                "prefix DEPLOY_TRACE rank=0 pid=123 phase=deploy.exec "
                "t_ns=1000 dur_us=0 md=0 bytes=0 detail=constructor\n"
                "prefix DEPLOY_TRACE rank=0 pid=123 phase=mpi.init "
                "t_ns=2000 dur_us=50 md=0 bytes=0 detail=MPI_Init\n"
            )
            events = MOD.collect([trace_dir])
        self.assertEqual(len(events), 2)
        self.assertEqual([e.name() for e in events], ["deploy.exec", "mpi.init"])
        self.assertEqual(events[0].rank, 0)
        self.assertEqual(events[1].dur_us, 50)

    def test_v2_binary_with_events_and_net_samples(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            evs = [make_event(1000, 3, 6, 0, 120, 256),   # recv_from_root
                   make_event(2000, 3, 12, 0, 340, 0)]    # ring_allgather
            nets = [make_net(2100, 3, 202, 0, 150, 20, 200000, 10, 2, 1, 0, 0),
                    make_net(2200, 3, 12, 0, 900, 80, 200000, 10, 7, 3, 1, 1)]
            (d / "rank00003_pid42.bin").write_bytes(make_v2_dump(3, 0, evs, nets))
            net_out = []
            events = MOD.collect([d], net_out)
        self.assertEqual(len(events), 2)
        self.assertEqual(len(net_out), 2)
        self.assertEqual(net_out[0].name(), "tcp.ready")
        self.assertEqual(net_out[0].rtt_us, 150)
        self.assertEqual(net_out[1].total_retrans, 7)
        self.assertEqual(net_out[1].ca_state, 1)

    def test_multi_segment_dump(self):
        # dumpThreadBuffer appends one segment per flush; reader must get all.
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            seg1 = make_v2_dump(2, 0, [make_event(100, 2, 6, 0, 120, 256)],
                                [make_net(110, 2, 202, 0, 150, 20, 201000, 10, 0, 0, 0, 0)])
            seg2 = make_v2_dump(2, 0, [make_event(200, 2, 12, 0, 340, 0)],
                                [make_net(210, 2, 12, 0, 900, 80, 201000, 10, 5, 2, 1, 1)])
            (d / "rank00002_pid7.bin").write_bytes(seg1 + seg2)
            net_out = []
            events = MOD.collect([d], net_out)
        self.assertEqual(len(events), 2)
        self.assertEqual(sorted(e.name() for e in events), ["recv_from_root", "ring_allgather"])
        self.assertEqual(len(net_out), 2)
        self.assertEqual(net_out[1].total_retrans, 5)

    def test_v1_dump_still_reads(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "rank00000_pid1.bin").write_bytes(
                make_v1_dump(0, 0, [make_event(500, 0, 0, 0, 999, 0)]))
            net_out = []
            events = MOD.collect([d], net_out)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].dur_us, 999)
        self.assertEqual(net_out, [])

    def test_bnetstat_text_parsing(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "job.out").write_text(
                "host:1:1 [0] NCCL INFO BNETSTAT rank=5 root=0 p=7 name=ring_allgather "
                "t_ns=9000 md=0 rtt_us=420 rttvar_us=55 rto_us=204000 cwnd=10 "
                "retrans=4 lost=2 unacked=1 ca_state=3\n"
            )
            net_out = []
            MOD.collect([d], net_out)
        self.assertEqual(len(net_out), 1)
        self.assertEqual(net_out[0].name(), "ring_allgather")
        self.assertEqual(net_out[0].rtt_us, 420)
        self.assertEqual(net_out[0].total_retrans, 4)
        self.assertEqual(net_out[0].ca_state, 3)

    def test_bad_magic_skipped(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "junk.bin").write_bytes(b"\x00" * 64)
            net_out = []
            events = MOD.collect([d], net_out)
        self.assertEqual(events, [])
        self.assertEqual(net_out, [])

    def test_perfetto_counter_tracks(self):
        import json
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            evs = [make_event(1000, 3, 6, 0, 120, 256)]
            nets = [make_net(1100, 3, 202, 0, 150, 20, 201000, 10, 4, 1, 0, 0)]
            (d / "r.bin").write_bytes(make_v2_dump(3, 0, evs, nets))
            net_out = []
            events = MOD.collect([d], net_out)
            out = Path(td) / "trace.json"
            MOD.write_chrome_trace(events, out, net_out)
            data = json.loads(out.read_text())
        counters = [e for e in data["traceEvents"] if e.get("ph") == "C"]
        self.assertEqual(len(counters), 1)
        self.assertEqual(counters[0]["args"]["rtt_us"], 150)
        self.assertEqual(counters[0]["args"]["total_retrans"], 4)
        # phase slice present too
        self.assertTrue(any(e.get("ph") == "X" for e in data["traceEvents"]))

    def test_net_csv_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            nets = [make_net(2100, 3, 202, 0, 150, 20, 200000, 10, 2, 1, 0, 0)]
            (d / "r.bin").write_bytes(make_v2_dump(3, 0, [], nets))
            net_out = []
            MOD.collect([d], net_out)
            out_csv = Path(td) / "net.csv"
            MOD.write_net_csv(net_out, out_csv)
            lines = out_csv.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)  # header + 1 row
        self.assertIn("tcp.ready", lines[1])


if __name__ == "__main__":
    unittest.main()
