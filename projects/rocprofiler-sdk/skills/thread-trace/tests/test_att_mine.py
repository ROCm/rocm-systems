#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""GPU-free unit tests for the thread-trace reading script, on synthetic records."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "att_mine.py"


def load_module():
    spec = importlib.util.spec_from_file_location("att_mine", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mine = load_module()

CATEGORY = {
    "s_waitcnt": "IMMED",
    "s_barrier": "IMMED",
    "s_endpgm": "IMMED",
    "s_load_dword": "SMEM",
    "s_add_i32": "SALU",
    "global_load_dword": "VMEM",
}


def inst(time, text, latency=4, idle=0, addr=None, source="/src/k.hip:10"):
    op = text.split(" ", 1)[0]
    category = CATEGORY.get(op, "VALU" if op.startswith("v_") else "SALU")
    pc = mine.Pc(addr if addr is not None else time, 1)
    return mine.Inst(time, latency, 0, idle, category, text, source, pc)


def wave(insts, simd=0, wave_id=0, workgroup=0, states=None):
    begin = insts[0].time if insts else 0
    end = max(i.time + i.latency for i in insts) if insts else 0
    return mine.WaveTrace(
        0, 1, simd, wave_id, workgroup, begin, end, 0, list(insts), Counter(states or {})
    )


class ParsingTests(unittest.TestCase):
    def test_short_source_prefers_call_site(self):
        self.assertEqual(mine.short_source("/a/b/k.hip:12"), "k.hip:12")
        self.assertEqual(
            mine.short_source("/x/h.h:5 -> /a/k.hip:9"), "k.hip:9 (inlined h.h:5)"
        )
        self.assertEqual(mine.short_source(""), "?")

    def test_latency_is_issue_plus_stall_as_codeindex_counts_it(self):
        i = mine.Inst(0, 4, 90, 0, "IMMED", "s_waitcnt vmcnt(0)", "", mine.Pc(0x10, 1))
        self.assertEqual(i.latency, 90)
        self.assertEqual(i.cost, 90)


class StatsCsvTests(unittest.TestCase):
    def test_ranks_instructions_and_lines(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "stats_ui_output_agent_1_dispatch_3.csv").write_text(
                '"CodeObj","Vaddr","Instruction","Hitcount","Latency","Stall","Idle","Source"\n'
                '2,0,"; _Z1kv",0,0,0,0,"k()"\n'
                '2,4,"s_waitcnt vmcnt(0)",10,900,900,0,"/a/k.hip:7"\n'
                '2,8,"global_load_dword v1, v[2:3], off",10,60,20,0,"/a/k.hip:7"\n'
                '2,12,"s_barrier",10,30,30,10,"/a/k.hip:9"\n'
            )
            stats = mine.read_stats(d, top=5)
        self.assertEqual(stats["instructions"][0]["text"], "s_waitcnt vmcnt(0)")
        self.assertEqual(stats["instructions"][0]["share"], 90.0)
        self.assertEqual(stats["lines"][0]["line"], "k.hip:7")
        self.assertEqual(stats["lines"][0]["cost"], 960)
        self.assertEqual(stats["lines"][1]["cost"], 40)


class HotspotTests(unittest.TestCase):
    def test_hotspots_without_hidden_latency_rank_by_cost(self):
        w = wave(
            [
                inst(0, "v_add_f32 v0, v0, v1", latency=4, addr=0x10),
                inst(4, "s_waitcnt vmcnt(0)", latency=400, addr=0x14),
            ]
        )
        top = mine.Capture.from_waves([w]).hotspots(top=1)[0]
        self.assertTrue(top["text"].startswith("s_waitcnt"))
        self.assertEqual(top["non_hidden"], 400)

    def test_lines_group_instructions_by_source_line(self):
        w = wave(
            [
                inst(0, "v_add_f32 v0, v0, v1", latency=10, addr=0x10, source="/k.hip:5"),
                inst(
                    10, "v_mul_f32 v0, v0, v1", latency=30, addr=0x14, source="/k.hip:5"
                ),
                inst(40, "s_barrier", latency=60, addr=0x18, source="/k.hip:9"),
            ]
        )
        lines = mine.Capture.from_waves([w]).hotspots(by_line=True)
        self.assertEqual([r["line"] for r in lines], ["k.hip:9", "k.hip:5"])
        self.assertEqual(lines[1]["cost"], 40)
        self.assertEqual(lines[1]["share"], 40.0)


class SummaryTests(unittest.TestCase):
    def test_wave_state_split_and_dispatch_resources(self):
        a = wave([inst(0, "v_add_f32 v0, v0, v1")], states={"EXEC": 30, "WAIT": 70})
        b = wave(
            [inst(0, "v_add_f32 v0, v0, v1")], wave_id=1, states={"EXEC": 10, "STALL": 90}
        )
        dispatch = SimpleNamespace(
            time=0,
            me_id=0,
            pipe_id=0,
            vgprs=64,
            sgprs=32,
            lds_size=4096,
            thread_dim_x=256,
            thread_dim_y=1,
            thread_dim_z=1,
        )
        s = mine.Capture.from_waves([a, b], dispatches=[dispatch]).summary()
        self.assertEqual(s["waves"], 2)
        self.assertEqual(s["wave_states"], {"STALL": 45.0, "WAIT": 35.0, "EXEC": 20.0})
        self.assertEqual(s["dispatch"]["vgprs"], 64)
        self.assertEqual(s["dispatch"]["workgroup"], [256, 1, 1])


def timed(time, text, duration, stall=0, category="VALU", addr=None):
    return mine.Inst(
        time,
        duration,
        stall,
        0,
        category,
        text,
        "/k.hip:3",
        mine.Pc(addr if addr is not None else time, 1),
    )


class PipeTests(unittest.TestCase):
    def test_classes_split_valu_and_name_waits_and_barriers(self):
        self.assertEqual(
            mine.inst_class(timed(0, "v_exp_f32_e32 v1, v1", 16)), "VALU transcendental"
        )
        self.assertEqual(
            mine.inst_class(timed(0, "v_rcp_iflag_f32 v1, v1", 16)), "VALU transcendental"
        )
        self.assertEqual(
            mine.inst_class(
                timed(0, "v_mfma_f32_32x32x8_bf16 a[0:15], v[0:1], v[2:3], 0", 4)
            ),
            "VALU matrix",
        )
        self.assertEqual(mine.inst_class(timed(0, "v_fma_f32 v1, v2, v3, v4", 4)), "VALU")
        self.assertEqual(
            mine.inst_class(timed(0, "s_waitcnt vmcnt(0)", 90, 90, "IMMED")),
            "wait (s_waitcnt, s_wait_*)",
        )
        self.assertEqual(
            mine.inst_class(timed(0, "s_wait_loadcnt 0x0", 9, 9, "IMMED")),
            "wait (s_waitcnt, s_wait_*)",
        )
        self.assertEqual(
            mine.inst_class(timed(0, "s_barrier", 50, 48, "IMMED")), "barrier (s_barrier)"
        )
        self.assertEqual(
            mine.inst_class(timed(0, "ds_read_b128 v[0:3], v4", 8, 4, "LDS")), "LDS"
        )

    def test_issue_is_duration_minus_stall_against_simd_time(self):
        # Two waves on one SIMD, resident from 0 to 100: 3 transcendental instructions issuing
        # 16 cycles each after a 10-cycle stall, and 5 ordinary VALU instructions issuing 4.
        a = wave(
            [
                timed(0, "v_exp_f32_e32 v1, v1", 26, 10),
                timed(26, "v_exp_f32_e32 v1, v1", 26, 10),
                timed(52, "v_add_f32 v1, v1, v2", 4),
                timed(96, "s_endpgm", 4, 0, "IMMED"),
            ]
        )
        b = wave(
            [
                timed(10, "v_exp_f32_e32 v1, v1", 26, 10),
                timed(40, "v_add_f32 v1, v1, v2", 4),
                timed(44, "v_add_f32 v1, v1, v2", 4),
                timed(48, "v_add_f32 v1, v1, v2", 4),
                timed(52, "v_add_f32 v1, v1, v2", 4),
            ],
            wave_id=1,
        )
        p = mine.Capture.from_waves([a, b]).pipes()
        by = {c["class"]: c for c in p["classes"]}
        self.assertEqual(p["simds"], 1)
        self.assertEqual(p["simd_cycles"], 100)
        self.assertEqual(by["VALU transcendental"]["issue_share"], 48.0)
        self.assertEqual(by["VALU transcendental"]["cycles_each"], 16.0)
        self.assertEqual(by["VALU transcendental"]["stall_share"], 100.0)
        self.assertEqual(by["VALU transcendental"]["per_wave"], 1.5)
        self.assertEqual(by["VALU"]["issue_share"], 20.0)
        self.assertEqual(p["valu_issue_share"], 68.0)
        self.assertEqual(p["classes"][0]["class"], "VALU transcendental")

    def test_waves_on_different_simds_add_their_resident_time(self):
        a = wave(
            [timed(0, "v_add_f32 v1, v1, v2", 4), timed(96, "v_add_f32 v1, v1, v2", 4)]
        )
        b = wave(
            [timed(0, "v_add_f32 v1, v1, v2", 4), timed(46, "v_add_f32 v1, v1, v2", 4)],
            simd=1,
        )
        p = mine.Capture.from_waves([a, b]).pipes()
        self.assertEqual((p["simds"], p["simd_cycles"]), (2, 150))
        self.assertEqual(p["valu_issue_share"], round(100 * 16 / 150, 1))


class NextReadTests(unittest.TestCase):
    def pipes(self, **stall):
        return {
            "valu_issue_share": stall.pop("valu", 10.0),
            "classes": [{"class": k, "stall_share": v} for k, v in stall.items()],
        }

    def test_the_largest_wave_state_picks_the_page(self):
        waits = self.pipes(**{"wait (s_waitcnt, s_wait_*)": 90, "barrier (s_barrier)": 5})
        self.assertIn(
            "resources/latency.md", mine.next_read({"WAIT": 80, "EXEC": 20}, waits)
        )
        self.assertIn(
            "resources/stalls.md (section 'Idle cycles')",
            mine.next_read({"IDLE": 60, "EXEC": 40}),
        )
        self.assertIn("resources/compute.md", mine.next_read({"STALL": 60, "WAIT": 40}))
        self.assertEqual(mine.next_read({}), "")

    def test_a_barrier_that_holds_the_wait_points_to_synchronization(self):
        barrier = self.pipes(
            **{"barrier (s_barrier)": 90, "wait (s_waitcnt, s_wait_*)": 5}
        )
        self.assertIn(
            "resources/synchronization.md",
            mine.next_read({"WAIT": 80, "EXEC": 20}, barrier),
        )

    def test_stall_names_the_class_that_holds_it_and_a_saturated_valu(self):
        busy = self.pipes(
            valu=112.5,
            **{"VALU transcendental": 64, "barrier (s_barrier)": 17, "VALU": 11},
        )
        hint = mine.next_read({"STALL": 62, "WAIT": 19, "EXEC": 19}, busy)
        self.assertIn("VALU transcendental instructions hold most of the stall", hint)
        self.assertIn("112.5%", hint)
        self.assertIn("resources/compute.md (section 'Ceiling check')", hint)
        self.assertTrue(
            Path(hint.split("read ", 1)[1].split(" (section", 1)[0]).is_file()
        )
        idle = mine.next_read({"IDLE": 60, "EXEC": 40})
        self.assertIn("resources/stalls.md (section 'Idle cycles')", idle)
        self.assertTrue(
            Path(idle.split("read ", 1)[1].split(" (section", 1)[0]).is_file()
        )

    def test_the_stats_csv_names_a_page_by_its_costliest_instruction(self):
        row = lambda text: {"instructions": [{"text": text}]}
        self.assertIn(
            "resources/synchronization.md", mine.next_read_from_stats(row("s_barrier"))
        )
        self.assertIn("`barriers`", mine.next_read_from_stats(row("s_barrier")))
        self.assertIn(
            "resources/latency.md", mine.next_read_from_stats(row("s_waitcnt vmcnt(0)"))
        )
        self.assertIn(
            "resources/compute.md", mine.next_read_from_stats(row("v_exp_f32_e32 v1, v1"))
        )
        self.assertEqual(mine.next_read_from_stats({"instructions": []}), "")

    def test_a_flat_ranking_gets_a_note_and_a_peaked_one_does_not(self):
        self.assertIn("`pipes`", mine.flat_profile_note([{"share": 2.8}], "instruction"))
        self.assertEqual(mine.flat_profile_note([{"share": 40.0}], "instruction"), "")
        self.assertEqual(mine.flat_profile_note([], "source line"), "")


class BarrierTests(unittest.TestCase):
    def workgroup(self, begin, leader_simd):
        """Four waves launched together; the leader runs a long series while the others wait."""
        waves = []
        for simd in range(4):
            if simd == leader_simd:
                insts = [
                    mine.Inst(
                        begin + 4,
                        900,
                        0,
                        0,
                        "VALU",
                        "v_fma_f32 v1, v2, v3, v4",
                        "/k.hip:23",
                        mine.Pc(0x10, 1),
                    ),
                    mine.Inst(
                        begin + 904,
                        4,
                        0,
                        0,
                        "IMMED",
                        "s_barrier",
                        "/k.hip:24",
                        mine.Pc(0x14, 1),
                    ),
                ]
            else:
                insts = [
                    mine.Inst(
                        begin + 4,
                        4,
                        0,
                        0,
                        "VALU",
                        "v_mov_b32 v1, 0",
                        "/k.hip:22",
                        mine.Pc(0x08, 1),
                    ),
                    mine.Inst(
                        begin + 908,
                        4,
                        896,
                        0,
                        "IMMED",
                        "s_barrier",
                        "/k.hip:24",
                        mine.Pc(0x14, 1),
                    ),
                ]
            waves.append(
                mine.WaveTrace(
                    0, 1, simd, 0, 0, begin + simd, begin + 912, 0, insts, Counter()
                )
            )
        return waves

    def test_the_waited_for_wave_is_found_in_each_workgroup(self):
        dispatch = SimpleNamespace(
            time=0,
            me_id=0,
            pipe_id=0,
            vgprs=32,
            sgprs=16,
            lds_size=0,
            thread_dim_x=256,
            thread_dim_y=1,
            thread_dim_z=1,
        )
        waves = self.workgroup(1000, 0) + self.workgroup(5000, 0)
        b = mine.Capture.from_waves(waves, dispatches=[dispatch]).barriers()
        self.assertEqual((b["workgroups"], b["waves_per_workgroup"]), (2, 4))
        self.assertEqual(b["least_waiting_rank"], {"rank 0": "100%"})
        self.assertLess(b["least_waiting_barrier_share_p50"], 1)
        self.assertGreater(b["others_barrier_share_p50"], 90)
        self.assertEqual(b["least_waiting_wave_ran"][0]["line"], "k.hip:23")
        self.assertEqual(b["others_waited_at"][0]["line"], "k.hip:24")

    def test_waves_launched_apart_are_different_workgroups(self):
        dispatch = SimpleNamespace(
            time=0,
            me_id=0,
            pipe_id=0,
            vgprs=32,
            sgprs=16,
            lds_size=0,
            thread_dim_x=128,
            thread_dim_y=1,
            thread_dim_z=1,
        )
        b = mine.Capture.from_waves(
            self.workgroup(1000, 1), dispatches=[dispatch]
        ).barriers()
        # 128 threads is two waves per workgroup, so four waves launched together are two workgroups.
        self.assertEqual((b["workgroups"], b["waves_per_workgroup"]), (2, 2))


class LifetimeTests(unittest.TestCase):
    def test_components_follow_the_wave_lifetime_sample(self):
        w = wave(
            [
                inst(0, "v_add_f32 v0, v0, v1", latency=10),
                inst(10, "s_waitcnt vmcnt(0)", latency=50),
                inst(80, "s_add_i32 s0, s0, 1", latency=20),
            ]
        )  # 20 idle cycles before it
        parts = {
            r["component"]: r
            for r in mine.Capture.from_waves([w]).lifetime()["components"]
        }
        self.assertEqual(parts["valu_latency"]["per_wave_p50"], 10)
        self.assertEqual(parts["s_wait_latency"]["per_wave_p50"], 50)
        self.assertEqual(parts["non_valu_latency"]["per_wave_p50"], 20)
        self.assertEqual(parts["idle_time"]["per_wave_p50"], 20)
        self.assertEqual(parts["s_wait_latency"]["share_of_lifetime"], 50.0)

    def test_fit_reports_which_component_grows_with_lifetime(self):
        short = wave(
            [inst(0, "s_waitcnt vmcnt(0)", latency=10), inst(10, "v_add_f32 v0, v0, v1")]
        )
        long = wave(
            [
                inst(0, "s_waitcnt vmcnt(0)", latency=210),
                inst(210, "v_add_f32 v0, v0, v1"),
            ],
            wave_id=1,
        )
        parts = {
            r["component"]: r
            for r in mine.Capture.from_waves([short, long]).lifetime()["components"]
        }
        self.assertEqual(parts["s_wait_latency"]["lifetime_slope"], 1.0)
        self.assertEqual(parts["s_wait_latency"]["r"], 1.0)
        self.assertNotIn("lifetime_slope", parts["valu_latency"])


class OccupancyTests(unittest.TestCase):
    def occ(self, time, wave_id, start, simd=0):
        return SimpleNamespace(
            time=time,
            cu=1,
            simd=simd,
            wave_id=wave_id,
            me_id=0,
            pipe_id=0,
            workgroup_id=0,
            start=start,
        )

    def test_active_waves_and_registers_follow_the_occupancy_sample(self):
        dispatch = SimpleNamespace(
            time=0,
            me_id=0,
            pipe_id=0,
            vgprs=64,
            sgprs=32,
            lds_size=0,
            thread_dim_x=64,
            thread_dim_y=1,
            thread_dim_z=1,
        )
        records = [
            self.occ(10, 0, 1),
            self.occ(20, 1, 1, simd=1),
            self.occ(60, 0, 0),
            self.occ(110, 1, 0, simd=1),
        ]
        cap = mine.Capture.from_waves([], dispatches=[dispatch], occupancy=records)
        rows = cap.occupancy_rows()
        self.assertEqual([r["active_waves"] for r in rows], [1, 2, 1, 0])
        self.assertEqual(rows[1]["active_vgprs"], 128)
        report = cap.occupancy_report(bins=4)
        self.assertEqual(report["occupancy_simds"], 2)
        self.assertEqual(report["peak_active_waves"], 2)
        # 1 wave for 10 cycles, 2 for 40, 1 for 50, over 100 cycles.
        self.assertEqual(report["mean_active_waves"], 1.4)
        self.assertEqual(report["active_waves_over_time"], [2, 2, 1, 1])


class GpuPropertiesTests(unittest.TestCase):
    def test_reads_the_rocpd_database_and_the_csv(self):
        import json
        import sqlite3
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            with sqlite3.connect(Path(d) / "cap_results.db") as db:
                db.execute(
                    "CREATE TABLE rocpd_info_agent (type TEXT, product_name TEXT, name TEXT, "
                    "extdata TEXT)"
                )
                db.execute(
                    "INSERT INTO rocpd_info_agent VALUES ('CPU', 'cpu', 'cpu', '{}')"
                )
                db.execute(
                    "INSERT INTO rocpd_info_agent VALUES ('GPU', 'GPU X', 'gfxX', ?)",
                    (json.dumps({"cu_count": 304, "simd_per_cu": 4, "size": 312}),),
                )
            gpu = mine.gpu_properties(Path(d))
        self.assertEqual(
            gpu, {"product": "GPU X", "name": "gfxX", "cu_count": 304, "simd_per_cu": 4}
        )
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "1_agent_info.csv").write_text(
                '"Agent_Type","Name","Product_Name","Cu_Count","Max_Waves_Per_Simd"\n'
                '"CPU","cpu","cpu",24,0\n"GPU","gfxY","GPU Y",64,10\n'
            )
            gpu = mine.gpu_properties(Path(d))
        self.assertEqual(
            gpu,
            {
                "product": "GPU Y",
                "name": "gfxY",
                "cu_count": 64,
                "max_waves_per_simd": 10,
            },
        )


class CompareTests(unittest.TestCase):
    def test_cost_per_line_per_wave(self):
        a = wave([inst(0, "v_add_f32 v0, v0, v1", latency=40, source="/k.hip:3")])
        b = wave(
            [inst(0, "v_add_f32 v0, v0, v1", latency=20, source="/k.hip:3")], wave_id=1
        )
        self.assertEqual(
            mine.Capture.from_waves([a, b]).cost_by_line(), {"k.hip:3": 30.0}
        )


class DecoderSetupTests(unittest.TestCase):
    def test_a_library_that_does_not_load_names_the_variable(self):
        import tempfile
        from unittest import mock

        def no_library():
            raise OSError("librocprof-trace-decoder.so: cannot open shared object file")

        decoder = SimpleNamespace(
            CodeObject=lambda path, cid: (path, cid),
            HiddenLatency=object,
            analyze_hidden_latency=None,
            Decoder=no_library,
            generate_code_artifacts=lambda objs: SimpleNamespace(code_index=None),
        )
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(
            sys.modules, {"rocprof_trace_decoder": decoder}
        ):
            (Path(d) / "k_shader_engine_0_1.att").write_bytes(b"")
            (Path(d) / "k_code_object_id_1.out").write_bytes(b"")
            with self.assertRaises(SystemExit) as err:
                mine.Capture(d)
        self.assertIn("ROCPROF_TRACE_DECODER_LIB", str(err.exception))

    def test_missing_pyelftools_is_named(self):
        import importlib.abc
        import tempfile
        from unittest import mock

        class NoElftools(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "rocprof_trace_decoder":
                    raise ModuleNotFoundError(
                        "No module named 'elftools'", name="elftools"
                    )
                return None

        finder = NoElftools()
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(sys.modules):
            sys.modules.pop("rocprof_trace_decoder", None)
            sys.meta_path.insert(0, finder)
            try:
                with self.assertRaises(SystemExit) as err:
                    mine.Capture(d)
            finally:
                sys.meta_path.remove(finder)
        self.assertIn("pyelftools", str(err.exception))


if __name__ == "__main__":
    unittest.main()
