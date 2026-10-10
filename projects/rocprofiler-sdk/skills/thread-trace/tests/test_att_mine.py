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

    def test_latency_is_the_duration_which_includes_the_stall(self):
        i = mine.Inst(0, 90, 86, 0, "IMMED", "s_waitcnt vmcnt(0)", "", mine.Pc(0x10, 1))
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
    def test_resources_come_from_the_dispatch_the_waves_belong_to(self):
        other = SimpleNamespace(
            me_id=0,
            pipe_id=0,
            time=0,
            vgprs=8,
            sgprs=8,
            lds_size=0,
            thread_dim_x=1,
            thread_dim_y=1,
            thread_dim_z=1,
        )
        traced = SimpleNamespace(
            me_id=0,
            pipe_id=0,
            time=500,
            vgprs=176,
            sgprs=112,
            lds_size=32768,
            thread_dim_x=256,
            thread_dim_y=1,
            thread_dim_z=1,
        )
        waves = [
            wave([inst(1000 + k, "v_add_f32 v0, v0, v1")], wave_id=k) for k in range(3)
        ]
        s = mine.Capture.from_waves(waves, dispatches=[traced, other]).summary()
        self.assertEqual(s["dispatch"]["vgprs"], 176)
        self.assertEqual(s["dispatches_with_traced_waves"], 1)

    def test_idle_before_the_first_instruction_is_not_counted_twice(self):
        # The idle before the first instruction is the IDLE state; only later gaps count.
        insts = [
            mine.Inst(
                10, 4, 0, 10, "VALU", "v_add_f32 v0, v0, v1", "/k.hip:1", mine.Pc(0, 1)
            ),
            mine.Inst(
                24, 4, 0, 10, "VALU", "v_add_f32 v0, v0, v1", "/k.hip:1", mine.Pc(4, 1)
            ),
        ]
        w = mine.WaveTrace(
            0, 1, 0, 0, 0, 0, 28, 0, insts, Counter({"IDLE": 10, "EXEC": 18})
        )
        self.assertEqual(
            mine.Capture.from_waves([w]).summary()["idle_between_instructions"],
            round(100 * 10 / 28, 1),
        )

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
    def test_a_capture_with_no_waves_reports_no_resident_time(self):
        result = mine.Capture.from_waves([]).pipes()
        self.assertEqual(result["waves"], 0)
        self.assertEqual(result["resident_cycles"], 0)
        self.assertEqual(result["valu_busy"], 0.0)

    def test_classes_split_valu_and_name_waits_and_barriers(self):
        self.assertEqual(mine.inst_class(timed(0, "v_exp_f32_e32 v1, v1", 16)), "VALU")
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

    def test_busy_share_counts_instructions_issued_together_once(self):
        # Two waves on one SIMD, resident from 0 to 100. Their VALU issue intervals
        # (time + stall to time + duration) overlap; the union covers 46 cycles.
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
        self.assertEqual((p["simds"], p["resident_cycles"]), (1, 100))
        self.assertEqual(by["VALU"]["busy_share"], 46.0)
        self.assertEqual(p["valu_busy"], 46.0)
        self.assertEqual(by["VALU"]["cycles_each"], 8.5)
        self.assertEqual(by["VALU"]["stall_share"], 100.0)
        self.assertEqual(by["VALU"]["per_wave"], 4.0)
        self.assertEqual(by["VALU"]["top_stalled"], "v_exp_f32_e32")

    def test_wait_time_is_kept_apart_from_stall(self):
        # A wait's "stall" is the time it waited; it must not dilute the VALU stall share.
        w = wave(
            [
                timed(0, "v_exp_f32_e32 v1, v1", 30, 26),
                timed(30, "s_waitcnt vmcnt(0)", 90, 90, "IMMED"),
            ]
        )
        by = {c["class"]: c for c in mine.Capture.from_waves([w]).pipes()["classes"]}
        self.assertEqual(
            (by["VALU"]["stall_share"], by["VALU"]["wait_share"]), (100.0, 0.0)
        )
        wait = by["wait (s_waitcnt, s_wait_*)"]
        self.assertEqual((wait["stall_share"], wait["wait_share"]), (0.0, 100.0))

    def test_busy_share_never_exceeds_resident_time(self):
        same = [timed(4 * k, "v_add_f32 v1, v1, v2", 4) for k in range(25)]
        waves = [wave(same, wave_id=k) for k in range(3)]  # three waves issuing together
        self.assertEqual(mine.Capture.from_waves(waves).pipes()["valu_busy"], 100.0)

    def test_waves_on_different_simds_add_their_resident_time(self):
        a = wave(
            [timed(0, "v_add_f32 v1, v1, v2", 4), timed(96, "v_add_f32 v1, v1, v2", 4)]
        )
        b = wave(
            [timed(0, "v_add_f32 v1, v1, v2", 4), timed(46, "v_add_f32 v1, v1, v2", 4)],
            simd=1,
        )
        p = mine.Capture.from_waves([a, b]).pipes()
        self.assertEqual((p["simds"], p["resident_cycles"]), (2, 150))
        self.assertEqual(p["valu_busy"], round(100 * 16 / 150, 1))


class NextReadTests(unittest.TestCase):
    def pipes(self, top_stalled="", **stall):
        return {
            "valu_busy": stall.pop("valu", 10.0),
            "classes": [
                {
                    "class": k,
                    "stall_share": 0.0 if k in mine.WAIT_CLASSES else v,
                    "wait_share": v if k in mine.WAIT_CLASSES else 0.0,
                    "top_stalled": top_stalled,
                }
                for k, v in stall.items()
            ],
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
            valu=95.5,
            top_stalled="v_exp_f32_e32",
            **{"VALU": 64, "barrier (s_barrier)": 17, "LDS": 11},
        )
        hint = mine.next_read({"STALL": 62, "WAIT": 19, "EXEC": 19}, busy)
        self.assertIn(
            "VALU instructions hold the largest share of the stall (most stalled: v_exp_f32_e32)",
            hint,
        )
        self.assertIn("95.5%", hint)
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
        def row(text):
            return {"instructions": [{"text": text}]}

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


class IdleRoutingTests(unittest.TestCase):
    def test_idle_between_instructions_counts_against_exec(self):
        # The decoder counts idle between instructions as EXEC; summary reports it apart.
        hint = mine.next_read({"EXEC": 70, "WAIT": 20, "STALL": 10}, idle_share=50)
        self.assertIn("resources/stalls.md", hint)
        self.assertIn("idle", hint)
        self.assertIn(
            "resources/compute.md", mine.next_read({"EXEC": 70, "WAIT": 20, "STALL": 10})
        )

    def test_stall_hint_names_the_class_whose_pipe_did_not_accept_it(self):
        p = {
            "classes": [{"class": "VALU matrix", "stall_share": 80}],
            "valu_busy": 20,
        }
        hint = mine.next_read({"STALL": 60, "EXEC": 40}, p)
        self.assertIn(
            "did not accept them, usually because the unit was busy or its queue full",
            hint,
        )
        self.assertNotIn("earlier result", hint)

    def test_exec_routes_to_the_ceiling_check(self):
        hint = mine.next_read({"EXEC": 70, "WAIT": 20, "STALL": 10})
        self.assertIn("compute.md (section 'Ceiling check')", hint)
        self.assertIn("EXEC is the largest wave state", hint)

    def test_stats_route_idle_dominated_top_instruction(self):
        rows = {
            "instructions": [{"text": "v_fma_f32 v1, v2, v3", "latency": 10, "idle": 90}]
        }
        self.assertIn("resources/stalls.md", mine.next_read_from_stats(rows))

    def test_the_idle_time_of_s_endpgm_is_the_wave_completing_not_a_stall(self):
        rows = {"instructions": [{"text": "s_endpgm", "latency": 10, "idle": 90}]}
        hint = mine.next_read_from_stats(rows)
        self.assertNotIn("stalls.md", hint)
        self.assertIn("wave completing", hint)


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
                        900,
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

    def test_a_large_workgroup_launched_over_many_cycles_stays_one_group(self):
        # 16 waves (1024 threads) launched 5 cycles apart: 75 cycles from first to last.
        dispatch = SimpleNamespace(
            time=0,
            me_id=0,
            pipe_id=0,
            vgprs=32,
            sgprs=16,
            lds_size=0,
            thread_dim_x=1024,
            thread_dim_y=1,
            thread_dim_z=1,
        )
        waves = []
        for k in range(16):
            begin = 1000 + 5 * k
            work = 900 if k == 15 else 4
            insts = [
                mine.Inst(
                    begin + 4,
                    work,
                    0,
                    0,
                    "VALU",
                    "v_fma_f32 v1, v2, v3, v4",
                    "/k.hip:23",
                    mine.Pc(0x10, 1),
                ),
                mine.Inst(
                    begin + 4 + work,
                    4 + (0 if k == 15 else 900 - work),  # duration includes the stall
                    0 if k == 15 else 900 - work,
                    0,
                    "IMMED",
                    "s_barrier",
                    "/k.hip:24",
                    mine.Pc(0x14, 1),
                ),
            ]
            waves.append(
                mine.WaveTrace(0, 1, k % 4, k // 4, 0, begin, 2000, 0, insts, Counter())
            )
        b = mine.Capture.from_waves(waves, dispatches=[dispatch]).barriers()
        self.assertEqual((b["workgroups"], b["waves_per_workgroup"]), (1, 16))
        self.assertEqual(b["least_waiting_rank"], {"rank 15": "100%"})

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

    def test_a_wave_ends_even_when_its_workgroup_id_changed(self):
        start, end = self.occ(10, 0, 1), self.occ(20, 0, 0)
        end.workgroup_id = 7
        cap = mine.Capture.from_waves([], dispatches=[], occupancy=[start, end])
        self.assertEqual([r["active_waves"] for r in cap.occupancy_rows()], [1, 0])

    def test_a_wave_ends_even_when_its_pipe_id_changed(self):
        # The decoder matches a wave's start and end by slot (cu, simd, wave); on gfx942 the end
        # record carried pipe_id 0 where the start had 1.
        start, end = self.occ(10, 0, 1), self.occ(20, 0, 0)
        start.pipe_id = 1
        cap = mine.Capture.from_waves([], dispatches=[], occupancy=[start, end])
        self.assertEqual([r["active_waves"] for r in cap.occupancy_rows()], [1, 0])

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


class WaitColumnAndScratchTests(unittest.TestCase):
    def test_wait_time_goes_to_the_wait_column_not_stall(self):
        load = mine.Inst(
            0, 40, 30, 0, "VMEM", "global_load_dword v1", "/a/k.hip:5", mine.Pc(0x10, 1)
        )
        wait = mine.Inst(
            40, 400, 400, 0, "IMMED", "s_waitcnt vmcnt(0)", "/a/k.hip:5", mine.Pc(0x14, 1)
        )
        rows = {
            r["text"].split(" ")[0]: r
            for r in mine.Capture.from_waves([wave([load, wait])]).hotspots()
        }
        self.assertEqual(
            (rows["global_load_dword"]["stall"], rows["global_load_dword"]["wait"]),
            (30, 0),
        )
        self.assertEqual(
            (rows["s_waitcnt"]["stall"], rows["s_waitcnt"]["wait"]), (0, 400)
        )
        line = mine.Capture.from_waves([wave([load, wait])]).hotspots(by_line=True)[0]
        self.assertEqual((line["stall"], line["wait"]), (30, 400))

    def test_stats_csv_splits_wait_from_stall(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "stats_x_dispatch_1.csv").write_text(
                '"CodeObj","Vaddr","Instruction","Hitcount","Latency","Stall","Idle","Source"\n'
                '2,4,"s_waitcnt vmcnt(0)",10,900,900,0,"/a/k.hip:7"\n'
                '2,8,"global_load_dword v1, v[2:3], off",10,60,20,0,"/a/k.hip:7"\n'
            )
            stats = mine.read_stats(d)
        self.assertEqual(
            (stats["lines"][0]["stall"], stats["lines"][0]["wait"]), (20, 900)
        )

    def test_scratch_instructions_are_reported_only_when_present(self):
        plain = mine.Capture.from_waves([wave([inst(0, "v_add_f32 v0, v0, v1")])])
        self.assertNotIn("scratch_instructions", plain.pipes())
        spill = mine.Inst(
            4, 20, 0, 0, "FLAT", "scratch_store_dword off, v1", "", mine.Pc(0x20, 1)
        )
        cap = mine.Capture.from_waves([wave([inst(0, "v_add_f32 v0, v0, v1"), spill])])
        sc = cap.pipes()["scratch_instructions"]
        self.assertEqual((sc["per_wave"], sc["opcodes"]), (1.0, "scratch_store_dword"))
        self.assertEqual(sc["share_of_cost"], round(100 * 20 / 24, 1))


class AlsoReadTests(unittest.TestCase):
    def test_a_large_second_state_adds_its_page(self):
        hint = mine.next_read({"STALL": 50, "WAIT": 30, "EXEC": 20})
        self.assertIn("read " + str(mine.RESOURCES / "compute.md"), hint)
        self.assertIn(
            "Also read "
            + str(mine.RESOURCES / "latency.md")
            + " (WAIT is 30% of wave time)",
            hint,
        )

    def test_small_states_and_the_main_page_are_not_repeated(self):
        self.assertNotIn(
            "Also read", mine.next_read({"STALL": 70, "WAIT": 19, "EXEC": 11})
        )
        self.assertNotIn("Also read", mine.next_read({"WAIT": 80, "EXEC": 20}))

    def test_idle_between_instructions_adds_the_idle_section(self):
        hint = mine.next_read({"STALL": 60, "EXEC": 40}, idle_share=25)
        self.assertIn("Idle cycles", hint)
        self.assertIn("idle between instructions is 25% of wave time", hint)

    def test_few_resident_waves_add_the_few_waves_section(self):
        few = {"mean": 1.8, "peak": 2.0, "max": 8}
        self.assertIn("Few waves", mine.next_read({"STALL": 60, "EXEC": 40}, waves=few))
        many = {"mean": 6.0, "peak": 7.0, "max": 8}
        self.assertNotIn(
            "Few waves", mine.next_read({"STALL": 60, "EXEC": 40}, waves=many)
        )
        self.assertNotIn(
            "Few waves",
            mine.next_read({"STALL": 60, "EXEC": 40}, waves={"peak": 1, "max": None}),
        )

    def test_waves_per_simd_from_occupancy_rows(self):
        rows = [
            {"time": 0, "active_waves": 2},
            {"time": 50, "active_waves": 4},
            {"time": 100, "active_waves": 0},
        ]
        self.assertEqual(mine.mean_and_peak_per_simd(rows, 2), {"mean": 1.5, "peak": 2.0})
        self.assertEqual(mine.mean_and_peak_per_simd(rows[:1], 2), {"mean": 0, "peak": 0})


class InstsAndGuardTests(unittest.TestCase):
    def test_lines_count_the_distinct_instructions_of_each_line(self):
        w = wave(
            [
                inst(0, "v_rcp_f32 v1, v2", addr=0x10, source="/a/k.hip:5"),
                inst(4, "v_fma_f32 v1, v2, v3, v4", addr=0x14, source="/a/k.hip:5"),
                inst(8, "v_mul_f32 v1, v2, v3", addr=0x18, source="/a/k.hip:5"),
                inst(12, "v_rcp_f32 v1, v2", addr=0x10, source="/a/k.hip:5"),
                inst(16, "v_add_f32 v1, v2, v3", addr=0x1C, source="/a/k.hip:6"),
            ]
        )
        rows = {r["line"]: r for r in mine.Capture.from_waves([w]).hotspots(by_line=True)}
        self.assertEqual((rows["k.hip:5"]["insts"], rows["k.hip:6"]["insts"]), (3, 1))

    def test_stats_lines_count_instructions_that_ran(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "stats_x_dispatch_1.csv").write_text(
                '"CodeObj","Vaddr","Instruction","Hitcount","Latency","Stall","Idle","Source"\n'
                '2,4,"v_rcp_f32 v1, v2",10,40,0,0,"/a/k.hip:5"\n'
                '2,8,"v_fma_f32 v1, v2, v3, v4",10,40,0,0,"/a/k.hip:5"\n'
                '2,12,"v_mul_f32 v1, v2, v3",0,0,0,0,"/a/k.hip:5"\n'
            )
            stats = mine.read_stats(d)
        self.assertEqual(stats["lines"][0]["insts"], 2)

    def test_every_next_line_ends_with_the_guard(self):
        self.assertTrue(mine.next_read({"STALL": 60, "EXEC": 40}).endswith(mine.GUARD))
        row = {
            "instructions": [
                {"text": "s_waitcnt vmcnt(0)", "share": 90, "idle": 0, "latency": 9}
            ]
        }
        self.assertTrue(mine.next_read_from_stats(row).endswith(mine.GUARD))


class WaitIsLatencyTests(unittest.TestCase):
    def test_a_barrier_whose_wait_is_in_its_latency_counts_as_wait(self):
        # gfx11 and later: a barrier's wait is in its duration, not its stall
        barrier = mine.Inst(
            0, 500, 0, 0, "MESSAGE", "s_barrier", "/a/k.hip:9", mine.Pc(0x20, 1)
        )
        row = mine.Capture.from_waves([wave([barrier])]).hotspots()[0]
        self.assertEqual((row["stall"], row["wait"]), (0, 500))
        p = mine.Capture.from_waves([wave([barrier])]).pipes()
        b = next(c for c in p["classes"] if c["class"] == "barrier (s_barrier)")
        self.assertEqual((b["stall_share"], b["wait_share"]), (0.0, 100.0))

    def test_the_flat_profile_hint_also_ends_with_the_guard(self):
        rows = {
            "instructions": [
                {"text": "v_add_f32 v1, v2, v3", "share": 2.0, "idle": 0, "latency": 4}
            ]
        }
        self.assertTrue(mine.next_read_from_stats(rows).endswith(mine.GUARD))


class LargestStateWordingTests(unittest.TestCase):
    def test_the_basis_is_named_only_when_moving_idle_changes_the_largest_state(self):
        changed = mine.next_read({"EXEC": 50, "WAIT": 40, "STALL": 10}, idle_share=15)
        self.assertIn(
            "WAIT is the largest wave state (with idle between instructions", changed
        )
        same = mine.next_read({"WAIT": 60, "EXEC": 30, "STALL": 10}, idle_share=5)
        self.assertIn("WAIT is the largest wave state.", same)
        self.assertNotIn("counted apart", same)


class CeilingAlsoReadTests(unittest.TestCase):
    def test_a_busy_vector_unit_adds_the_ceiling_check_whatever_the_largest_state(self):
        hint = mine.next_read(
            {"WAIT": 75, "EXEC": 20, "STALL": 5}, {"valu_busy": 87.2, "classes": []}
        )
        self.assertIn("Ceiling check", hint)
        quiet = mine.next_read(
            {"WAIT": 75, "EXEC": 20, "STALL": 5}, {"valu_busy": 30.0, "classes": []}
        )
        self.assertNotIn("Ceiling check", quiet)


class DispatchKeyTests(unittest.TestCase):
    def test_one_dispatch_on_two_shader_engines_counts_once(self):
        d0 = SimpleNamespace(time=100, thread_dim_x=64, thread_dim_y=1, thread_dim_z=1)
        d1 = SimpleNamespace(time=90, thread_dim_x=64, thread_dim_y=1, thread_dim_z=1)
        w0 = wave([inst(200, "v_add_f32 v0, v0, v1")])
        w1 = wave([inst(210, "v_add_f32 v0, v0, v1")])
        mine.assign_dispatches([w0], [d0], ("53377", "1"))
        mine.assign_dispatches([w1], [d1], ("53377", "1"))
        self.assertEqual(w0.dispatch_key, w1.dispatch_key)

    def test_consecutive_kernels_in_one_file_count_separately(self):
        a = SimpleNamespace(time=100)
        b = SimpleNamespace(time=500)
        w0 = wave([inst(200, "v_add_f32 v0, v0, v1")])
        w1 = wave([inst(600, "v_add_f32 v0, v0, v1")])
        mine.assign_dispatches([w0, w1], [a, b], ("53377", "1"))
        self.assertNotEqual(w0.dispatch_key, w1.dispatch_key)

    def test_a_wave_takes_the_latest_dispatch_of_its_own_queue(self):
        q0 = SimpleNamespace(time=100, me_id=1, pipe_id=0)
        q1 = SimpleNamespace(time=150, me_id=1, pipe_id=2)
        w0 = wave([inst(200, "v_add_f32 v0, v0, v1")])
        w1 = wave([inst(200, "v_add_f32 v0, v0, v1")])
        w0.dispatcher, w1.dispatcher = 1 << 4 | 0, 1 << 4 | 2
        mine.assign_dispatches([w0, w1], [q0, q1], ("1", "1"))
        self.assertIs(w0.dispatch, q0)
        self.assertIs(w1.dispatch, q1)

    def test_a_queue_with_no_record_falls_back_to_the_latest_dispatch(self):
        q0 = SimpleNamespace(time=100, me_id=1, pipe_id=0)
        w = wave([inst(200, "v_add_f32 v0, v0, v1")])
        w.dispatcher = 3 << 4 | 5
        mine.assign_dispatches([w], [q0], ("1", "1"))
        self.assertIs(w.dispatch, q0)


class PageListTests(unittest.TestCase):
    def test_a_page_named_with_a_section_is_read_from_that_section(self):
        anchored = mine.page_lists("compute.md#ceiling-check")
        self.assertTrue(anchored.startswith(", starting there; the page lists"))
        self.assertTrue(mine.page_lists("latency.md").startswith(": it lists"))


@unittest.skipUnless(
    importlib.util.find_spec("rocprof_trace_decoder"),
    "needs the decoder's Python package",
)
class UnresolvedRecordTests(unittest.TestCase):
    """Records with no pc: only those that are not context or cut-off records are unresolved."""

    @staticmethod
    def load(category):
        pc = SimpleNamespace(address=0, code_object_id=0)
        record = SimpleNamespace(
            pc=pc, category=int(category), time=10, duration=4, stall=0
        )
        wave = SimpleNamespace(
            cu=0,
            simd=0,
            wave_id=0,
            workgroup_id=0,
            begin_time=0,
            end_time=100,
            contexts=0,
            dispatcher=0,
            instructions=[record],
            timeline=[],
        )
        cap = object.__new__(mine.Capture)
        cap.code_index = SimpleNamespace(entries={})
        cap.unresolved = 0
        return cap, cap._wave(0, wave)

    def test_a_wave_cut_off_by_the_end_of_the_trace_loads(self):
        cap, trace = self.load(
            14
        )  # the decoder's WAVE_NOT_FINISHED, past InstCategory.LAST
        self.assertEqual(cap.unresolved, 0)
        self.assertEqual(trace.insts, [])

    def test_context_records_are_not_unresolved(self):
        from rocprof_trace_decoder import InstCategory

        cap, trace = self.load(InstCategory.CONTEXT)
        self.assertEqual(cap.unresolved, 0)
        self.assertEqual(trace.insts, [])

    def test_other_records_without_a_pc_are_unresolved(self):
        from rocprof_trace_decoder import InstCategory

        cap, trace = self.load(InstCategory.VMEM)
        self.assertEqual(cap.unresolved, 1)
        self.assertEqual(trace.insts, [])


@unittest.skipUnless(
    importlib.util.find_spec("rocprof_trace_decoder"),
    "needs the decoder's Python package",
)
class ReusedDirectoryTests(unittest.TestCase):
    def test_code_objects_of_two_runs_say_to_use_a_new_directory(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for pid in ("100", "200"):
                (Path(tmp) / f"{pid}_gfx942_code_object_id_1.out").write_bytes(
                    b"\x7fELF" + bytes(60)
                )
            (Path(tmp) / "100_7_shader_engine_0_1.att").write_bytes(b"")
            with self.assertRaises(SystemExit) as caught:
                mine.Capture(tmp)
        message = str(caught.exception)
        self.assertIn("Code object id 1 is used by both", message)
        self.assertIn("new directory", message)


if __name__ == "__main__":
    unittest.main()
