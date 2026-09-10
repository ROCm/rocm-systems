#!/usr/bin/env python3

from __future__ import annotations

import random
import unittest

from rocprof_trace_decoder import (
    CodeEntry,
    CodeIndex,
    HiddenLatency,
    InstCategory,
    Instruction,
    OtherSimdInstruction,
    Pc,
    TraceRecords,
    Wave,
    analyze_hidden_latency,
)
from rocprof_trace_decoder.analysis.hidden_latency import (
    _build_pipe_utilization,
    _compute_union,
    _compute_util,
    _index_util,
    _indexed_interval,
    _matrix_instruction_pcs,
)


def _instruction(
    pc: int,
    category: InstCategory,
    *,
    time: int,
    duration: int,
    stall: int = 0,
) -> Instruction:
    return Instruction(
        category=int(category),
        stall=stall,
        duration=duration,
        time=time,
        pc=Pc(pc, 1),
    )


def _wave(*instructions: Instruction, simd: int = 0, begin: int = 0) -> Wave:
    end = max((inst.time + inst.duration for inst in instructions), default=begin)
    return Wave(
        cu=1,
        simd=simd,
        wave_id=0,
        contexts=0,
        dispatcher=0,
        workgroup_id=0,
        cluster_id=0,
        reserved=0,
        size=0,
        begin_time=begin,
        end_time=end,
        timeline=[],
        instructions=list(instructions),
    )


def _reference_intersection(
    intervals: list[tuple[int, int]],
    clock: int,
    cycles: int,
) -> int:
    if cycles <= 0:
        return 0
    interval_end = clock + cycles
    return sum(
        max(min(end, interval_end) - max(end - duration, clock), 0)
        for end, duration in intervals
    )


def _reference_interval(
    intervals: list[tuple[int, int]],
    last_time: int,
    clock: int,
    stall: int,
    cycles: int,
) -> HiddenLatency:
    return HiddenLatency(
        idle=_reference_intersection(intervals, last_time, clock - last_time),
        stall=_reference_intersection(intervals, clock, stall),
        issue=_reference_intersection(intervals, clock + stall, cycles - stall),
    )


class IntervalCalculationTest(unittest.TestCase):
    """Known-answer interval and utilization tests."""

    def test_compute_util_adds_overlapping_durations(self):
        self.assertEqual(
            _compute_util([(0, 10), (2, 3), (20, 2)]),
            [(13, 13), (22, 2)],
        )

    def test_reference_intersection_oracle(self):
        # Pins the brute-force helper that test_indexed_interval_matches_reference compares
        # against, so a wrong oracle cannot make that comparison vacuous.
        self.assertEqual(_reference_intersection([(10, 10)], 5, 10), 5)
        self.assertEqual(
            _reference_intersection([(5, 5), (15, 5)], 3, 10),
            5,
        )
        self.assertEqual(_reference_intersection([(10, 10)], 11, 3), 0)
        self.assertEqual(_reference_intersection([(10, 10)], 5, 0), 0)

    def test_compute_union(self):
        self.assertEqual(
            _compute_union([(10, 10)], [(15, 10), (30, 5)]),
            [(15, 15), (30, 5)],
        )

    def test_compute_interval(self):
        intervals = [(20, 20)]
        expected = HiddenLatency(idle=5, stall=5, issue=5)
        self.assertEqual(
            _reference_interval(intervals, 0, 5, 5, 10),
            expected,
        )
        self.assertEqual(
            _indexed_interval(_index_util(intervals), 0, 5, 5, 10),
            (expected.idle, expected.stall, expected.issue),
        )

    def test_indexed_interval_matches_reference(self):
        randomizer = random.Random(942)
        for _ in range(256):
            intervals = _compute_util(
                [
                    (randomizer.randrange(0, 200), randomizer.randrange(0, 32))
                    for _ in range(randomizer.randrange(1, 20))
                ]
            )
            last_time = randomizer.randrange(0, 100)
            clock = randomizer.randrange(last_time, 201)
            cycles = randomizer.randrange(1, 32)
            stall = randomizer.randrange(0, cycles + 1)
            expected = _reference_interval(intervals, last_time, clock, stall, cycles)
            self.assertEqual(
                _indexed_interval(
                    _index_util(intervals),
                    last_time,
                    clock,
                    stall,
                    cycles,
                ),
                (expected.idle, expected.stall, expected.issue),
            )


class HiddenLatencyAnalysisTest(unittest.TestCase):
    def test_same_pipe_hides_idle_and_stall_but_not_valu_issue(self):
        target = _instruction(1, InstCategory.VALU, time=10, stall=4, duration=10)
        busy = _instruction(2, InstCategory.VALU, time=0, duration=14)
        result = analyze_hidden_latency({0: TraceRecords(waves=[_wave(target), _wave(busy)])})
        self.assertEqual(result.by_pc[target.pc], HiddenLatency(idle=10, stall=4, issue=0))

    def test_same_pipe_issue_is_not_hidden(self):
        for category in (
            InstCategory.VMEM,
            InstCategory.LDS,
            InstCategory.FLAT,
            InstCategory.SALU,
            InstCategory.SMEM,
        ):
            with self.subTest(category=category):
                target = _instruction(1, category, time=0, duration=6)
                busy = _instruction(2, category, time=0, duration=6)
                result = analyze_hidden_latency(
                    {0: TraceRecords(waves=[_wave(target), _wave(busy)])}
                )
                self.assertEqual(result.by_pc[target.pc], HiddenLatency())

    def test_higher_pipe_hides_vmem_issue(self):
        target = _instruction(1, InstCategory.VMEM, time=10, stall=4, duration=10)
        busy = _instruction(2, InstCategory.VALU, time=14, duration=6)
        result = analyze_hidden_latency({0: TraceRecords(waves=[_wave(target), _wave(busy)])})
        self.assertEqual(result.by_pc[target.pc], HiddenLatency(issue=6))

    def test_vector_pipe_hides_scalar_issue(self):
        target = _instruction(1, InstCategory.SALU, time=10, stall=4, duration=10)
        for category in (InstCategory.VMEM, InstCategory.LDS, InstCategory.FLAT):
            with self.subTest(category=category):
                busy = _instruction(2, category, time=14, duration=6)
                result = analyze_hidden_latency(
                    {0: TraceRecords(waves=[_wave(target), _wave(busy)])}
                )
                self.assertEqual(result.by_pc[target.pc], HiddenLatency(issue=6))

    def test_any_busy_pipe_hides_other_issue(self):
        for category in (
            InstCategory.NONE,
            InstCategory.JUMP,
            InstCategory.NEXT,
            InstCategory.IMMED,
            InstCategory.CONTEXT,
            InstCategory.MESSAGE,
            InstCategory.BVH,
        ):
            with self.subTest(category=category):
                target = _instruction(1, category, time=10, stall=4, duration=10)
                busy = _instruction(2, InstCategory.SMEM, time=14, duration=6)
                result = analyze_hidden_latency(
                    {0: TraceRecords(waves=[_wave(target), _wave(busy)])}
                )
                self.assertEqual(result.by_pc[target.pc], HiddenLatency(issue=6))

    def test_matrix_override_uses_wmma_priority(self):
        matrix = _instruction(1, InstCategory.VALU, time=0, duration=8)
        scalar = _instruction(2, InstCategory.SALU, time=0, duration=8)
        code_index = CodeIndex(
            [
                CodeEntry(matrix.pc, "v_mfma_f32_16x16x16f16 v[0:3], v0, v1, v[0:3]", 1),
                CodeEntry(scalar.pc, "s_add_u32 s0, s0, 1", 2),
            ]
        )
        result = analyze_hidden_latency(
            {0: TraceRecords(waves=[_wave(matrix), _wave(scalar)])},
            code_index=code_index,
        )
        self.assertEqual(result.by_pc[scalar.pc], HiddenLatency(issue=8))

    def test_matrix_contributes_three_quarters_to_valu_utilization(self):
        matrix = _instruction(1, InstCategory.VALU, time=0, duration=8)
        code_index = CodeIndex(
            [CodeEntry(matrix.pc, "v_wmma_f32_16x16x16_f16 v0, v1, v2", 1)]
        )
        wmma, valu, _vmem, _scal = _build_pipe_utilization(
            [_wave(matrix)],
            [],
            _matrix_instruction_pcs(code_index),
        )
        self.assertEqual(wmma, [(8, 8)])
        self.assertEqual(valu, [(6, 6)])

    def test_all_matrix_prefixes_are_recognized(self):
        prefixes = ("v_mfma", "v_smfma", "v_wmma", "v_swmma")
        entries = [
            CodeEntry(Pc(index, 1), f"{prefix}_test", index)
            for index, prefix in enumerate(prefixes, 1)
        ]
        self.assertEqual(
            _matrix_instruction_pcs(CodeIndex(entries)),
            {entry.pc for entry in entries},
        )

    def test_results_are_scoped_by_shader_engine_and_simd(self):
        first = _instruction(1, InstCategory.IMMED, time=0, duration=4)
        second = _instruction(1, InstCategory.IMMED, time=0, duration=4)
        records = {
            2: TraceRecords(waves=[_wave(first, simd=1)]),
            3: TraceRecords(waves=[_wave(second, simd=2)]),
        }
        result = analyze_hidden_latency(records)
        self.assertEqual(set(result.by_scope), {(2, 1), (3, 2)})
        self.assertIn(first.pc, result.by_pc)

    def test_busy_pipe_on_another_simd_does_not_hide_issue(self):
        target = _instruction(1, InstCategory.IMMED, time=0, duration=4)
        busy = _instruction(2, InstCategory.VALU, time=0, duration=4)
        result = analyze_hidden_latency(
            {0: TraceRecords(waves=[_wave(target, simd=0), _wave(busy, simd=1)])}
        )
        self.assertEqual(result.by_pc[target.pc], HiddenLatency())

    def test_other_simd_memory_activity_hides_other_issue(self):
        target = _instruction(1, InstCategory.IMMED, time=0, duration=4)
        other = OtherSimdInstruction(
            size=0,
            time=0,
            cycles=4,
            wgp=1,
            category=int(InstCategory.VMEM),
        )
        result = analyze_hidden_latency(
            {0: TraceRecords(waves=[_wave(target)], other_simd=[other])}
        )
        self.assertEqual(result.by_pc[target.pc], HiddenLatency(issue=4))

    def test_nonpositive_other_simd_activity_is_ignored(self):
        for cycles in (0, -1):
            with self.subTest(cycles=cycles):
                target = _instruction(1, InstCategory.IMMED, time=0, duration=4)
                other = OtherSimdInstruction(
                    size=0,
                    time=0,
                    cycles=cycles,
                    wgp=1,
                    category=int(InstCategory.VMEM),
                )
                result = analyze_hidden_latency(
                    {0: TraceRecords(waves=[_wave(target)], other_simd=[other])}
                )
                self.assertEqual(result.by_pc[target.pc], HiddenLatency())

    def test_same_pc_is_accumulated_across_scopes(self):
        target_pc = Pc(1, 1)
        records = {}
        for shader_engine in (0, 1):
            target = Instruction(
                category=int(InstCategory.IMMED),
                stall=0,
                duration=4,
                time=0,
                pc=target_pc,
            )
            busy = _instruction(2, InstCategory.VALU, time=0, duration=4)
            records[shader_engine] = TraceRecords(waves=[_wave(target), _wave(busy)])

        result = analyze_hidden_latency(records)
        self.assertEqual(result.by_scope[0, 0][target_pc], HiddenLatency(issue=4))
        self.assertEqual(result.by_scope[1, 0][target_pc], HiddenLatency(issue=4))
        self.assertEqual(result.by_pc[target_pc], HiddenLatency(issue=8))

    def test_overlapping_instructions_use_previous_instruction_end(self):
        first = _instruction(1, InstCategory.IMMED, time=0, duration=10)
        overlapping = _instruction(2, InstCategory.IMMED, time=2, duration=2)
        target = _instruction(3, InstCategory.IMMED, time=8, duration=2)
        busy = _instruction(4, InstCategory.VALU, time=4, duration=4)
        result = analyze_hidden_latency(
            {
                0: TraceRecords(
                    waves=[_wave(first, overlapping, target), _wave(busy)]
                )
            }
        )
        self.assertEqual(result.by_pc[target.pc], HiddenLatency(idle=4))

    def test_unresolved_pc_is_not_aggregated(self):
        unresolved = Instruction(
            category=int(InstCategory.IMMED),
            stall=0,
            duration=4,
            time=0,
            pc=Pc(0, 0),
        )
        result = analyze_hidden_latency({0: TraceRecords(waves=[_wave(unresolved)])})
        self.assertNotIn(unresolved.pc, result.by_pc)


if __name__ == "__main__":
    unittest.main()
