from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

from ..code_index import CodeIndex
from ..records import InstCategory, OtherSimdInstruction, Pc, TraceRecords, Wave

__all__ = [
    "HiddenLatency",
    "HiddenLatencyResult",
    "analyze_hidden_latency",
]


@dataclass
class HiddenLatency:
    """Cycles hidden by concurrent instruction-pipe activity."""

    idle: int = 0
    stall: int = 0
    issue: int = 0

    def total(self) -> int:
        return self.idle + self.stall + self.issue

    def __iadd__(self, other: HiddenLatency) -> HiddenLatency:
        self.idle += other.idle
        self.stall += other.stall
        self.issue += other.issue
        return self


@dataclass
class HiddenLatencyResult:
    """Hidden latency aggregated globally and by ``(shader engine, SIMD)``."""

    by_pc: dict[Pc, HiddenLatency]
    by_scope: dict[tuple[int, int], dict[Pc, HiddenLatency]]


_MATRIX_PREFIXES = ("v_mfma", "v_smfma", "v_wmma", "v_swmma")
_UNRESOLVED_PC = Pc(0, 0)
_VALU = int(InstCategory.VALU)
_VECTOR_MEMORY = (
    int(InstCategory.LDS),
    int(InstCategory.VMEM),
    int(InstCategory.FLAT),
)
_SCALAR = (int(InstCategory.SALU), int(InstCategory.SMEM))


@dataclass(frozen=True)
class _UtilIndex:
    starts: list[int]
    ends: list[int]
    prefix_cycles: list[int]


def _compute_util(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coalesce pipe-utilization intervals."""
    if not intervals:
        return []

    ordered = sorted(intervals)
    out = [[ordered[0][0], 0]]
    for clock, cycles in ordered:
        current = out[-1]
        if clock > current[0] + current[1]:
            out.append([clock, cycles])
        else:
            # Overlapping or touching intervals contribute their full durations.
            current[1] += cycles

    # Store the interval end in place of its start after coalescing.
    return [(clock + cycles, cycles) for clock, cycles in out]


def _compute_union(
    first: list[tuple[int, int]],
    second: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    out: list[list[int]] = []

    def append(interval: tuple[int, int]) -> None:
        end, cycles = interval
        begin = end - cycles
        if end <= begin:
            return
        if not out:
            out.append([end, cycles])
            return

        out_begin = out[-1][0] - out[-1][1]
        out_end = out[-1][0]
        if begin > out_end:
            out.append([end, cycles])
        else:
            merged_end = max(out_end, end)
            out[-1][0] = merged_end
            out[-1][1] = merged_end - out_begin

    first_index = 0
    second_index = 0
    while first_index < len(first) and second_index < len(second):
        first_begin = first[first_index][0] - first[first_index][1]
        second_begin = second[second_index][0] - second[second_index][1]
        if first_begin <= second_begin:
            append(first[first_index])
            first_index += 1
        else:
            append(second[second_index])
            second_index += 1
    for interval in first[first_index:]:
        append(interval)
    for interval in second[second_index:]:
        append(interval)
    return [(end, cycles) for end, cycles in out]


def _matrix_instruction_pcs(code_index: CodeIndex | None) -> set[Pc]:
    if code_index is None:
        return set()
    return {
        pc
        for pc, entry in code_index.entries.items()
        if entry.inst.startswith(_MATRIX_PREFIXES)
    }


def _add_hidden(target: dict[Pc, HiddenLatency], pc: Pc, value: HiddenLatency) -> None:
    current = target.get(pc)
    if current is None:
        target[pc] = HiddenLatency(value.idle, value.stall, value.issue)
    else:
        current += value


def _index_util(intervals: list[tuple[int, int]]) -> _UtilIndex:
    starts = []
    ends = []
    prefix_cycles = [0]
    for end, cycles in intervals:
        starts.append(end - cycles)
        ends.append(end)
        prefix_cycles.append(prefix_cycles[-1] + cycles)
    return _UtilIndex(starts, ends, prefix_cycles)


def _covered_until(index: _UtilIndex, clock: int) -> int:
    complete = bisect_right(index.ends, clock)
    covered = index.prefix_cycles[complete]
    if complete < len(index.ends) and clock > index.starts[complete]:
        covered += min(clock, index.ends[complete]) - index.starts[complete]
    return covered


def _indexed_interval(
    index: _UtilIndex,
    last_time: int,
    clock: int,
    stall: int,
    cycles: int,
) -> tuple[int, int, int]:
    if not index.ends:
        return 0, 0, 0

    at_clock = None
    at_issue = None
    hidden_idle = 0
    hidden_stall = 0
    hidden_issue = 0

    if clock > last_time:
        at_clock = _covered_until(index, clock)
        hidden_idle = at_clock - _covered_until(index, last_time)
    if stall > 0:
        if at_clock is None:
            at_clock = _covered_until(index, clock)
        at_issue = _covered_until(index, clock + stall)
        hidden_stall = at_issue - at_clock
    if cycles > stall:
        if at_issue is None:
            at_issue = _covered_until(index, clock + stall)
        hidden_issue = _covered_until(index, clock + cycles) - at_issue

    return hidden_idle, hidden_stall, hidden_issue


def _build_pipe_utilization(
    waves: list[Wave],
    other_simd: list[OtherSimdInstruction],
    matrix_pcs: set[Pc],
) -> tuple[
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
    list[tuple[int, int]],
]:
    wmma: list[tuple[int, int]] = []
    valu: list[tuple[int, int]] = []
    vmem: list[tuple[int, int]] = []
    scal: list[tuple[int, int]] = []

    for wave in waves:
        for instruction in wave.instructions:
            clock = instruction.time + instruction.stall
            cycles = instruction.duration - instruction.stall
            category = int(instruction.category)
            if instruction.pc in matrix_pcs:
                wmma.append((clock, cycles))
                valu.append((clock, 3 * cycles // 4))
            elif category == _VALU:
                valu.append((clock, cycles))
            elif category in _VECTOR_MEMORY:
                vmem.append((clock, cycles))
            elif category in _SCALAR:
                scal.append((clock, cycles))

    for record in other_simd:
        if record.cycles > 0:
            vmem.append((record.time, record.cycles))

    wmma = _compute_util(wmma)
    valu = _compute_util(valu)
    vmem = _compute_util(vmem)
    scal = _compute_util(scal)
    return wmma, valu, vmem, scal


def _analyze_scope(
    waves: list[Wave],
    other_simd: list[OtherSimdInstruction],
    matrix_pcs: set[Pc],
) -> dict[Pc, HiddenLatency]:
    wmma, valu, vmem, scal = _build_pipe_utilization(waves, other_simd, matrix_pcs)

    math_union = _compute_union(valu, wmma)
    vector_union = _compute_union(math_union, vmem)
    all_union = _compute_union(vector_union, scal)
    wmma_index = _index_util(wmma)
    valu_index = _index_util(valu)
    vmem_index = _index_util(vmem)
    scal_index = _index_util(scal)
    math_index = _index_util(math_union)
    vector_index = _index_util(vector_union)
    all_index = _index_util(all_union)

    by_pc: dict[Pc, list[int]] = {}
    for wave in waves:
        last_time = wave.begin_time
        for instruction in wave.instructions:
            category = int(instruction.category)
            clock = instruction.time
            stall = instruction.stall
            cycles = instruction.duration

            if instruction.pc in matrix_pcs:
                valu_hidden = _indexed_interval(valu_index, last_time, clock, stall, cycles)
                wmma_hidden = _indexed_interval(wmma_index, last_time, clock, stall, cycles)
                valu_hidden = (valu_hidden[0], valu_hidden[1], 0)
                wmma_hidden = (wmma_hidden[0], wmma_hidden[1], 0)
                hidden = (
                    valu_hidden
                    if valu_hidden[0] + valu_hidden[1] > wmma_hidden[0] + wmma_hidden[1]
                    else wmma_hidden
                )
            elif category == _VALU:
                valu_hidden = _indexed_interval(valu_index, last_time, clock, stall, cycles)
                valu_hidden = (valu_hidden[0], valu_hidden[1], 0)
                wmma_hidden = _indexed_interval(wmma_index, last_time, clock, stall, cycles)
                hidden = (
                    valu_hidden
                    if valu_hidden[0] + valu_hidden[1]
                    > wmma_hidden[0] + wmma_hidden[1] + wmma_hidden[2]
                    else wmma_hidden
                )
            elif category in _VECTOR_MEMORY:
                math_hidden = _indexed_interval(math_index, last_time, clock, stall, cycles)
                vmem_hidden = _indexed_interval(vmem_index, last_time, clock, stall, cycles)
                vmem_hidden = (vmem_hidden[0], vmem_hidden[1], 0)
                hidden = (
                    math_hidden
                    if math_hidden[0] + math_hidden[1] + math_hidden[2]
                    > vmem_hidden[0] + vmem_hidden[1]
                    else vmem_hidden
                )
            elif category in _SCALAR:
                vector_hidden = _indexed_interval(
                    vector_index, last_time, clock, stall, cycles
                )
                scal_hidden = _indexed_interval(scal_index, last_time, clock, stall, cycles)
                scal_hidden = (scal_hidden[0], scal_hidden[1], 0)
                hidden = (
                    vector_hidden
                    if vector_hidden[0] + vector_hidden[1] + vector_hidden[2]
                    > scal_hidden[0] + scal_hidden[1]
                    else scal_hidden
                )
            else:
                hidden = _indexed_interval(all_index, last_time, clock, stall, cycles)

            if instruction.pc != _UNRESOLVED_PC:
                current = by_pc.get(instruction.pc)
                if current is None:
                    by_pc[instruction.pc] = [hidden[0], hidden[1], hidden[2]]
                else:
                    current[0] += hidden[0]
                    current[1] += hidden[1]
                    current[2] += hidden[2]
            last_time = clock + cycles

    return {pc: HiddenLatency(*hidden) for pc, hidden in by_pc.items()}


def analyze_hidden_latency(
    records_by_se: Mapping[int, TraceRecords],
    *,
    code_index: CodeIndex | None = None,
) -> HiddenLatencyResult:
    """Estimate latency hidden by concurrent instruction-pipe activity.

    Analysis is scoped to each ``(shader engine, SIMD)`` and then aggregated by
    program counter.
    """
    by_scope: dict[tuple[int, int], dict[Pc, HiddenLatency]] = {}
    by_pc: dict[Pc, HiddenLatency] = {}
    matrix_pcs = _matrix_instruction_pcs(code_index)

    for shader_engine, records in records_by_se.items():
        waves_by_simd: dict[int, list[Wave]] = defaultdict(list)
        for wave in records.waves:
            waves_by_simd[wave.simd].append(wave)
        for simd, waves in sorted(waves_by_simd.items()):
            scope = (int(shader_engine), int(simd))
            scope_result = _analyze_scope(waves, records.other_simd, matrix_pcs)
            by_scope[scope] = scope_result
            for pc, hidden in scope_result.items():
                _add_hidden(by_pc, pc, hidden)

    return HiddenLatencyResult(by_pc=by_pc, by_scope=by_scope)
