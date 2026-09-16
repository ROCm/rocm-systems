// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "hidden_latency.hpp"

#include <algorithm>
#include <array>
#include <utility>

#include "stitch/stitch.hpp"

namespace
{
// An interval is stored as its end and its length, matching the coalescing rule below:
// overlapping intervals within one pipe add their lengths instead of merging to the later
// endpoint, so the end can run past the last instruction that contributed to it.
struct Interval
{
    int64_t end;
    int64_t cycles;
};

using Intervals = std::vector<Interval>;
using RawIntervals = std::vector<std::pair<int64_t, int64_t>>; // (clock, cycles)

/// Prefix sums of coalesced intervals, for O(log n) "cycles busy before this clock".
struct Index
{
    std::vector<int64_t> starts;
    std::vector<int64_t> ends;
    std::vector<int64_t> prefix; // prefix[i] = cycles covered by intervals [0, i)

    bool empty() const { return ends.empty(); }
};

constexpr int64_t MATRIX_VALU_NUMERATOR = 3;
constexpr int64_t MATRIX_VALU_DENOMINATOR = 4;

bool is_vector_memory(uint32_t category)
{
    return category == WaveInstCategory::LDS || category == WaveInstCategory::VMEM ||
           category == WaveInstCategory::FLAT;
}

bool is_scalar(uint32_t category)
{
    return category == WaveInstCategory::SALU || category == WaveInstCategory::SMEM;
}

Intervals coalesce(RawIntervals& raw)
{
    if (raw.empty()) return {};

    std::sort(raw.begin(), raw.end());

    // (start, cycles) while accumulating; converted to (end, cycles) on the way out.
    RawIntervals accumulated;
    accumulated.emplace_back(raw.front().first, 0);
    for (const auto& [clock, cycles] : raw)
    {
        if (clock > accumulated.back().first + accumulated.back().second)
            accumulated.emplace_back(clock, cycles);
        else
            accumulated.back().second += cycles;
    }

    Intervals out;
    out.reserve(accumulated.size());
    for (const auto& [start, cycles] : accumulated) out.push_back({start + cycles, cycles});
    return out;
}

Intervals unite(const Intervals& first, const Intervals& second)
{
    Intervals out;
    out.reserve(first.size() + second.size());

    auto append = [&out](const Interval& interval) {
        const int64_t begin = interval.end - interval.cycles;
        if (interval.end <= begin) return;
        if (out.empty())
        {
            out.push_back(interval);
            return;
        }

        const int64_t out_begin = out.back().end - out.back().cycles;
        const int64_t out_end = out.back().end;
        if (begin > out_end)
        {
            out.push_back(interval);
            return;
        }

        const int64_t merged_end = std::max(out_end, interval.end);
        out.back().end = merged_end;
        out.back().cycles = merged_end - out_begin;
    };

    size_t first_index = 0;
    size_t second_index = 0;
    while (first_index < first.size() && second_index < second.size())
    {
        const int64_t first_begin = first[first_index].end - first[first_index].cycles;
        const int64_t second_begin = second[second_index].end - second[second_index].cycles;
        if (first_begin <= second_begin)
            append(first[first_index++]);
        else
            append(second[second_index++]);
    }
    for (; first_index < first.size(); ++first_index) append(first[first_index]);
    for (; second_index < second.size(); ++second_index) append(second[second_index]);
    return out;
}

Index build_index(const Intervals& intervals)
{
    Index index;
    index.starts.reserve(intervals.size());
    index.ends.reserve(intervals.size());
    index.prefix.reserve(intervals.size() + 1);
    index.prefix.push_back(0);
    for (const auto& interval : intervals)
    {
        index.starts.push_back(interval.end - interval.cycles);
        index.ends.push_back(interval.end);
        index.prefix.push_back(index.prefix.back() + interval.cycles);
    }
    return index;
}

int64_t covered_until(const Index& index, int64_t clock)
{
    const size_t complete =
        static_cast<size_t>(std::upper_bound(index.ends.begin(), index.ends.end(), clock) - index.ends.begin());
    int64_t covered = index.prefix[complete];
    if (complete < index.ends.size() && clock > index.starts[complete])
        covered += std::min(clock, index.ends[complete]) - index.starts[complete];
    return covered;
}

/// Splits the window [last_time, clock + cycles) into idle, stall, and issue parts and
/// returns how much of each overlapped the busy intervals in *index*.
HiddenLatency interval_hidden(const Index& index, int64_t last_time, int64_t clock, int64_t stall, int64_t cycles)
{
    HiddenLatency hidden{};
    if (index.empty()) return hidden;

    int64_t at_clock = 0;
    int64_t at_issue = 0;
    bool have_at_clock = false;
    bool have_at_issue = false;

    if (clock > last_time)
    {
        at_clock = covered_until(index, clock);
        have_at_clock = true;
        hidden.idle = at_clock - covered_until(index, last_time);
    }
    if (stall > 0)
    {
        if (!have_at_clock) at_clock = covered_until(index, clock);
        at_issue = covered_until(index, clock + stall);
        have_at_issue = true;
        hidden.stall = at_issue - at_clock;
    }
    if (cycles > stall)
    {
        if (!have_at_issue) at_issue = covered_until(index, clock + stall);
        hidden.issue = covered_until(index, clock + cycles) - at_issue;
    }
    return hidden;
}

HiddenLatency without_issue(const HiddenLatency& hidden) { return {hidden.idle, hidden.stall, 0}; }

/// Keeps the candidate with more overlap, which is what gives the pipe priority its effect.
HiddenLatency larger(const HiddenLatency& preferred, const HiddenLatency& fallback)
{
    return preferred.total() > fallback.total() ? preferred : fallback;
}
} // namespace

bool is_matrix_instruction(std::string_view instruction)
{
    static constexpr std::array<std::string_view, 4> prefixes{"v_mfma", "v_smfma", "v_wmma", "v_swmma"};
    return std::any_of(prefixes.begin(), prefixes.end(), [instruction](std::string_view prefix) {
        return instruction.rfind(prefix, 0) == 0;
    });
}

uint32_t HiddenLatencyAnalysis::intern_pc(pcinfo_t pc, const PCTranslator& pctranslator)
{
    auto [entry, inserted] = pc_ids.emplace(pc, static_cast<uint32_t>(pcs.size()));
    if (!inserted) return entry->second;

    // The ISA text is already cached for every program counter that stitched, so this only
    // reads what the decoder resolved; it never asks for a new disassembly.
    const auto line = pctranslator.addrmap.find(pc);
    const bool matrix = line != pctranslator.addrmap.end() && line->second && is_matrix_instruction(line->second->line);

    pcs.push_back(pc);
    pc_is_matrix.push_back(matrix);
    return entry->second;
}

void HiddenLatencyAnalysis::add_wave(const WaveDataInternal& wave, const PCTranslator& pctranslator)
{
    Wave copy;
    copy.begin_time = wave.begin_time;
    copy.instructions.reserve(wave.instructions.size());
    for (const auto& inst : wave.instructions)
    {
        Inst retained{};
        retained.time = inst.time;
        retained.cycles = std::max<int32_t>(inst.duration, static_cast<int32_t>(inst.stall));
        retained.stall = inst.stall;
        retained.category = inst.category;
        retained.pc_id = intern_pc(inst.pc, pctranslator);
        copy.instructions.push_back(retained);
    }
    waves_by_simd[wave.simd].push_back(std::move(copy));
}

void HiddenLatencyAnalysis::add_other_simd(const std::vector<att_other_simd_t>& records)
{
    other_simd.insert(other_simd.end(), records.begin(), records.end());
}

void HiddenLatencyAnalysis::finalize(rocprof_trace_decoder_trace_callback_t callback, void* cbdata)
{
    std::vector<rocprofiler_thread_trace_decoder_hidden_latency_t> out;

    for (const auto& [simd, waves] : waves_by_simd)
    {
        // Pass one: per-pipe busy intervals for this SIMD.
        RawIntervals matrix_raw;
        RawIntervals valu_raw;
        RawIntervals vmem_raw;
        RawIntervals scalar_raw;

        for (const auto& wave : waves)
            for (const auto& inst : wave.instructions)
            {
                const int64_t clock = inst.time + inst.stall;
                const int64_t cycles = int64_t{inst.cycles} - inst.stall;
                if (pc_is_matrix[inst.pc_id])
                {
                    matrix_raw.emplace_back(clock, cycles);
                    valu_raw.emplace_back(clock, MATRIX_VALU_NUMERATOR * cycles / MATRIX_VALU_DENOMINATOR);
                }
                else if (inst.category == WaveInstCategory::VALU)
                    valu_raw.emplace_back(clock, cycles);
                else if (is_vector_memory(inst.category))
                    vmem_raw.emplace_back(clock, cycles);
                else if (is_scalar(inst.category))
                    scalar_raw.emplace_back(clock, cycles);
            }

        // Other-SIMD activity is not attributed to a SIMD, so every scope sees all of it.
        for (const auto& record : other_simd)
            if (record.cycles > 0) vmem_raw.emplace_back(record.time, record.cycles);

        const Intervals matrix = coalesce(matrix_raw);
        const Intervals valu = coalesce(valu_raw);
        const Intervals vmem = coalesce(vmem_raw);
        const Intervals scalar = coalesce(scalar_raw);

        const Index matrix_index = build_index(matrix);
        const Index valu_index = build_index(valu);
        const Index vmem_index = build_index(vmem);
        const Index scalar_index = build_index(scalar);
        const Intervals math_union = unite(valu, matrix);
        const Intervals vector_union = unite(math_union, vmem);
        const Index math_index = build_index(math_union);
        const Index vector_index = build_index(vector_union);
        const Index all_index = build_index(unite(vector_union, scalar));

        // Pass two: attribute each instruction's window against those intervals.
        std::vector<HiddenLatency> by_pc(pcs.size());
        std::vector<bool> seen(pcs.size(), false);
        std::vector<uint32_t> touched;

        for (const auto& wave : waves)
        {
            int64_t last_time = wave.begin_time;
            for (const auto& inst : wave.instructions)
            {
                const int64_t clock = inst.time;
                const int64_t stall = inst.stall;
                const int64_t cycles = inst.cycles;
                HiddenLatency hidden{};

                if (pc_is_matrix[inst.pc_id])
                {
                    const HiddenLatency as_valu =
                        without_issue(interval_hidden(valu_index, last_time, clock, stall, cycles));
                    const HiddenLatency as_matrix =
                        without_issue(interval_hidden(matrix_index, last_time, clock, stall, cycles));
                    hidden = larger(as_valu, as_matrix);
                }
                else if (inst.category == WaveInstCategory::VALU)
                {
                    const HiddenLatency own =
                        without_issue(interval_hidden(valu_index, last_time, clock, stall, cycles));
                    hidden = larger(own, interval_hidden(matrix_index, last_time, clock, stall, cycles));
                }
                else if (is_vector_memory(inst.category))
                {
                    const HiddenLatency own =
                        without_issue(interval_hidden(vmem_index, last_time, clock, stall, cycles));
                    hidden = larger(interval_hidden(math_index, last_time, clock, stall, cycles), own);
                }
                else if (is_scalar(inst.category))
                {
                    const HiddenLatency own =
                        without_issue(interval_hidden(scalar_index, last_time, clock, stall, cycles));
                    hidden = larger(interval_hidden(vector_index, last_time, clock, stall, cycles), own);
                }
                else
                {
                    hidden = interval_hidden(all_index, last_time, clock, stall, cycles);
                }

                if (bValid(pcs[inst.pc_id]))
                {
                    if (!seen[inst.pc_id])
                    {
                        seen[inst.pc_id] = true;
                        touched.push_back(inst.pc_id);
                    }
                    by_pc[inst.pc_id].idle += hidden.idle;
                    by_pc[inst.pc_id].stall += hidden.stall;
                    by_pc[inst.pc_id].issue += hidden.issue;
                }
                last_time = clock + cycles;
            }
        }

        std::sort(touched.begin(), touched.end(), [this](uint32_t left, uint32_t right) {
            const pcinfo_t& first = pcs[left];
            const pcinfo_t& second = pcs[right];
            if (first.code_object_id != second.code_object_id) return first.code_object_id < second.code_object_id;
            return first.address < second.address;
        });

        for (uint32_t pc_id : touched)
        {
            rocprofiler_thread_trace_decoder_hidden_latency_t record{};
            record.size = sizeof(record);
            record.pc = pcs[pc_id];
            record.idle = by_pc[pc_id].idle;
            record.stall = by_pc[pc_id].stall;
            record.issue = by_pc[pc_id].issue;
            record.simd = simd;
            out.push_back(record);
        }
    }

    if (!out.empty())
        callback(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_HIDDEN_LATENCY, out.data(), out.size(), cbdata);

    waves_by_simd.clear();
    other_simd.clear();
    pc_ids.clear();
    pcs.clear();
    pc_is_matrix.clear();
}
