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

#pragma once

#include <cstdint>
#include <map>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "rocprof_trace_decoder/rocprof_trace_decoder.h"
#include "trace_parser.hpp"

/// Cycles hidden by concurrent instruction-pipe activity.
struct HiddenLatency
{
    int64_t idle = 0;
    int64_t stall = 0;
    int64_t issue = 0;

    int64_t total() const { return idle + stall + issue; }
};

/// True for mnemonics the estimator treats as MATRIX rather than VALU.
bool is_matrix_instruction(std::string_view instruction);

/// Estimates how much of each instruction's cost overlapped another instruction pipe.
///
/// Busy intervals are built per (SIMD, pipe) using the priority
/// MATRIX > VALU > VMEM/LDS/FLAT > SMEM/SALU, then every instruction is compared against
/// its own pipe and against the union of the higher-priority pipes.
///
/// Results cannot be produced inside the wave callback: waves are stitched out of order,
/// and other-SIMD records arrive in batches that continue past the last wave. So waves are
/// retained until finalize(), which is also why this is opt-in.
class HiddenLatencyAnalysis
{
public:
    /// Retains a stitched wave. Matrix classification reads the ISA text already cached for
    /// each program counter, so nothing is disassembled here.
    void add_wave(const WaveDataInternal& wave, const class PCTranslator& pctranslator);

    /// Retains other-SIMD vector-memory activity before the caller clears its batch.
    void add_other_simd(const std::vector<att_other_simd_t>& records);

    /// Computes and emits one record per (SIMD, program counter), then releases the retained
    /// waves. Emits nothing when no waves were seen.
    void finalize(rocprof_trace_decoder_trace_callback_t callback, void* cbdata);

private:
    struct Inst
    {
        int64_t time;
        int32_t duration;
        uint32_t stall : 24;
        uint32_t category : 8;
        uint32_t pc_id;
    };

    struct Wave
    {
        int64_t begin_time;
        std::vector<Inst> instructions;
    };

    uint32_t intern_pc(pcinfo_t pc, const class PCTranslator& pctranslator);

    std::unordered_map<pcinfo_t, uint32_t> pc_ids{};
    std::vector<pcinfo_t> pcs{};
    std::vector<bool> pc_is_matrix{};

    // Ordered so results are emitted by ascending SIMD regardless of stitch order.
    std::map<uint8_t, std::vector<Wave>> waves_by_simd{};
    std::vector<att_other_simd_t> other_simd{};
};
