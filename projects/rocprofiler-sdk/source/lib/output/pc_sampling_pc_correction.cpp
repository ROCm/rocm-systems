// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

// PC sampling reported-PC correction (gfx1250 stochastic sampling only).
//
// On gfx1250, stochastic PC samples whose snapshot lands on or near a run of
// internal-like SALU instructions can report a silently-wrong PC: the reported
// PC lands inside the run of internals, while the accompanying snapshot fields
// (issue/arbitration signals) describe an adjacent external instruction. This
// module reconstructs the intended PC from the surrounding instruction stream.
//
// Heterogeneous-system caveat: rocprofv3 uses a single PC sampling
// buffer/callback shared across all devices, with no convenient per-sample
// "which agent emitted this" field. On a mixed system (e.g. gfx950 + gfx1250),
// every sample therefore runs through the gate. The cost is bounded:
//   - A process-wide flag (any_gfx1250_agent_pc_sampled) short-circuits the
//     entire path when no gfx1250 agent is being sampled.
//   - Classifications are built ONLY for code objects on gfx1250 agents, so a
//     non-gfx1250 sample misses the classification map and passes through.
// A per-device-buffer refactor would remove this overhead but is a separate,
// larger effort and is intentionally out of scope here.

#include "lib/output/pc_sampling_pc_correction.hpp"

#include <array>

namespace rocprofiler
{
namespace tool
{
namespace pc_correction
{
namespace
{
// C++17 has no std::string_view::starts_with (that is C++20). This is the
// project standard (see cmake/rocprofiler_options.cmake), so we provide a local
// equivalent.
bool
starts_with(std::string_view str, std::string_view prefix)
{
    return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}
}  // namespace

Kind
classify(std::string_view inst)
{
    // Start-small authoritative-on-gfx1250 list. Regular internals contribute to
    // the snapshot PC correction factor; s_icache_inv does not. Extend the
    // regular-internal list as the hardware-confirmed set grows (candidates:
    // s_setprio, s_sendmsg, s_trap, s_clause). Erring toward EXT is safe: a
    // missed internal means "no correction", never a corrupted healthy sample.
    static constexpr std::array<std::string_view, 4> regular_internals = {
        "s_nop",
        "s_sleep",
        "s_wait",  // prefix-matches all s_wait* variants
        "s_barrier_wait",
    };

    if(starts_with(inst, "s_icache_inv")) return Kind::S_ICACHE_INV;

    for(const auto& prefix : regular_internals)
    {
        if(starts_with(inst, prefix)) return Kind::REGULAR_INTERNAL;
    }

    return Kind::EXT;
}

PCCorrectionManager::PCCorrectionManager()  = default;
PCCorrectionManager::~PCCorrectionManager() = default;

void
PCCorrectionManager::build(const rocprofiler_callback_tracing_code_object_load_data_t& obj_data)
{
    // TODO: single-shot pass over the code object's symbol map; build, sort by
    // offset, and publish a shared_ptr<const CodeObjectClassification>.
    (void) obj_data;
}

void
PCCorrectionManager::erase(rocprofiler_code_object_id_t co_id)
{
    // TODO: brief wlock + erase. In-flight readers stay safe via the
    // shared_ptr<const ...> copy-out in the lookup path.
    (void) co_id;
}

bool
PCCorrectionManager::should_correct(const rocprofiler_pc_sampling_record_stochastic_v0_t& s,
                                    std::string_view decoded_inst) const
{
    // TODO: hot-path gate (env-var flag, gfx1250 flag, PC-on-internal check,
    // impossible-signal check).
    (void) s;
    (void) decoded_inst;
    return false;
}

CorrectionResult
PCCorrectionManager::correct(rocprofiler_tool_pc_sampling_stochastic_record_t& s) const
{
    // TODO: lookup + cascade; mutate s in place and return Keep, or return Drop
    // for boundary / ambiguous samples.
    (void) s;
    return CorrectionResult::Keep;
}

}  // namespace pc_correction
}  // namespace tool
}  // namespace rocprofiler
