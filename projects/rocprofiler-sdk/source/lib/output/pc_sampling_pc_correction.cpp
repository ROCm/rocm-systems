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
//   - PCCorrectionManager::enabled() short-circuits the entire path when
//     correction is disabled (env-var toggle) or no gfx1250 agent is configured.
//   - Classifications are built ONLY for code objects on gfx1250 agents, so a
//     non-gfx1250 sample misses the classification map and passes through.
// A per-device-buffer refactor would remove this overhead but is a separate,
// larger effort and is intentionally out of scope here.

#include "lib/output/pc_sampling_pc_correction.hpp"

#include "lib/common/logging.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <mutex>
#include <utility>

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

// Walk every symbol of a code object and build its (unsorted) classification.
// This is the only comgr-coupled layer: it adapts the real decoder into the
// per-instruction `decode` callback consumed by
// CodeObjectClassification::add_symbol. The window state machine itself lives
// in add_symbol and is exercised directly by the unit tests with synthetic
// instructions (no decoder needed) -- this adapter is intentionally thin.
//
// Returns the built-and-sorted classification, or an empty one if the code
// object has no symbols (a survivable case, e.g. data-only objects).
std::shared_ptr<const CodeObjectClassification>
build_classification(code_obj_decoder_t& decoder, rocprofiler_code_object_id_t co_id)
{
    auto classification = std::make_shared<CodeObjectClassification>();

    // Symbols are keyed by load-relative vaddr offset -- the same space as a
    // sample's pc.code_object_offset and the offset add_symbol records.
    std::map<uint64_t, SymbolInfo> symbols = decoder.getSymbolMap(co_id);

    for(const auto& [_, sym] : symbols)
    {
        classification->add_symbol(sym.vaddr, sym.mem_size, [&](uint64_t voffset) {
            return decoder.get(co_id, voffset);
        });
    }

    classification->sort();
    return classification;
}
}  // namespace

Kind
classify(std::string_view inst)
{
    // Instructions that contribute to the snapshot PC correction factor on gfx1250.
    // s_icache_inv does not — it is handled separately below.
    // Erring toward EXT is safe: a missed entry means "no correction", never corruption.
    // "s_wait" prefix-matches all s_wait* variants. "s_setprio" also matches
    // "s_setprio_inc_wg". "s_sleep" does not match "s_monitor_sleep", hence both entries.
    static constexpr std::array<std::string_view, 13> regular_internals = {
        "s_nop",
        "s_sleep",
        "s_monitor_sleep",
        "s_wait",  // prefix-matches all s_wait* variants
        "s_barrier_wait",
        "s_setprio",
        "s_delay_alu",
        "s_sethalt",
        "s_setkill",
        "s_singleuse_vdst",
        "s_round_mode",
        "s_denorm_mode",
        "s_version",
    };

    if(starts_with(inst, "s_icache_inv")) return Kind::S_ICACHE_INV;

    for(const auto& prefix : regular_internals)
    {
        if(starts_with(inst, prefix)) return Kind::REGULAR_INTERNAL;
    }

    return Kind::EXT;
}

void
CodeObjectClassification::add_symbol(uint64_t         symbol_vaddr,
                                     uint64_t         symbol_size,
                                     const decode_fn& decode)
{
    auto           window = std::make_shared<InstructionStreamWindow>();  // has_ext1=false
    const uint64_t end    = symbol_vaddr + symbol_size;

    for(uint64_t voff = symbol_vaddr; voff < end;)
    {
        std::unique_ptr<Instruction> inst = decode(voff);
        if(!inst || inst->size == 0 || voff + inst->size > end) break;

        switch(classify(inst->inst))
        {
            case Kind::EXT:
                // Close the current window only if it actually opened on an
                // external and captured at least one internal.
                if(window->has_ext1 && (window->M > 0 || window->N > 0))
                {
                    window->ext2_offset = voff;
                    window->has_ext2    = true;
                }
                // Open a fresh window with this external as the new EXT1.
                window              = std::make_shared<InstructionStreamWindow>();
                window->ext1_offset = voff;
                window->has_ext1    = true;
                break;
            case Kind::REGULAR_INTERNAL:
                window->M++;
                window->regular_internal_total_bytes += inst->size;
                entries.push_back({voff, window});
                break;
            case Kind::S_ICACHE_INV:
                window->N++;
                entries.push_back({voff, window});
                break;
        }

        voff += inst->size;
    }
    // End of symbol: a still-open window keeps has_ext2 false (trailing
    // internals). The local `window` goes out of scope; nothing else to do.
}

void
CodeObjectClassification::sort()
{
    std::sort(entries.begin(), entries.end(), [](const InternalEntry& a, const InternalEntry& b) {
        return a.offset < b.offset;
    });
}

std::optional<InternalEntry>
CodeObjectClassification::find(uint64_t offset) const
{
    auto it = std::lower_bound(
        entries.begin(), entries.end(), offset, [](const InternalEntry& e, uint64_t off) {
            return e.offset < off;
        });
    if(it == entries.end() || it->offset != offset) return std::nullopt;
    return *it;
}

PCCorrectionManager::PCCorrectionManager(common::Synchronized<code_obj_decoder_t, true>& decoder)
: decoder_(decoder)
{}

PCCorrectionManager::~PCCorrectionManager() = default;

void
PCCorrectionManager::build(const rocprofiler_callback_tracing_code_object_load_data_t& obj_data)
{
    const auto co_id = obj_data.code_object_id;

    // Decode under the decoder's lock; the symbol walk only reads, but the
    // synced decoder exposes mutation via wlock, so we take the writer lock.
    auto classification = decoder_.wlock(
        [&](auto& decoder) { return build_classification(decoder, co_id); });

    if(classification->entries.empty())
    {
        // No internal-like instructions (or no symbols at all -- e.g. data-only
        // code objects). Nothing to correct; publish nothing. A later sample
        // for this code object will simply miss the map and pass through.
        ROCP_INFO << "PC correction: no entries for code object " << co_id
                  << "; classification not published";
        return;
    }

    publish(co_id, std::move(classification));
}

void
PCCorrectionManager::publish(rocprofiler_code_object_id_t                    co_id,
                             std::shared_ptr<const CodeObjectClassification> classification)
{
    // Publish atomically: readers observe either no entry or the fully-built,
    // sorted, immutable classification.
    map_.wlock([&](auto& m) { m.insert_or_assign(co_id, std::move(classification)); });
}

void
PCCorrectionManager::erase(rocprofiler_code_object_id_t co_id)
{
    // Brief wlock + erase. Any in-flight reader already copied the shared_ptr
    // out under lookup()'s rlock, so the classification stays alive past this
    // erase until that reader releases it.
    map_.wlock([&](auto& m) { m.erase(co_id); });
}

std::optional<InternalEntry>
PCCorrectionManager::lookup(rocprofiler_code_object_id_t co_id, uint64_t offset) const
{
    // Step 1: brief rlock, copy the shared_ptr out. The copy keeps the
    // classification alive even if the code object is concurrently unloaded.
    std::shared_ptr<const CodeObjectClassification> classification;
    map_.rlock([&](const auto& m) {
        auto it = m.find(co_id);
        if(it != m.end()) classification = it->second;
    });
    if(!classification) return std::nullopt;

    // Step 2: lock-free binary search on the immutable, sorted entries vector.
    return classification->find(offset);
}

bool
PCCorrectionManager::should_correct(const rocprofiler_pc_sampling_record_stochastic_v0_t& s,
                                    std::string_view decoded_inst) const
{
    // Single enable gate: folds the env-var toggle and "a gfx1250 agent is being
    // sampled" into one relaxed atomic read (set during configuration). Cheapest
    // check, so it runs first.
    if(!enabled()) return false;

    // Condition 1: the reported PC lands on an internal or s_icache_inv. Uses the
    // instruction string the tool already decoded for this sample -- no extra
    // decoder call and no classification-map lookup on this path.
    if(classify(decoded_inst) == Kind::EXT) return false;

    // Condition 2: the snapshot carries a signal that is impossible for an
    // internal instruction, meaning it leaked from an adjacent external.
    //  - wave_issued: internals are never issued.
    //  - reason_not_issued == ARBITER_NOT_WIN: internals are never arbitrated, so
    //    they cannot lose arbitration. Any other reason (e.g. an
    //    internal-instruction reason) is consistent with a healthy internal --
    //    leave it alone.
    if(s.wave_issued) return true;
    if(s.snapshot.reason_not_issued ==
       ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN)
        return true;

    return false;
}

CorrectionResult
PCCorrectionManager::correct(rocprofiler_tool_pc_sampling_stochastic_record_t& s) const
{
    const uint64_t co_id  = s.pc_sample_record.pc.code_object_id;
    const uint64_t offset = s.pc_sample_record.pc.code_object_offset;

    auto entry = lookup(co_id, offset);
    if(!entry)
    {
        // should_correct already established this PC is on an internal, so a
        // missing classification means the code-object bookkeeping raced or a
        // future change decoupled the two checks. Keeping the sample untouched
        // is safer than dropping one we don't understand. Warn once per process
        // (rate-limited) so a broken invariant is visible without flooding logs.
        static std::once_flag warn_once;
        std::call_once(warn_once, [] {
            ROCP_WARNING << "PC correction: classification missing for a sampled code object; "
                            "sample(s) passed through unchanged";
        });
        return CorrectionResult::Keep;
    }

    const InstructionStreamWindow& w = *entry->window;

    // Boundary samples: a window with no opening external (leading internals) or
    // no closing external (trailing internals) cannot be corrected -- drop.
    // Presence is tracked by the has_ext1/has_ext2 flags, not a zero-offset
    // sentinel, because offset 0 is a valid load-relative offset.
    if(!w.has_ext1 || !w.has_ext2) return CorrectionResult::Drop;

    uint64_t corrected = 0;
    if(w.M == 0)
    {
        // s_icache_inv-only chain: correct backward to the opening external.
        corrected = w.ext1_offset;
    }
    else if(w.N == 0)
    {
        // Regular-internal-only chain: correct forward to the closing external.
        corrected = w.ext2_offset;
    }
    else
    {
        // Mixed chain. Below the trailing-M boundary the reported PC is only
        // reachable by the s_icache_inv mechanism -> correct backward to EXT1.
        // At or above the boundary both mechanisms can produce it -> ambiguous,
        // so drop.
        const uint64_t trailing_m_boundary = w.ext2_offset - w.regular_internal_total_bytes;
        if(offset < trailing_m_boundary)
            corrected = w.ext1_offset;
        else
            return CorrectionResult::Drop;
    }

    // Stash originals for optional debug retention (not serialized in v1), then
    // mutate in place. The caller re-resolves inst_index from the corrected PC.
    s.original_pc_offset                     = offset;
    s.original_inst_index                    = s.inst_index;
    s.pc_sample_record.pc.code_object_offset = corrected;
    return CorrectionResult::Keep;
}

}  // namespace pc_correction
}  // namespace tool
}  // namespace rocprofiler
