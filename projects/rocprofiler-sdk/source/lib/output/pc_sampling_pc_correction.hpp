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

#pragma once

#include "lib/common/synchronized.hpp"
#include "lib/output/pc_sample_transform.hpp"

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/pc_sampling.h>
#include <rocprofiler-sdk/cxx/codeobj/code_printing.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tool
{
namespace pc_correction
{
// The shared instruction decoder type used by the rocprofv3 tool. Aliased to
// match metadata::code_obj_decoder_t without depending on metadata.hpp.
using code_obj_decoder_t = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

// One decoded instruction, as produced by the code-object decoder.
using Instruction = rocprofiler::sdk::codeobj::disassembly::Instruction;

// Per-symbol kernel/symbol descriptor produced by the decoder (carries name,
// faddr, vaddr, mem_size).
using SymbolInfo = rocprofiler::sdk::codeobj::disassembly::SymbolInfo;

/**
 * @brief Alias for a code-object id.
 *
 * The SDK has no `rocprofiler_code_object_id_t` typedef; a code-object id is a
 * plain uint64_t (see inst_t::code_object_id and rocprofiler_pc_t::code_object_id).
 * This local alias documents intent at every use site.
 */
using rocprofiler_code_object_id_t = uint64_t;

/**
 * @brief Classification of a single decoded instruction within the
 * EXT1 -> INT_CHAIN -> EXT2 framing.
 */
enum class Kind
{
    EXT,               ///< External (non-internal) instruction; bounds a chain.
    REGULAR_INTERNAL,  ///< Internal contributing to the snapshot PC correction factor.
    S_ICACHE_INV       ///< Internal NOT contributing to the correction factor.
};

/**
 * @brief Outcome of applying correction to a sample.
 */
enum class CorrectionResult
{
    Keep,  ///< Sample is healthy or successfully corrected — write it out.
    Drop   ///< Sample is unrecoverable — skip the ring-buffer write entirely.
};

/**
 * @brief Classify a decoded instruction string by its opcode mnemonic prefix.
 *
 * The opcode list starts small and is extended as the authoritative
 * internal-instruction set is confirmed. The safe direction is to err toward
 * EXT: a missed internal means "no correction" for that sample, whereas a
 * mis-classified EXT would corrupt otherwise-healthy samples.
 *
 * @param [in] inst Decoded instruction text (mnemonic + operands).
 * @return The instruction's @ref Kind.
 */
Kind
classify(std::string_view inst);

/**
 * @brief Test whether a gfx_target_version identifies a gfx1250 agent.
 *
 * Encoding (per the SDK agent API): major = (ver / 10000) % 100,
 * minor = (ver / 100) % 100. gfx1250 -> major=12, minor=50 -> [125000, 125100).
 *
 * @param [in] gfx_target_version Packed gfx target version of the agent.
 * @return true if @p gfx_target_version is in the gfx1250 range.
 */
constexpr bool
is_gfx1250(uint32_t gfx_target_version)
{
    return gfx_target_version >= 125000u && gfx_target_version < 125100u;
}

/**
 * @brief One "EXT1 -> INT_CHAIN -> EXT2" window.
 *
 * All internals inside the window share one shared_ptr<const
 * InstructionStreamWindow>. Filled during the code-object walk; immutable once
 * published.
 */
struct InstructionStreamWindow
{
    uint64_t ext1_offset                  = 0;      ///< Opening-EXT offset (valid iff has_ext1).
    uint64_t ext2_offset                  = 0;      ///< Closing-EXT offset (valid iff has_ext2).
    bool     has_ext1                     = false;  ///< Window opened on an external.
    bool     has_ext2                     = false;  ///< Window closed on an external.
    uint64_t regular_internal_total_bytes = 0;
    uint16_t M                            = 0;  ///< Regular-internal count.
    uint16_t N                            = 0;  ///< s_icache_inv count.

    /// @var ext1_offset
    /// @brief Offset of the external that opened this window. Only meaningful
    /// when has_ext1 is true. Offset 0 is a legitimate value (the first
    /// instruction of a symbol), so presence is tracked by has_ext1, not by a
    /// magic-zero sentinel.
    /// @var ext2_offset
    /// @brief Offset of the external that closed this window. Only meaningful
    /// when has_ext2 is true. As with ext1_offset, presence is tracked by the
    /// has_ext2 flag rather than a magic-zero sentinel.
    /// @var regular_internal_total_bytes
    /// @brief Running sum of regular-internal instruction sizes in this window.
    /// Bounds how far the snapshot PC correction factor can subtract back from
    /// EXT2. s_icache_inv does not contribute and is excluded from this sum.
    /// @var M
    /// @brief Regular-internal count. The correction logic uses `M == 0` to
    /// detect an s_icache_inv-only chain (correct backward to EXT1).
    /// @var N
    /// @brief s_icache_inv count. The correction logic uses `N == 0` to detect a
    /// regular-internal-only chain (correct forward to EXT2).
};

/**
 * @brief One entry per internal-like instruction (regular internal or s_icache_inv).
 */
struct InternalEntry
{
    uint64_t                                       offset = 0;  ///< Sort key.
    std::shared_ptr<const InstructionStreamWindow> window;      ///< Owning window.
};

/**
 * @brief Per code object classification.
 *
 * Built in one synchronous pass at code-object load, sorted by offset, then
 * never modified. The classification builds itself: @ref add_symbol walks a
 * single symbol's instruction stream and appends the resulting entries, then
 * @ref sort orders them for binary-search lookup via @ref find.
 */
struct CodeObjectClassification
{
    /// Pulls one instruction at a given load-relative vaddr offset. Returns null
    /// (or an instruction of size 0) to stop the walk. Production code wraps the
    /// real decoder; tests return synthetic instructions (see @ref add_symbol).
    using decode_fn = std::function<std::unique_ptr<Instruction>(uint64_t voffset)>;

    std::vector<InternalEntry> entries;  ///< Sorted by offset (after @ref sort); immutable thereafter.

    /**
     * @brief Walk one symbol's instruction stream in a single pass, appending
     * its internal-like entries to @ref entries.
     *
     * Implements the EXT1 -> INT_CHAIN -> EXT2 state machine for one symbol.
     * All internals between two externals share one InstructionStreamWindow;
     * the closing external back-fills ext2_offset and sets has_ext2. Leading
     * internals (no preceding external) leave has_ext1 false; trailing internals
     * (no following external before symbol end) leave has_ext2 false.
     *
     * Instructions are pulled through @p decode rather than from a decoder
     * reference so this logic is unit-testable without comgr: production code
     * wraps the real decoder, while tests return synthetic @ref Instruction
     * objects. @p decode is a std::function (not a template) so this stays a
     * regular out-of-line method; the type-erasure cost is irrelevant here since
     * the walk runs at code-object load time, not on the sample hot path.
     *
     * @param [in] symbol_vaddr Symbol's load-relative vaddr offset (the same
     *   space as a sample's `pc.code_object_offset`).
     * @param [in] symbol_size  Symbol size in bytes; the walk stops at
     *   symbol_vaddr + symbol_size.
     * @param [in] decode       Instruction source; see @ref decode_fn.
     */
    void add_symbol(uint64_t symbol_vaddr, uint64_t symbol_size, const decode_fn& decode);

    /// Sort @ref entries by offset. Call once after all symbols are added.
    void sort();

    /**
     * @brief Binary-search for the entry at an exact offset.
     *
     * @param [in] offset A sample's `pc.code_object_offset`.
     * @return The matching entry, or std::nullopt if no internal-like
     *   instruction sits at @p offset. Requires @ref sort to have run.
     */
    std::optional<InternalEntry> find(uint64_t offset) const;
};

/**
 * @brief Per-code-object classification map.
 *
 * The inner map is a plain unordered_map; the Synchronized wrapper lives on the
 * manager. The shared_ptr<const ...> value makes the reader path
 * safe-by-construction — entries are published read-only and never mutated
 * afterwards.
 */
using ClassificationMap =
    std::unordered_map<rocprofiler_code_object_id_t,
                       std::shared_ptr<const CodeObjectClassification>>;

/**
 * @brief Owns the per-code-object classification map and exposes the correction API.
 *
 * The manager is intentionally unaware of `metadata`: it is constructed with a
 * reference to the shared instruction decoder and a single @c enabled flag.
 * All enablement policy (env-var toggle, "a relevant agent is being sampled")
 * is folded into that one flag by the wiring code, so the manager never sees
 * config, gfx targets, or metadata.
 */
class PCCorrectionManager
{
public:
    /**
     * @brief Construct the manager.
     *
     * @param [in] decoder Shared instruction decoder, owned elsewhere (e.g. by
     *   metadata). Used only to walk symbols at build time; the manager does
     *   not take ownership and must not outlive @p decoder.
     */
    explicit PCCorrectionManager(common::Synchronized<code_obj_decoder_t, true>& decoder);
    ~PCCorrectionManager();

    PCCorrectionManager(const PCCorrectionManager&)            = delete;
    PCCorrectionManager& operator=(const PCCorrectionManager&) = delete;

    /**
     * @brief Enable or disable correction.
     *
     * Folds every enablement condition into one flag. Read (relaxed) on the
     * hot path by @ref should_correct. Set once during configuration, before
     * any samples are processed.
     */
    void set_enabled(bool enabled) { enabled_.store(enabled, std::memory_order_relaxed); }

    /// @return Whether correction is currently enabled.
    bool enabled() const { return enabled_.load(std::memory_order_relaxed); }

    /**
     * @brief Build the classification for a code object at load time.
     *
     * @param [in] obj_data Code-object load payload (rocprofiler_code_object_info_t
     * in metadata.hpp is an alias for this type).
     */
    void build(const rocprofiler_callback_tracing_code_object_load_data_t& obj_data);

    /**
     * @brief Publish a pre-built classification for a code object.
     *
     * Separated from @ref build so tests can inject a synthetic classification
     * without a comgr-backed decoder. Publishes under a brief wlock; readers
     * observe either no entry or the fully-built one.
     *
     * @param [in] co_id          Code-object id.
     * @param [in] classification Fully built, sorted, immutable classification.
     */
    void publish(rocprofiler_code_object_id_t                    co_id,
                 std::shared_ptr<const CodeObjectClassification> classification);

    /**
     * @brief Erase the classification for a code object at unload time.
     *
     * @param [in] co_id Id of the code object being unloaded.
     */
    void erase(rocprofiler_code_object_id_t co_id);

    /**
     * @brief Look up the internal-like entry at a sample's PC, if any.
     *
     * Pure read path: brief rlock on the map, copy the shared_ptr out, then
     * lock-free binary search. The copy-out keeps the classification alive even
     * if the code object is concurrently unloaded.
     *
     * @param [in] co_id  Code-object id.
     * @param [in] offset Sample's `pc.code_object_offset`.
     * @return The matching entry, or std::nullopt.
     */
    std::optional<InternalEntry> lookup(rocprofiler_code_object_id_t co_id, uint64_t offset) const;

    /**
     * @brief Hot-path gate: decide whether a sample is a correction candidate.
     *
     * @param [in] s            The stochastic sample record.
     * @param [in] decoded_inst Decoded instruction text at the sample's PC.
     * @return true if the sample should be passed to @ref correct.
     */
    bool should_correct(const rocprofiler_pc_sampling_record_stochastic_v0_t& s,
                        std::string_view decoded_inst) const;

    /**
     * @brief Hot-path correction: apply the cascade, mutating @p s in place on Keep.
     *
     * @param [in,out] s The tool sample record; its PC is mutated on a Keep
     * correction.
     * @return ::CorrectionResult indicating whether to keep or drop the sample.
     */
    CorrectionResult correct(rocprofiler_tool_pc_sampling_stochastic_record_t& s) const;

private:
    common::Synchronized<code_obj_decoder_t, true>&       decoder_;
    std::atomic<bool>                                     enabled_{false};
    mutable common::Synchronized<ClassificationMap, true> map_;
};

}  // namespace pc_correction
}  // namespace tool
}  // namespace rocprofiler
