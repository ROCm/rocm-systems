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

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/pc_sampling.h>

#include <cstdint>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace tool
{
// Forward declarations — full definitions live in metadata.hpp /
// pc_sample_transform.hpp. Only the names are needed here for the manager API.
struct metadata;
struct rocprofiler_tool_pc_sampling_stochastic_record_t;

namespace pc_correction
{
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
    uint64_t ext1_offset                  = 0;  ///< Opening-EXT offset; 0 = sentinel "no EXT1".
    uint64_t ext2_offset                  = 0;  ///< Closing-EXT offset; 0 = sentinel "no EXT2".
    uint64_t regular_internal_total_bytes = 0;
    uint16_t M                            = 0;  ///< Regular-internal count.
    uint16_t N                            = 0;  ///< s_icache_inv count.

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
 * never modified.
 */
struct CodeObjectClassification
{
    std::vector<InternalEntry> entries;  ///< Sorted by offset; immutable after build.
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
 * This class is the seam used by the rocprofv3 tool; the implementation details
 * (the symbol walk, lookup, gating, and cascade) live as file-local free
 * functions in the .cpp.
 */
class PCCorrectionManager
{
public:
    PCCorrectionManager();
    ~PCCorrectionManager();

    PCCorrectionManager(const PCCorrectionManager&)            = delete;
    PCCorrectionManager& operator=(const PCCorrectionManager&) = delete;

    /**
     * @brief Build the classification for a code object at load time.
     *
     * @param [in] obj_data Code-object load payload (rocprofiler_code_object_info_t
     * in metadata.hpp is an alias for this type).
     */
    void build(const rocprofiler_callback_tracing_code_object_load_data_t& obj_data);

    /**
     * @brief Erase the classification for a code object at unload time.
     *
     * @param [in] co_id Id of the code object being unloaded.
     */
    void erase(rocprofiler_code_object_id_t co_id);

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
    mutable common::Synchronized<ClassificationMap, true> map_;
};

}  // namespace pc_correction
}  // namespace tool
}  // namespace rocprofiler
