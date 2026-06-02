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

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <optional>
#include <string>

namespace rocprofiler
{
namespace counters
{
/**
 * Internal representation of a hardware counter specified by raw block location.
 * Mirrors the hsa_ven_amd_aqlprofile_event_t layout but lives entirely within
 * the tool layer — not exposed as public API.
 */
struct raw_counter_t
{
    uint32_t block_id;    ///< hsa_ven_amd_aqlprofile_block_name_t numeric value
    uint32_t block_index; ///< Block instance index (0-based)
    uint32_t counter_id;  ///< Hardware event/counter ID within the block
};

/**
 * Strip known aqlprofile enum prefixes from a block name string so that the
 * short form (e.g. "CPC") is passed to the aqlprofile lookup.
 *
 *  "HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC" -> "CPC"
 *  "AQLPROFILE_BLOCK_NAME_CPC"             -> "CPC"
 *  "CPC"                                   -> "CPC"
 */
__attribute__((visibility("default"))) std::string
normalize_block_name(std::string_view input);

/**
 * Parse a raw counter specification string into a raw_counter_t.
 *
 * Accepted formats:
 *   "BLOCK_NAME, block_index, counter_id"   (comma-separated, spaces ignored)
 *   "HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 2"
 *   "AQLPROFILE_BLOCK_NAME_CPC, 0, 2"
 *   "block_id:block_index:counter_id"       (numeric, colon-separated)
 *
 * When the block token is purely numeric the value is used directly as block_id
 * without an agent lookup.  When it is a string the function queries aqlprofile
 * via get_block_id_from_name() to resolve it.
 *
 * @param input     The specification string.
 * @param agent_id  Agent used for the string -> block_id lookup (ignored for
 *                  the all-numeric colon format).
 * @return          Populated raw_counter_t on success, std::nullopt if the
 *                  string cannot be parsed or the block name is unknown.
 */
__attribute__((visibility("default"))) std::optional<raw_counter_t>
parse_raw_counter_string(const std::string& input, rocprofiler_agent_id_t agent_id);

}  // namespace counters
}  // namespace rocprofiler
