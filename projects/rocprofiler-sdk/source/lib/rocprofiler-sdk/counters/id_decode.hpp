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

#include "lib/common/logging.hpp"

#include <limits>
#include <string_view>
#include <unordered_map>

namespace rocprofiler
{
namespace counters
{
constexpr uint64_t COUNTER_BIT_LENGTH = 16;
constexpr uint64_t DIM_BIT_LENGTH     = 48;
constexpr uint64_t MAX_64             = std::numeric_limits<uint64_t>::max();
constexpr uint64_t BITS_IN_UINT64     = std::numeric_limits<uint64_t>::digits;
enum rocprofiler_profile_counter_instance_types
{
    ROCPROFILER_DIMENSION_NONE = 0,       ///< No dimension data, returns/sets 48 bit value as is
    ROCPROFILER_DIMENSION_XCC,            ///< XCC dimension of result
    ROCPROFILER_DIMENSION_AID,            ///< AID dimension of result
    ROCPROFILER_DIMENSION_SHADER_ENGINE,  ///< SE dimension of result
    ROCPROFILER_DIMENSION_AGENT,  ///< Agent dimension (internal use only - do not set externally)
    ROCPROFILER_DIMENSION_SHADER_ARRAY,  ///< Number of shader arrays
    ROCPROFILER_DIMENSION_WGP,           ///< Number of workgroup processors
    ROCPROFILER_DIMENSION_INSTANCE,      ///< Number of instances
    ROCPROFILER_DIMENSION_LAST
};

using DimensionMap =
    std::unordered_map<rocprofiler_profile_counter_instance_types, std::string_view>;

const DimensionMap&
dimension_map();

inline rocprofiler_counter_id_t
rec_to_counter_id(rocprofiler_counter_instance_id_t id);
inline void
set_dim_in_rec(rocprofiler_counter_instance_id_t&         id,
               rocprofiler_profile_counter_instance_types dim,
               size_t                                     value);
inline void
set_counter_in_rec(rocprofiler_counter_instance_id_t& id, rocprofiler_counter_id_t value);

inline size_t
rec_to_dim_pos(rocprofiler_counter_instance_id_t          id,
               rocprofiler_profile_counter_instance_types dim);

// Counter ID encoding/decoding functions for agent-specific counter IDs
void
set_agent_in_counter_id(rocprofiler_counter_id_t& id, uint8_t agent_logical_node_id);
uint8_t
get_agent_from_counter_id(rocprofiler_counter_id_t id);
void
set_base_metric_in_counter_id(rocprofiler_counter_id_t& id, uint16_t metric_id);
uint16_t
get_base_metric_from_counter_id(rocprofiler_counter_id_t id);
bool
is_agent_encoded_counter_id(rocprofiler_counter_id_t id);

// Counter ID encoding constants
constexpr uint64_t AGENT_BIT_OFFSET       = 32;
constexpr uint64_t AGENT_BIT_LENGTH       = 6;
constexpr uint64_t BASE_METRIC_BIT_LENGTH = 16;
constexpr uint64_t AGENT_MASK             = ((1ULL << AGENT_BIT_LENGTH) - 1);
constexpr uint64_t BASE_METRIC_MASK       = ((1ULL << BASE_METRIC_BIT_LENGTH) - 1);
constexpr uint8_t  AGENT_ENCODING_OFFSET  = 1;  // Offset to reserve 0 for detection

const std::unordered_map<int, rocprofiler_profile_counter_instance_types>&
aqlprofile_id_to_rocprof_instance();

}  // namespace counters
}  // namespace rocprofiler

rocprofiler_counter_id_t
rocprofiler::counters::rec_to_counter_id(rocprofiler_counter_instance_id_t id)
{
    // Extract base metric ID from instance record (bits 63-48)
    uint16_t base_metric = static_cast<uint16_t>(id >> DIM_BIT_LENGTH);

    // Extract agent encoding from ROCPROFILER_DIMENSION_AGENT dimension field
    uint8_t agent_encoded = static_cast<uint8_t>(rec_to_dim_pos(id, ROCPROFILER_DIMENSION_AGENT));

    // Reconstruct full agent-encoded counter ID
    // Note: agent_encoded includes the offset, but set_agent_in_counter_id() adds the offset,
    // so we need to subtract it first to get the raw logical_node_id
    rocprofiler_counter_id_t counter_id{.handle = 0};
    set_base_metric_in_counter_id(counter_id, base_metric);
    set_agent_in_counter_id(counter_id,
                            agent_encoded > 0 ? agent_encoded - AGENT_ENCODING_OFFSET : 0);

    return counter_id;
}

void
rocprofiler::counters::set_dim_in_rec(rocprofiler_counter_instance_id_t&         id,
                                      rocprofiler_profile_counter_instance_types dim,
                                      size_t                                     value)
{
    uint64_t bit_length = DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST;

    if(dim == ROCPROFILER_DIMENSION_NONE)
    {
        // Set all 48 bits of dimension
        id         = (id & ~(MAX_64 >> COUNTER_BIT_LENGTH)) | value;
        bit_length = DIM_BIT_LENGTH;
    }
    else
    {
        uint64_t mask = (MAX_64 >> (BITS_IN_UINT64 - bit_length)) << ((dim - 1) * bit_length);
        // Reset bits to 0 for dimension. Does so by getting the bit length as F's then
        // shifiting that into the position of dim. Not's that value and then and's it
        // with id.
        id = (id & ~(mask));
        // Set the value for the dimenstion
        id = id | (value << ((dim - 1) * bit_length));
    }

    CHECK(value <= (MAX_64 >> (BITS_IN_UINT64 - bit_length)))
        << "Dimension value exceeds max allowed";
}

void
rocprofiler::counters::set_counter_in_rec(rocprofiler_counter_instance_id_t& id,
                                          rocprofiler_counter_id_t           value)
{
    // Extract base metric from agent-encoded counter ID
    uint16_t base_metric = get_base_metric_from_counter_id(value);

    // Maximum counter value given the current setup (16-bit field)
    CHECK(base_metric <= 0xffff) << "Base metric ID exceeds max allowed";

    // Reset bits to 0 for counter id (bits 63-48)
    id = id & ~((MAX_64 >> (BITS_IN_UINT64 - DIM_BIT_LENGTH)) << (DIM_BIT_LENGTH));
    // Set the base metric ID in bits 63-48
    id = id | (static_cast<uint64_t>(base_metric) << DIM_BIT_LENGTH);

    // Store agent encoding in ROCPROFILER_DIMENSION_AGENT dimension field
    // NOTE: ROCPROFILER_DIMENSION_AGENT is a special dimension used to store agent information
    // This field is for internal use only and should not be set by external code
    uint8_t agent_encoded = get_agent_from_counter_id(value);

    // Unconditionally set DIMENSION_AGENT, even if agent_encoded is 0
    // (This preserves the agent encoding for all counter IDs)
    set_dim_in_rec(id, ROCPROFILER_DIMENSION_AGENT, agent_encoded);
}

size_t
rocprofiler::counters::rec_to_dim_pos(rocprofiler_counter_instance_id_t          id,
                                      rocprofiler_profile_counter_instance_types dim)
{
    if(dim == ROCPROFILER_DIMENSION_NONE)
    {
        // read all 48 bits of dimension
        return id & (MAX_64 >> COUNTER_BIT_LENGTH);
    }

    size_t bit_length = DIM_BIT_LENGTH / ROCPROFILER_DIMENSION_LAST;
    id = id & ((MAX_64 >> (BITS_IN_UINT64 - bit_length)) << ((dim - 1) * bit_length));
    return id >> ((dim - 1) * bit_length);
}

// Counter ID encoding/decoding implementations
//
// NEW COUNTER ID REPRESENTATION (Agent-Specific Counter IDs):
// ============================================================
// Counter IDs (rocprofiler_counter_id_t::handle, 64-bit) are now agent-specific.
// The counter ID encodes both the base metric ID and the agent's logical_node_id.
//
// Bit Layout:
//   Bits 63-38: Reserved/unused (26 bits)
//   Bits 37-32: Agent logical_node_id (6 bits) - supports up to 64 agents
//   Bits 31-16: Reserved/unused (16 bits)
//   Bits 15-0:  Base metric ID (16 bits) - architecture-based metric identifier
//
// Rationale:
// - Allows unique counter IDs for agents with same architecture but different configurations
//   (e.g., same gfx90a but different CU counts: 110 vs 104)
// - Maintains consistency: counter IDs are now agent-specific, matching agent-specific dimensions
// - Agent encoding is mandatory: All counter IDs must have agent encoding (agent bits != 0)
//
// Usage:
// - use set_agent_in_counter_id() / set_base_metric_in_counter_id() to encode
// - use get_agent_from_counter_id() / get_base_metric_from_counter_id() to decode
// - use is_agent_encoded_counter_id() to check if counter ID has agent encoding
