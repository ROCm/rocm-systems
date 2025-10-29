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

#include "lib/rocprofiler-sdk/counters/id_decode.hpp"

#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <unordered_map>

#include "lib/common/static_object.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/aql/aql_profile_v2.h"

namespace rocprofiler
{
namespace counters
{
const DimensionMap&
dimension_map()
{
    static auto*& _v = common::static_object<DimensionMap>::construct(DimensionMap{
        {ROCPROFILER_DIMENSION_NONE, std::string_view("DIMENSION_NONE")},
        {ROCPROFILER_DIMENSION_XCC, std::string_view("DIMENSION_XCC")},
        {ROCPROFILER_DIMENSION_AID, std::string_view("DIMENSION_AID")},
        {ROCPROFILER_DIMENSION_SHADER_ENGINE, std::string_view("DIMENSION_SHADER_ENGINE")},
        {ROCPROFILER_DIMENSION_AGENT, std::string_view("DIMENSION_AGENT")},
        {ROCPROFILER_DIMENSION_SHADER_ARRAY, std::string_view("DIMENSION_SHADER_ARRAY")},
        {ROCPROFILER_DIMENSION_WGP, std::string_view("DIMENSION_WGP")},
        {ROCPROFILER_DIMENSION_INSTANCE, std::string_view("DIMENSION_INSTANCE")},
    });
    return *_v;
}

const std::unordered_map<int, rocprofiler_profile_counter_instance_types>&
aqlprofile_id_to_rocprof_instance()
{
    using dims_map_t = std::unordered_map<int, rocprofiler_profile_counter_instance_types>;

    static auto*& aql_to_rocprof_dims =
        common::static_object<dims_map_t>::construct([]() -> dims_map_t {
            dims_map_t data;

            aqlprofile_iterate_event_ids(
                [](int id, const char* name, void* userdata) -> hsa_status_t {
                    const std::unordered_map<std::string_view,
                                             rocprofiler_profile_counter_instance_types>
                        aql_string_to_dim = {
                            {"XCD", ROCPROFILER_DIMENSION_XCC},
                            {"AID", ROCPROFILER_DIMENSION_AID},
                            {"SE", ROCPROFILER_DIMENSION_SHADER_ENGINE},
                            {"SA", ROCPROFILER_DIMENSION_SHADER_ARRAY},
                            {"WGP", ROCPROFILER_DIMENSION_WGP},
                            {"INSTANCE", ROCPROFILER_DIMENSION_INSTANCE},
                        };

                    if(const auto* inst_type =
                           rocprofiler::common::get_val(aql_string_to_dim, name))
                    {
                        // Supported instance type
                        auto& map = *static_cast<
                            std::unordered_map<int, rocprofiler_profile_counter_instance_types>*>(
                            userdata);
                        map.emplace(id, *inst_type);
                    }
                    return HSA_STATUS_SUCCESS;
                },
                static_cast<void*>(&data));
            return data;
        }());

    return *aql_to_rocprof_dims;
}

// Counter ID encoding/decoding implementations
void
set_agent_in_counter_id(rocprofiler_counter_id_t& id, uint8_t agent_logical_node_id)
{
    // Check that logical_node_id + offset fits in 6 bits
    // With AGENT_ENCODING_OFFSET=1, this allows logical_node_id 0-62 (63 agents)
    CHECK(agent_logical_node_id < ((1 << AGENT_BIT_LENGTH) - AGENT_ENCODING_OFFSET))
        << "Agent logical_node_id " << static_cast<int>(agent_logical_node_id)
        << " exceeds limit (max " << ((1 << AGENT_BIT_LENGTH) - AGENT_ENCODING_OFFSET - 1)
        << " to allow for encoding offset)";

    // Add encoding offset to ensure agent 0 is detectable (non-zero)
    uint8_t agent_encoded = agent_logical_node_id + AGENT_ENCODING_OFFSET;

    // Clear agent bits and set new value
    id.handle = (id.handle & ~(AGENT_MASK << AGENT_BIT_OFFSET)) |
                (static_cast<uint64_t>(agent_encoded) << AGENT_BIT_OFFSET);
}

uint8_t
get_agent_from_counter_id(rocprofiler_counter_id_t id)
{
    return static_cast<uint8_t>((id.handle >> AGENT_BIT_OFFSET) & AGENT_MASK);
}

void
set_base_metric_in_counter_id(rocprofiler_counter_id_t& id, uint16_t metric_id)
{
    CHECK(metric_id <= BASE_METRIC_MASK) << "Base metric ID exceeds 16-bit limit";
    // Clear base metric bits and set new value
    id.handle = (id.handle & ~BASE_METRIC_MASK) | metric_id;
}

uint16_t
get_base_metric_from_counter_id(rocprofiler_counter_id_t id)
{
    return static_cast<uint16_t>(id.handle & BASE_METRIC_MASK);
}

bool
is_agent_encoded_counter_id(rocprofiler_counter_id_t id)
{
    // Check if agent bits are non-zero
    return ((id.handle >> AGENT_BIT_OFFSET) & AGENT_MASK) != 0;
}

}  // namespace counters
}  // namespace rocprofiler
