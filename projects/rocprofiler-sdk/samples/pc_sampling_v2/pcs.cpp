// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
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

#include "pcs.hpp"
#include "utils.hpp"

#include "common/defines.hpp"

#include <cassert>
#include <cstdio>
#include <iomanip>
#include <memory>
#include <sstream>

namespace client
{
namespace pcs
{
tool_agent_info_vec_t        gpu_agents = {};
pc_sampling_buffer_id_vec_t* buffer_ids = nullptr;

namespace
{
constexpr uint64_t host_trap_interval  = 10000;    // 10ms
constexpr uint64_t stochastic_interval = 1048576;  // 2 ^ 20 cycles
}  // namespace

void
init()
{
    buffer_ids = new pc_sampling_buffer_id_vec_t();
}

void
fini()
{
    buffer_ids->clear();
    delete buffer_ids;
    buffer_ids = nullptr;
}

pc_sampling_buffer_id_vec_t*
get_pc_sampling_buffer_ids()
{
    return buffer_ids;
}

rocprofiler_status_t
find_all_gpu_agents_supporting_pc_sampling_impl(rocprofiler_agent_version_t version,
                                                const void**                agents,
                                                size_t                      num_agents,
                                                void*                       user_data)
{
    assert(version == ROCPROFILER_AGENT_INFO_VERSION_0);
    if(!user_data) return ROCPROFILER_STATUS_ERROR;

    std::stringstream ss;

    auto* _out_agents = static_cast<tool_agent_info_vec_t*>(user_data);
    auto* _agents     = reinterpret_cast<const rocprofiler_agent_t**>(agents);
    for(size_t i = 0; i < num_agents; i++)
    {
        if(_agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            auto tool_gpu_agent           = std::make_unique<tool_agent_info>();
            tool_gpu_agent->agent_id      = _agents[i]->id;
            tool_gpu_agent->avail_configs = std::make_unique<avail_configs_vec_t>();
            tool_gpu_agent->arbiter_fields = std::make_unique<arbiter_fields_vec_t>();
            tool_gpu_agent->agent         = _agents[i];
            // Check if the GPU agent supports PC sampling. If so, add it to the
            // output list `_out_agents`.
            if(query_avail_configs_for_agent(tool_gpu_agent.get()))
            {
                // Query arbiter state fields supported by this agent and store per-agent.
                query_arbiter_fields_for_agent(tool_gpu_agent.get());
                _out_agents->push_back(std::move(tool_gpu_agent));
            }
        }

        ss << "[" << __FUNCTION__ << "] " << _agents[i]->name << " :: "
           << "id=" << _agents[i]->id.handle << ", "
           << "type=" << _agents[i]->type << "\n";
    }

    *utils::get_output_stream() << ss.str() << "\n";

    return ROCPROFILER_STATUS_SUCCESS;
}

void
find_all_gpu_agents_supporting_pc_sampling()
{
    ROCPROFILER_CHECK(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           &find_all_gpu_agents_supporting_pc_sampling_impl,
                                           sizeof(rocprofiler_agent_t),
                                           static_cast<void*>(&gpu_agents)));
}

bool
query_avail_configs_for_agent(tool_agent_info* agent_info)
{
    agent_info->avail_configs->clear();

    auto cb = [](const rocprofiler_pc_sampling_configuration_t* configs,
                 size_t                                         num_config,
                 void*                                          user_data) {
        auto* avail_configs = static_cast<avail_configs_vec_t*>(user_data);
        for(size_t i = 0; i < num_config; i++)
        {
            avail_configs->emplace_back(configs[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto status = rocprofiler_query_pc_sampling_agent_configurations(
        agent_info->agent_id, cb, agent_info->avail_configs.get());

    std::stringstream ss;

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        ss << "Querying PC sampling capabilities failed with status=" << status
           << " :: " << rocprofiler_get_status_string(status) << "\n";
        *utils::get_output_stream() << ss.str() << "\n";
        return false;
    }
    else if(agent_info->avail_configs->empty())
    {
        return false;
    }

    ss << "The agent with the id: " << agent_info->agent_id.handle << " supports the "
       << agent_info->avail_configs->size() << " configurations: "
       << "\n";
    size_t ind = 0;
    for(auto& cfg : *agent_info->avail_configs)
    {
        ss << "(" << ++ind << ".) "
           << "method: " << cfg.method << ", "
           << "unit: " << cfg.unit << ", "
           << "min_interval: " << cfg.min_interval << ", "
           << "max_interval: " << cfg.max_interval << ", "
           << "flags: " << std::hex << cfg.flags << std::dec
           << ((cfg.flags == ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_INTERVAL_POW2)
                   ? " (an interval value must be power of 2)"
                   : "")
           << "\n";
    }

    *utils::get_output_stream() << ss.str() << std::flush;

    return true;
}

void
query_arbiter_fields_for_agent(tool_agent_info* agent_info)
{
    agent_info->arbiter_fields->clear();

    auto cb = [](const rocprofiler_pc_sampling_arbiter_state_field_id_t* fields,
                 size_t                                                  num_fields,
                 void*                                                   user_data) {
        auto* out = static_cast<arbiter_fields_vec_t*>(user_data);
        for(size_t i = 0; i < num_fields; i++)
        {
            out->emplace_back(fields[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto status = rocprofiler_query_pc_sampling_arbiter_fields(
        agent_info->agent_id, cb, agent_info->arbiter_fields.get());

    std::stringstream ss;

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        ss << "Querying arbiter state fields for agent " << agent_info->agent_id.handle
           << " failed with status=" << status << " :: " << rocprofiler_get_status_string(status)
           << "\n";
        *utils::get_output_stream() << ss.str();
        return;
    }

    ss << "Agent " << agent_info->agent_id.handle << " supports "
       << agent_info->arbiter_fields->size() << " arbiter state fields:";
    for(auto field_id : *agent_info->arbiter_fields)
    {
        const char* name     = nullptr;
        uint64_t    name_len = 0;
        auto        name_status =
            rocprofiler_get_pc_sampling_arbiter_state_field_name(field_id, &name, &name_len);
        if(name_status == ROCPROFILER_STATUS_SUCCESS && name != nullptr)
        {
            ss << " " << std::string(name, name_len);
        }
        else
        {
            ss << " UNKNOWN(" << static_cast<int>(field_id) << ")";
        }
    }
    ss << "\n";

    *utils::get_output_stream() << ss.str() << std::flush;
}

void
configure_pc_sampling_v2_prefer_stochastic(tool_agent_info*         agent_info,
                                           rocprofiler_context_id_t context_id,
                                           rocprofiler_buffer_id_t  buffer_id)
{
    auto   stochastic_picked = false;
    int    failures          = 10;
    size_t interval          = 0;
    do
    {
        auto success = query_avail_configs_for_agent(agent_info);
        if(!success)
        {
            ROCPROFILER_CHECK(ROCPROFILER_STATUS_ERROR);
        }

        const rocprofiler_pc_sampling_configuration_t* first_host_trap_config  = nullptr;
        const rocprofiler_pc_sampling_configuration_t* first_stochastic_config = nullptr;
        for(auto const& cfg : *agent_info->avail_configs)
        {
            if(cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC)
            {
                first_stochastic_config = &cfg;
                stochastic_picked       = true;
                break;
            }
            else if(!first_host_trap_config &&
                    cfg.method == ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP)
            {
                first_host_trap_config = &cfg;
            }
        }

        const rocprofiler_pc_sampling_configuration_t* picked_cfg =
            (first_stochastic_config != nullptr) ? first_stochastic_config : first_host_trap_config;

        if(picked_cfg->min_interval == picked_cfg->max_interval)
        {
            interval = picked_cfg->min_interval;
        }
        else
        {
            interval = stochastic_picked ? stochastic_interval : host_trap_interval;
        }

        // Pick record kind based on sampling method:
        //   - Host-trap -> V1 (has hw_id, workgroup_position, wave_in_group)
        //   - Stochastic -> V2 (V1 fields + snapshot_information with arbiter_state)
        // Always include INVALID_SAMPLE to observe any invalid/error samples.
        auto record_kind = stochastic_picked ? ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE
                                             : ROCPROFILER_PC_SAMPLING_RECORD_V1_SAMPLE;

        rocprofiler_pc_sampling_record_kind_t record_kinds[] = {
            record_kind, ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE};

        auto status =
            rocprofiler_configure_pc_sampling_service_v2(context_id,
                                                        agent_info->agent_id,
                                                        picked_cfg->method,
                                                        picked_cfg->unit,
                                                        interval,
                                                        buffer_id,
                                                        record_kinds,
                                                        2,
                                                        0);
        if(status == ROCPROFILER_STATUS_SUCCESS)
        {
            *utils::get_output_stream()
                << ">>> Configured v2 PC sampling: "
                << (stochastic_picked ? "stochastic (V2 record)" : "host-trap (V1 record)")
                << " with interval: " << interval << " "
                << (stochastic_picked ? "clock-cycles" : "micro seconds")
                << " on agent: " << agent_info->agent->id.handle << "\n";
            return;
        }
        else if(status != ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE)
        {
            ROCPROFILER_CHECK(status);
        }
    } while(--failures);

    ROCPROFILER_CHECK(ROCPROFILER_STATUS_ERROR);
}

// ---------------------------------------------------------------------------
// Printing helpers for v2 record types
// ---------------------------------------------------------------------------

void
print_sample_v0(std::ostream& os, const rocprofiler_pc_sampling_record_v0_t* sample)
{
    os << "(code_obj_id, offset): (" << sample->pc.code_object_id << ", 0x" << std::hex
       << sample->pc.code_object_offset << "), " << std::dec
       << "timestamp: " << sample->timestamp << ", "
       << "exec: " << std::hex << std::setw(16) << sample->exec_mask << std::dec << ", "
       << "dispatch_id: " << std::setw(7) << sample->dispatch_id << ", "
       << "correlation: {internal=" << std::setw(7) << sample->correlation_id.internal << ", "
       << "external=" << std::setw(5) << sample->correlation_id.external.value << "}"
       << "\n";
}

void
print_sample_v1(std::ostream& os, const rocprofiler_pc_sampling_record_v1_t* sample)
{
    os << "(code_obj_id, offset): (" << sample->pc.code_object_id << ", 0x" << std::hex
       << sample->pc.code_object_offset << "), " << std::dec
       << "timestamp: " << sample->timestamp << ", "
       << "exec: " << std::hex << std::setw(16) << sample->exec_mask << std::dec << ", "
       << "workgroup_pos_(x=" << std::setw(5) << sample->workgroup_position.x << ", "
       << "y=" << std::setw(5) << sample->workgroup_position.y << ", "
       << "z=" << std::setw(5) << sample->workgroup_position.z << "), "
       << "wave_in_group: " << std::setw(2) << sample->wave_in_group << ", "
       << "chiplet: " << std::setw(2) << static_cast<unsigned int>(sample->hw_id.chiplet) << ", "
       << "dispatch_id: " << std::setw(7) << sample->dispatch_id << ", "
       << "correlation: {internal=" << std::setw(7) << sample->correlation_id.internal << ", "
       << "external=" << std::setw(5) << sample->correlation_id.external.value << "}"
       << "\n";
}

/**
 * @brief Print a V2 stochastic PC sampling record, including arbiter state
 * decoded via the per-agent arbiter fields.
 *
 * @param agent_info The agent info containing the pre-queried arbiter_fields.
 */
void
print_sample_v2(std::ostream&                              os,
                const rocprofiler_pc_sampling_record_v2_t* sample,
                const tool_agent_info*                     agent_info)
{
    os << "(code_obj_id, offset): (" << sample->pc.code_object_id << ", 0x" << std::hex
       << sample->pc.code_object_offset << "), " << std::dec
       << "timestamp: " << sample->timestamp << ", "
       << "exec: " << std::hex << std::setw(16) << sample->exec_mask << std::dec << ", "
       << "workgroup_pos_(x=" << std::setw(5) << sample->workgroup_position.x << ", "
       << "y=" << std::setw(5) << sample->workgroup_position.y << ", "
       << "z=" << std::setw(5) << sample->workgroup_position.z << "), "
       << "wave_in_group: " << std::setw(2) << sample->wave_in_group << ", "
       << "chiplet: " << std::setw(2) << static_cast<unsigned int>(sample->hw_id.chiplet) << ", "
       << "dispatch_id: " << std::setw(7) << sample->dispatch_id << ", "
       << "correlation: {internal=" << std::setw(7) << sample->correlation_id.internal << ", "
       << "external=" << std::setw(5) << sample->correlation_id.external.value << "}, ";

    // Print snapshot_information fields
    auto& snap = sample->snapshot_information;
    if(snap.wave_issued)
    {
        const char* inst_name     = nullptr;
        uint64_t    inst_name_len = 0;
        auto        inst_status   = rocprofiler_get_pc_sampling_instruction_type_name_(
            static_cast<rocprofiler_pc_sampling_instruction_type_t>(snap.instruction_type),
            &inst_name,
            &inst_name_len);
        if(inst_status == ROCPROFILER_STATUS_SUCCESS && inst_name != nullptr)
            os << "wave issued " << std::string(inst_name, inst_name_len) << " instruction, ";
        else
            os << "wave issued instruction (type=" << static_cast<unsigned int>(snap.instruction_type)
               << "), ";
    }
    else
    {
        const char* reason_name     = nullptr;
        uint64_t    reason_name_len = 0;
        auto        reason_status   = rocprofiler_get_pc_sampling_instruction_not_issued_reason_name_(
            static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(
                snap.no_issue_reason),
            &reason_name,
            &reason_name_len);
        if(reason_status == ROCPROFILER_STATUS_SUCCESS && reason_name != nullptr)
            os << "wave stalled: " << std::string(reason_name, reason_name_len) << ", ";
        else
            os << "wave stalled (reason=" << static_cast<unsigned int>(snap.no_issue_reason)
               << "), ";
    }

    os << "wave_count: " << static_cast<unsigned int>(snap.wave_count) << ", ";

    // Decode arbiter_state using the per-agent arbiter fields
    if(agent_info != nullptr && agent_info->arbiter_fields != nullptr &&
       !agent_info->arbiter_fields->empty())
    {
        os << "arbiter_state: {";

        // Use the extraction API to get field values
        auto arbiter_cb = [](const rocprofiler_pc_sampling_arbiter_state_field_id_t* field_ids,
                             const uint32_t*                                         values,
                             size_t                                                  num_fields,
                             void* user_data) -> rocprofiler_status_t {
            auto& out = *static_cast<std::ostream*>(user_data);
            for(size_t i = 0; i < num_fields; i++)
            {
                if(i > 0) out << ", ";
                const char* field_name     = nullptr;
                uint64_t    field_name_len = 0;
                auto        name_status    = rocprofiler_get_pc_sampling_arbiter_state_field_name(
                    field_ids[i], &field_name, &field_name_len);
                if(name_status == ROCPROFILER_STATUS_SUCCESS && field_name != nullptr)
                    out << std::string(field_name, field_name_len);
                else
                    out << "field_" << static_cast<int>(field_ids[i]);
                out << "=" << values[i];
            }
            return ROCPROFILER_STATUS_SUCCESS;
        };

        auto extract_status = rocprofiler_pc_sampling_get_arbiter_state_fields(
            snap.arbiter_state,
            agent_info->arbiter_fields->data(),
            agent_info->arbiter_fields->size(),
            arbiter_cb,
            static_cast<void*>(&os));

        if(extract_status != ROCPROFILER_STATUS_SUCCESS)
        {
            os << "ERROR extracting arbiter state";
        }
        os << "}";
    }

    os << "\n";
}

void
print_sample_invalid(std::ostream& os, const rocprofiler_pc_sampling_record_invalid_t* /*sample*/)
{
    os << "Invalid sample detected.\n";
}

// ---------------------------------------------------------------------------
// Buffer callback
// ---------------------------------------------------------------------------

void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void* /*data*/,
                                 uint64_t drop_count)
{
    std::stringstream ss;
    ss << "The number of delivered samples is: " << num_headers << ", "
       << "while the number of dropped samples is: " << drop_count << "\n";

    for(size_t i = 0; i < num_headers; i++)
    {
        auto* cur_header = headers[i];

        if(cur_header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(cur_header->hash !=
                rocprofiler_record_header_compute_hash(cur_header->category, cur_header->kind))
        {
            throw std::runtime_error{"rocprofiler_record_header_t (category | kind) != hash"};
        }
        else if(cur_header->category == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING)
        {
            if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_V0_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_v0_t*>(cur_header->payload);
                print_sample_v0(ss, pc_sample);
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_V1_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_v1_t*>(cur_header->payload);
                print_sample_v1(ss, pc_sample);
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_v2_t*>(cur_header->payload);
                // Find the agent info to look up per-agent arbiter fields.
                // In this sample, we use the first agent that has arbiter fields.
                // A production tool would map buffer_id -> agent_id for precise lookup.
                const tool_agent_info* agent_for_sample = nullptr;
                for(const auto& agent : gpu_agents)
                {
                    if(agent->arbiter_fields && !agent->arbiter_fields->empty())
                    {
                        agent_for_sample = agent.get();
                        break;
                    }
                }
                print_sample_v2(ss, pc_sample, agent_for_sample);
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_invalid_t*>(cur_header->payload);
                print_sample_invalid(ss, pc_sample);
            }
            else
            {
                // Also handle old record types for completeness
                assert(false && "Unexpected PC sampling record kind in v2 sample");
            }
        }
        else
        {
            throw std::runtime_error{"unexpected rocprofiler_record_header_t category + kind"};
        }
    }

    *utils::get_output_stream() << ss.str() << "\n";
}
}  // namespace pcs
}  // namespace client
