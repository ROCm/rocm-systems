// MIT License
//
// Copyright (c) 2023-2026 ROCm Developer Tools
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
tool_agent_info_vec_t*       gpu_agents = nullptr;
pc_sampling_buffer_id_vec_t* buffer_ids = nullptr;

namespace
{
/**
 * @brief Helper: query v2 configs for a single record kind + INVALID_SAMPLE
 * with the given API flags.  Returns the collected configurations.
 */
std::vector<rocprofiler_pc_sampling_configuration_v2_t>
query_v2_configs(rocprofiler_agent_id_t                agent_id,
                 rocprofiler_pc_sampling_record_kind_t record_kind,
                 rocprofiler_pc_sampling_api_flags_t   flags)
{
    auto configs = std::vector<rocprofiler_pc_sampling_configuration_v2_t>{};

    auto cb = [](const rocprofiler_pc_sampling_configuration_v2_t** cfgs, size_t num, void* ud) {
        auto* out = static_cast<std::vector<rocprofiler_pc_sampling_configuration_v2_t>*>(ud);
        for(size_t i = 0; i < num; i++)
            out->emplace_back(*cfgs[i]);
        return ROCPROFILER_STATUS_SUCCESS;
    };

    // INVALID_SAMPLE is method-agnostic: it does not constrain method selection, so it neither
    // blocks the stochastic->host-trap fallback nor causes host-trap configurations to be filtered
    // out. Requesting it uniformly keeps the sample simple; host-trap sessions simply deliver no
    // invalid records today.
    rocprofiler_pc_sampling_record_kind_t record_kinds[] = {
        record_kind, ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE};

    auto status = rocprofiler_pc_sampling_query_agent_configurations_v2(
        agent_id, record_kinds, 2, flags, cb, &configs);

    if(status != ROCPROFILER_STATUS_SUCCESS) configs.clear();

    return configs;
}
}  // namespace

void
init()
{
    buffer_ids = new pc_sampling_buffer_id_vec_t();
    gpu_agents = new tool_agent_info_vec_t();
}

void
fini()
{
    buffer_ids->clear();
    delete buffer_ids;
    buffer_ids = nullptr;

    gpu_agents->clear();
    delete gpu_agents;
    gpu_agents = nullptr;
}

pc_sampling_buffer_id_vec_t*
get_pc_sampling_buffer_ids()
{
    return buffer_ids;
}

tool_agent_info_vec_t*
get_gpu_agents()
{
    return gpu_agents;
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

    ss << "Discovered " << num_agents << " agent(s):\n";
    size_t gpu_index = 0;
    for(size_t i = 0; i < num_agents; i++)
    {
        ss << "  " << (i + 1) << ". " << _agents[i]->name << " (id=" << _agents[i]->id.handle
           << ", type=" << _agents[i]->type << ")\n";

        if(_agents[i]->type == ROCPROFILER_AGENT_TYPE_GPU)
        {
            auto tool_gpu_agent        = std::make_unique<tool_agent_info>();
            tool_gpu_agent->agent_id   = _agents[i]->id;
            tool_gpu_agent->ext_fields = std::make_unique<ext_fields_vec_t>();
            tool_gpu_agent->agent      = _agents[i];

            ++gpu_index;
            // Discover the most comprehensive record version via the v2 query API.
            // If the agent supports PC sampling, memoize the config and add it.
            if(query_most_comprehensive_config_for_agent(tool_gpu_agent.get()))
            {
                // Query snapshot ext_data fields supported by this agent and store per-agent.
                query_snapshot_ext_fields_for_agent(tool_gpu_agent.get());
                _out_agents->push_back(std::move(tool_gpu_agent));
            }
        }
    }

    *utils::get_output_stream() << ss.str() << std::flush;

    return ROCPROFILER_STATUS_SUCCESS;
}

void
find_all_gpu_agents_supporting_pc_sampling()
{
    ROCPROFILER_CHECK(
        rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                           &find_all_gpu_agents_supporting_pc_sampling_impl,
                                           sizeof(rocprofiler_agent_t),
                                           static_cast<void*>(get_gpu_agents())));
}

bool
query_most_comprehensive_config_for_agent(tool_agent_info* agent_info)
{
    std::stringstream ss;
    ss << "Agent " << agent_info->agent_id.handle
       << ": searching for most comprehensive PC sampling record version...\n";

    // Use PREFER_STOCHASTIC: the runtime will fall back to host-trap
    // automatically if stochastic is not available on the agent.
    constexpr auto flags = ROCPROFILER_PC_SAMPLING_API_FLAG_PREFER_STOCHASTIC;

    // Iterate from the most comprehensive record version down to V0.
    // Unsupported versions (e.g. V3/V4 on non-GFX12 agents) will be rejected by the library,
    // so the loop naturally skips them.
    //
    // This top-down scan assumes newer (more comprehensive) record kinds are introduced for newer
    // architectures only. If a future record kind were ever made available on older architectures,
    // this scan could select it (and its associated method) on an older arch; revisit the discovery
    // strategy here if that ever happens. In practice this does not occur today.
    for(int kind = static_cast<int>(ROCPROFILER_PC_SAMPLING_RECORD_LAST) - 1;
        kind >= static_cast<int>(ROCPROFILER_PC_SAMPLING_RECORD_V0_SAMPLE);
        --kind)
    {
        auto record_kind = static_cast<rocprofiler_pc_sampling_record_kind_t>(kind);
        auto configs     = query_v2_configs(agent_info->agent_id, record_kind, flags);

        if(!configs.empty())
        {
            agent_info->most_comprehensive_record_kind = record_kind;
            agent_info->most_comprehensive_api_flags   = flags;
            agent_info->most_comprehensive_config      = configs[0];

            ss << "  Selected record kind: " << static_cast<int>(record_kind)
               << ", unit: " << configs[0].unit << ", interval: [" << configs[0].min_interval
               << ", " << configs[0].max_interval << "]\n";
            *utils::get_output_stream() << ss.str() << std::flush;
            return true;
        }
    }

    ss << "  No PC sampling configuration found.\n";
    *utils::get_output_stream() << ss.str() << std::flush;
    return false;
}

void
query_snapshot_ext_fields_for_agent(tool_agent_info* agent_info)
{
    agent_info->ext_fields->clear();
    agent_info->ext_field_names.clear();

    auto cb = [](const rocprofiler_pc_sampling_snapshot_ext_field_id_t* fields,
                 size_t                                                 num_fields,
                 void*                                                  user_data) {
        auto* out = static_cast<ext_fields_vec_t*>(user_data);
        for(size_t i = 0; i < num_fields; i++)
        {
            out->emplace_back(fields[i]);
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    // Call unconditionally for the chosen record kind. Record kinds that carry no snapshot
    // information (host-trap suitable records V0/V1/V3) return SUCCESS with an empty field list, so
    // an empty result simply means ext_data extraction is not applicable -- no special-casing here.
    auto status = rocprofiler_pc_sampling_query_snapshot_ext_fields(
        agent_info->agent_id,
        agent_info->most_comprehensive_record_kind,
        cb,
        agent_info->ext_fields.get());

    std::stringstream ss;

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        ss << "Querying snapshot ext_data fields for agent " << agent_info->agent_id.handle
           << " failed with status=" << status << " :: " << rocprofiler_get_status_string(status)
           << "\n";
        *utils::get_output_stream() << ss.str();
        return;
    }

    if(agent_info->ext_fields->empty())
    {
        ss << "Agent " << agent_info->agent_id.handle
           << " exposes no snapshot ext_data fields for the selected record kind ("
           << static_cast<int>(agent_info->most_comprehensive_record_kind) << ").\n";
        *utils::get_output_stream() << ss.str() << std::flush;
        return;
    }

    // Build the field-name LUT once so the buffer callback never has to query names.
    ss << "Agent " << agent_info->agent_id.handle << " supports " << agent_info->ext_fields->size()
       << " snapshot ext_data field(s):\n";
    size_t field_index = 0;
    for(auto field_id : *agent_info->ext_fields)
    {
        const char* name     = nullptr;
        uint64_t    name_len = 0;
        auto        name_status =
            rocprofiler_pc_sampling_get_snapshot_ext_field_name(field_id, &name, &name_len);
        ++field_index;
        if(name_status == ROCPROFILER_STATUS_SUCCESS && name != nullptr)
        {
            auto name_str = std::string(name, name_len);
            agent_info->ext_field_names.emplace(field_id, name_str);
            ss << "  " << field_index << ". " << name_str << "\n";
        }
        else
        {
            auto fallback = "field_" + std::to_string(static_cast<int>(field_id));
            agent_info->ext_field_names.emplace(field_id, fallback);
            ss << "  " << field_index << ". UNKNOWN(" << static_cast<int>(field_id) << ")\n";
        }
    }

    *utils::get_output_stream() << ss.str() << std::flush;
}

void
configure_pc_sampling_for_agent(tool_agent_info*         agent_info,
                                rocprofiler_context_id_t context_id,
                                rocprofiler_buffer_id_t  buffer_id)
{
    if(agent_info->most_comprehensive_record_kind == ROCPROFILER_PC_SAMPLING_RECORD_NONE)
    {
        ROCPROFILER_CALL(
            ROCPROFILER_STATUS_ERROR,
            "configure_pc_sampling_for_agent called without previous configuration inquiry");
    }

    auto& cfg = agent_info->most_comprehensive_config;

    // Cycle-based (stochastic) sampling needs a larger interval due to
    // hardware constraints; time-based (host-trap) uses a fixed 10ms interval.
    auto interval = (cfg.unit == ROCPROFILER_PC_SAMPLING_UNIT_CYCLES) ? STOCHASTIC_INTERVAL
                                                                      : HOST_TRAP_INTERVAL;

    // INVALID_SAMPLE is method-agnostic (see query_v2_configs): it does not affect method
    // selection, so it is safe to request alongside any record kind. On host-trap sessions no
    // invalid records are delivered today; on stochastic sessions it opts into invalid-sample
    // delivery.
    rocprofiler_pc_sampling_record_kind_t record_kinds[] = {
        agent_info->most_comprehensive_record_kind, ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE};

    int retries = 10;
    do
    {
        auto status =
            rocprofiler_pc_sampling_configure_service_v2(context_id,
                                                         agent_info->agent_id,
                                                         cfg.unit,
                                                         interval,
                                                         buffer_id,
                                                         record_kinds,
                                                         2,
                                                         agent_info->most_comprehensive_api_flags);
        if(status == ROCPROFILER_STATUS_SUCCESS)
        {
            *utils::get_output_stream()
                << ">>> Configured PC sampling (record kind="
                << static_cast<int>(agent_info->most_comprehensive_record_kind)
                << ", interval=" << interval << ") on agent " << agent_info->agent->id.handle
                << "\n";
            return;
        }
        else if(status != ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE)
        {
            ROCPROFILER_CHECK(status);
        }
    } while(--retries);

    ROCPROFILER_CHECK(ROCPROFILER_STATUS_ERROR);
}

// ---------------------------------------------------------------------------
// Printing helpers for v2 record types
// ---------------------------------------------------------------------------

/**
 * @brief Print fields common to V1 and V2 records: pc, timestamp, exec_mask,
 * workgroup_position, wave_in_group, chiplet, dispatch_id, and correlation_id.
 *
 * V0 records lack the workgroup/hw_id fields and are printed separately.
 */
template <typename PcSamplingRecordT>
void
print_sample_common_fields(std::ostream& os, const PcSamplingRecordT* sample)
{
    os << "(code_obj_id, offset): (" << sample->pc.code_object_id << ", 0x" << std::hex
       << sample->pc.code_object_offset << "), " << std::dec << "timestamp: " << sample->timestamp
       << ", "
       << "exec: " << std::hex << std::setw(16) << sample->exec_mask << std::dec << ", "
       << "workgroup_pos_(x=" << std::setw(5) << sample->workgroup_position.x << ", "
       << "y=" << std::setw(5) << sample->workgroup_position.y << ", "
       << "z=" << std::setw(5) << sample->workgroup_position.z << "), "
       << "wave_in_group: " << std::setw(2) << sample->wave_in_group << ", "
       << "chiplet: " << std::setw(2) << static_cast<unsigned int>(sample->hw_id.chiplet) << ", "
       << "dispatch_id: " << std::setw(7) << sample->dispatch_id << ", "
       << "correlation: {internal=" << std::setw(7) << sample->correlation_id.internal << ", "
       << "external=" << std::setw(5) << sample->correlation_id.external.value << "}";
}

void
print_sample_v0(std::ostream& os, const rocprofiler_pc_sampling_record_v0_t* sample)
{
    os << "(code_obj_id, offset): (" << sample->pc.code_object_id << ", 0x" << std::hex
       << sample->pc.code_object_offset << "), " << std::dec << "timestamp: " << sample->timestamp
       << ", "
       << "exec: " << std::hex << std::setw(16) << sample->exec_mask << std::dec << ", "
       << "dispatch_id: " << std::setw(7) << sample->dispatch_id << ", "
       << "correlation: {internal=" << std::setw(7) << sample->correlation_id.internal << ", "
       << "external=" << std::setw(5) << sample->correlation_id.external.value << "}"
       << "\n";
}

void
print_sample_v1(std::ostream& os, const rocprofiler_pc_sampling_record_v1_t* sample)
{
    print_sample_common_fields(os, sample);
    os << "\n";
}

/**
 * @brief Print the snapshot_information block (wave issued/stalled + wave_count) shared by the
 * stochastic records V2 and V4. Emits a trailing ", ".
 */
void
print_snapshot_information(std::ostream&                                            os,
                           const rocprofiler_pc_sampling_snapshot_information_v0_t& snap)
{
    if(snap.wave_issued)
    {
        const char* inst_name     = nullptr;
        uint64_t    inst_name_len = 0;
        auto        inst_status   = rocprofiler_pc_sampling_get_instruction_type_name_v2(
            static_cast<rocprofiler_pc_sampling_instruction_type_t>(snap.instruction_type),
            &inst_name,
            &inst_name_len);
        if(inst_status == ROCPROFILER_STATUS_SUCCESS && inst_name != nullptr)
            os << "wave issued " << std::string(inst_name, inst_name_len) << " instruction, ";
        else
            os << "wave issued instruction (type="
               << static_cast<unsigned int>(snap.instruction_type) << "), ";
    }
    else
    {
        const char* reason_name     = nullptr;
        uint64_t    reason_name_len = 0;
        auto reason_status = rocprofiler_pc_sampling_get_instruction_not_issued_reason_name_v2(
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
}

/**
 * @brief Decode and print the ext_data block for a stochastic record (V2/V4) using the per-agent
 * snapshot ext fields and the memoized name map (avoids calling the name API per-sample).
 *
 * Prints nothing when the agent exposes no ext fields (e.g. host-trap suitable records).
 */
void
print_snapshot_ext_data(std::ostream&                         os,
                        rocprofiler_pc_sampling_record_kind_t record_kind,
                        const void*                           sample,
                        const tool_agent_info*                agent_info)
{
    if(agent_info == nullptr || agent_info->ext_fields == nullptr ||
       agent_info->ext_fields->empty())
        return;

    os << "ext_data: {";

    struct ext_cb_data
    {
        std::ostream*               os;
        const ext_field_name_map_t* name_map;
    };
    auto cb_data = ext_cb_data{&os, &agent_info->ext_field_names};

    auto ext_cb = [](const rocprofiler_pc_sampling_snapshot_ext_field_id_t* field_ids,
                     const uint32_t*                                        values,
                     size_t                                                 num_fields,
                     void* user_data) -> rocprofiler_status_t {
        auto& data  = *static_cast<ext_cb_data*>(user_data);
        bool  first = true;
        for(size_t i = 0; i < num_fields; i++)
        {
            // Extract the field name from the memoized map.
            auto it = data.name_map->find(field_ids[i]);
            if(it == data.name_map->end()) continue;
            if(!first) *data.os << ", ";
            // Show the field name and the value.
            *data.os << it->second << "=" << values[i];
            first = false;
        }
        return ROCPROFILER_STATUS_SUCCESS;
    };

    auto extract_status =
        rocprofiler_pc_sampling_extract_snapshot_ext_field_values(record_kind,
                                                                  sample,
                                                                  agent_info->ext_fields->data(),
                                                                  agent_info->ext_fields->size(),
                                                                  ext_cb,
                                                                  static_cast<void*>(&cb_data));

    if(extract_status != ROCPROFILER_STATUS_SUCCESS) os << "ERROR extracting ext_data fields";
    os << "}";
}

/**
 * @brief Print a V2 stochastic PC sampling record, including ext_data
 * decoded via the per-agent snapshot ext fields.
 *
 * @param agent_info The agent info containing the pre-queried ext_fields
 *                   and the memoized ext_field_names map.  Passed as
 *                   buffer client_data so there is no need to iterate over
 *                   the global agent list -- this is the recommended policy.
 */
void
print_sample_v2(std::ostream&                              os,
                const rocprofiler_pc_sampling_record_v2_t* sample,
                const tool_agent_info*                     agent_info)
{
    print_sample_common_fields(os, sample);
    os << ", ";

    print_snapshot_information(os, sample->snapshot_information);

    print_snapshot_ext_data(
        os, ROCPROFILER_PC_SAMPLING_RECORD_V2_SAMPLE, static_cast<const void*>(sample), agent_info);

    os << "\n";
}

/**
 * @brief Print a V3 host-trap PC sampling record with cluster information.
 * V3 records are only available on gfx1250.
 */
void
print_sample_v3(std::ostream& os, const rocprofiler_pc_sampling_record_v3_t* sample)
{
    print_sample_common_fields(os, sample);
    os << ", cluster_id: " << static_cast<unsigned int>(sample->cluster_id);
    if(sample->cluster_id != 0)
    {
        os << ", cluster_flat_nwg: " << static_cast<unsigned int>(sample->cluster_flat_nwg)
           << ", cluster_nwg_(x=" << static_cast<unsigned int>(sample->cluster_nwg_x)
           << ", y=" << static_cast<unsigned int>(sample->cluster_nwg_y)
           << ", z=" << static_cast<unsigned int>(sample->cluster_nwg_z) << ")"
           << ", cluster_pos_(x=" << sample->cluster_position.x
           << ", y=" << sample->cluster_position.y << ", z=" << sample->cluster_position.z << ")";
    }
    os << "\n";
}

/**
 * @brief Print a V4 stochastic PC sampling record with cluster info, snapshot_information,
 * memory counters and ext_data decoded via the per-agent snapshot ext fields.
 * V4 records are only available on gfx1250.
 */
void
print_sample_v4(std::ostream&                              os,
                const rocprofiler_pc_sampling_record_v4_t* sample,
                const tool_agent_info*                     agent_info)
{
    print_sample_common_fields(os, sample);
    os << ", ";

    // Print cluster information
    os << "cluster_id: " << static_cast<unsigned int>(sample->cluster_id);
    if(sample->cluster_id != 0)
    {
        os << ", cluster_flat_nwg: " << static_cast<unsigned int>(sample->cluster_flat_nwg)
           << ", cluster_nwg_(x=" << static_cast<unsigned int>(sample->cluster_nwg_x)
           << ", y=" << static_cast<unsigned int>(sample->cluster_nwg_y)
           << ", z=" << static_cast<unsigned int>(sample->cluster_nwg_z) << ")"
           << ", cluster_pos_(x=" << sample->cluster_position.x
           << ", y=" << sample->cluster_position.y << ", z=" << sample->cluster_position.z << ")";
    }
    os << ", ";

    // Print snapshot_information fields (same as V2)
    print_snapshot_information(os, sample->snapshot_information);

    // Print memory counters
    auto& mc = sample->memory_counters;
    os << "mem_counters: {"
       << "load=" << static_cast<unsigned int>(mc.load_count)
       << ", store=" << static_cast<unsigned int>(mc.store_count)
       << ", ds=" << static_cast<unsigned int>(mc.ds_count)
       << ", km=" << static_cast<unsigned int>(mc.km_count)
       << ", bvh=" << static_cast<unsigned int>(mc.bvh_count)
       << ", sample=" << static_cast<unsigned int>(mc.sample_count)
       << ", async=" << static_cast<unsigned int>(mc.async_count)
       << ", tensor=" << static_cast<unsigned int>(mc.tensor_count)
       << ", xnack=" << static_cast<unsigned int>(mc.xnack_count) << "}, ";

    print_snapshot_ext_data(
        os, ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE, static_cast<const void*>(sample), agent_info);

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

/**
 * @brief Buffer callback for PC sampling records.
 *
 * The @p data parameter carries the @c tool_agent_info* for the GPU agent
 * that owns this buffer.  Creating one buffer per agent and passing the
 * agent info as client_data is the recommended policy: it gives the
 * callback direct access to the per-agent arbiter field IDs and the
 * memoized name map without having to iterate over the global agent list.
 */
void
rocprofiler_pc_sampling_callback(rocprofiler_context_id_t /*context_id*/,
                                 rocprofiler_buffer_id_t /*buffer_id*/,
                                 rocprofiler_record_header_t** headers,
                                 size_t                        num_headers,
                                 void*                         data,
                                 uint64_t                      drop_count)
{
    auto* agent_info = static_cast<const tool_agent_info*>(data);

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
                print_sample_v2(ss, pc_sample, agent_info);
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_V3_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_v3_t*>(cur_header->payload);
                print_sample_v3(ss, pc_sample);
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_V4_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_v4_t*>(cur_header->payload);
                print_sample_v4(ss, pc_sample, agent_info);
            }
            else if(cur_header->kind == ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE)
            {
                auto* pc_sample =
                    static_cast<rocprofiler_pc_sampling_record_invalid_t*>(cur_header->payload);
                print_sample_invalid(ss, pc_sample);
            }
            else
            {
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
