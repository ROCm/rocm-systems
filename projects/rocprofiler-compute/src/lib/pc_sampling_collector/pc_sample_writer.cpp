// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#ifndef ROCPROFILER_SDK_EXPERIMENTAL
#    define ROCPROFILER_SDK_EXPERIMENTAL
#endif

#include "pc_sample_writer.h"

#include "file_writer.h"
#include "nlohmann/json.hpp"

#include <rocprofiler-sdk/pc_sampling.h>

#include <iostream>
#include <memory>

using namespace rocprofiler_compute_tool;

namespace
{
std::string instruction_type_name(uint32_t inst_type)
{
    const char* name = rocprofiler_get_pc_sampling_instruction_type_name(
        static_cast<rocprofiler_pc_sampling_instruction_type_t>(inst_type));
    return name != nullptr ? std::string{name} : std::string{};
}

std::string not_issued_reason_name(uint32_t reason)
{
    const char* name = rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
        static_cast<rocprofiler_pc_sampling_instruction_not_issued_reason_t>(reason));
    return name != nullptr ? std::string{name} : std::string{};
}

nlohmann::json hw_id_to_json(const pc_sample_hw_id_t& hw)
{
    return nlohmann::json::object({
        {"chiplet", hw.chiplet},
        {"wave_id", hw.wave_id},
        {"simd_id", hw.simd_id},
        {"pipe_id", hw.pipe_id},
        {"cu_or_wgp_id", hw.cu_or_wgp_id},
        {"shader_array_id", hw.shader_array_id},
        {"shader_engine_id", hw.shader_engine_id},
        {"workgroup_id", hw.workgroup_id},
        {"vm_id", hw.vm_id},
        {"queue_id", hw.queue_id},
        {"microengine_id", hw.microengine_id},
    });
}

nlohmann::json pc_to_json(const pc_sample_pc_t& pc)
{
    return nlohmann::json::object({
        {"code_object_id", pc.code_object_id},
        {"code_object_offset", pc.code_object_offset},
    });
}

nlohmann::json corr_to_json(const pc_sample_correlation_t& corr)
{
    return nlohmann::json::object({
        {"internal", corr.internal},
        {"external", corr.external},
    });
}

nlohmann::json dim3_to_json(const pc_sample_dim3_t& d)
{
    return nlohmann::json::object({
        {"x", d.x},
        {"y", d.y},
        {"z", d.z},
    });
}

nlohmann::json snapshot_to_json(const pc_sample_snapshot_t& s)
{
    auto out = nlohmann::json::object();
    out["stall_reason"] = not_issued_reason_name(s.stall_reason);
    // Emit the plain uint32 fields from the single-sourced field list.
#define PC_SAMPLE_SNAPSHOT_EMIT(field) out[#field] = s.field;
    PC_SAMPLE_SNAPSHOT_PLAIN_FIELDS(PC_SAMPLE_SNAPSHOT_EMIT)
#undef PC_SAMPLE_SNAPSHOT_EMIT
    return out;
}

// Host-trap samples emit exactly these fields; stochastic adds more on top.
nlohmann::json common_record_to_json(const pc_sample_record_t& r)
{
    return nlohmann::json::object({
        {"hw_id", hw_id_to_json(r.hw_id)},
        {"pc", pc_to_json(r.pc)},
        {"exec_mask", r.exec_mask},
        {"timestamp", r.timestamp},
        {"dispatch_id", r.dispatch_id},
        {"corr_id", corr_to_json(r.corr_id)},
        {"wrkgrp_id", dim3_to_json(r.wrkgrp_id)},
        {"wave_in_grp", r.wave_in_grp},
    });
}

nlohmann::json stochastic_record_to_json(const pc_sample_record_t& r)
{
    auto out           = common_record_to_json(r);
    out["flags"]       = nlohmann::json::object({{"has_mem_cnt", r.flags.has_mem_cnt}});
    out["wave_issued"] = r.wave_issued;
    out["inst_type"]   = instruction_type_name(r.inst_type);
    out["wave_cnt"]    = r.wave_cnt;
    out["snapshot"]    = snapshot_to_json(r.snapshot);
    return out;
}
}  // namespace

std::shared_ptr<pc_sample_writer_t> pc_sample_writer_t::create()
{
    return std::make_shared<pc_sample_writer_json_t>();
}

void pc_sample_writer_json_t::begin()
{
    m_pid = 0;
    m_stochastic.clear();
    m_host_trap.clear();
    m_instructions.clear();
    m_comments.clear();
    m_kernel_symbols.clear();
    m_agents.clear();
    m_kernel_dispatches.clear();
}

void pc_sample_writer_json_t::append_stochastic(const pc_sample_record_t& r, size_t inst_index)
{
    m_stochastic.push_back(sample_entry_t{r, inst_index});
}

void pc_sample_writer_json_t::append_host_trap(const pc_sample_record_t& r, size_t inst_index)
{
    m_host_trap.push_back(sample_entry_t{r, inst_index});
}

void pc_sample_writer_json_t::set_strings(const pc_string_table_t& table)
{
    const auto& instructions = table.instructions();
    const auto& comments     = table.comments();
    m_instructions.assign(instructions.begin(), instructions.end());
    m_comments.assign(comments.begin(), comments.end());
}

void pc_sample_writer_json_t::set_kernel_symbols(const std::vector<kernel_symbol_entry_t>& syms)
{
    m_kernel_symbols = syms;
}

void pc_sample_writer_json_t::set_agents(const std::vector<agent_record_t>& agents)
{
    m_agents = agents;
}

void pc_sample_writer_json_t::set_kernel_dispatches(const std::vector<kernel_dispatch_record_t>& dispatches)
{
    m_kernel_dispatches = dispatches;
}

void pc_sample_writer_json_t::set_metadata(int pid)
{
    m_pid = pid;
}

std::string pc_sample_writer_json_t::get_result()
{
    auto stochastic_records = nlohmann::json::array();
    for (const auto& e : m_stochastic)
    {
        stochastic_records.push_back(nlohmann::json::object({
            {"record", stochastic_record_to_json(e.record)},
            {"inst_index", e.inst_index},
        }));
    }

    auto host_trap_records = nlohmann::json::array();
    for (const auto& e : m_host_trap)
    {
        host_trap_records.push_back(nlohmann::json::object({
            {"record", common_record_to_json(e.record)},
            {"inst_index", e.inst_index},
        }));
    }

    auto kernel_symbols = nlohmann::json::array();
    for (const auto& s : m_kernel_symbols)
    {
        kernel_symbols.push_back(nlohmann::json::object({
            {"code_object_id", s.code_object_id},
            {"formatted_kernel_name", s.formatted_kernel_name},
            {"kernel_id", s.kernel_id},
        }));
    }

    auto agents = nlohmann::json::array();
    for (const auto& a : m_agents)
    {
        agents.push_back(nlohmann::json::object({
            {"size", a.size},
            {"id", nlohmann::json::object({{"handle", a.id_handle}})},
            {"type", a.type},
            {"node_id", a.node_id},
            {"logical_node_id", a.logical_node_id},
            {"cu_count", a.cu_count},
            {"gpu_id", a.gpu_id},
            {"wave_front_size", a.wave_front_size},
            {"simd_count", a.simd_count},
        }));
    }

    auto kernel_dispatch = nlohmann::json::array();
    for (const auto& d : m_kernel_dispatches)
    {
        kernel_dispatch.push_back(nlohmann::json::object({
            {"size", d.size},
            {"kind", d.kind},
            {"operation", d.operation},
            {"thread_id", d.thread_id},
            {"correlation_id",
             nlohmann::json::object({
                 {"internal", d.corr_internal},
                 {"external", d.corr_external},
             })},
            {"start_timestamp", d.start_timestamp},
            {"end_timestamp", d.end_timestamp},
            {"dispatch_info",
             nlohmann::json::object({
                 {"size", d.dispatch_info_size},
                 {"agent_id", nlohmann::json::object({{"handle", d.agent_id_handle}})},
                 {"queue_id", nlohmann::json::object({{"handle", d.queue_id_handle}})},
                 {"kernel_id", d.kernel_id},
                 {"dispatch_id", d.dispatch_id},
                 {"private_segment_size", d.private_segment_size},
                 {"group_segment_size", d.group_segment_size},
                 {"workgroup_size", dim3_to_json(d.workgroup_size)},
                 {"grid_size", dim3_to_json(d.grid_size)},
             })},
        }));
    }

    auto entry = nlohmann::json::object({
        {"metadata", nlohmann::json::object({{"pid", m_pid}})},
        {"agents", std::move(agents)},
        {"counters", nlohmann::json::array()},
        {"summary", nlohmann::json::array()},
        {"host_functions", nlohmann::json::array()},
        {"callback_records", nlohmann::json::object()},
        {"code_objects", nlohmann::json::array()},
        {"kernel_symbols", std::move(kernel_symbols)},
        {"strings",
         nlohmann::json::object({
             {"pc_sample_instructions", m_instructions},
             {"pc_sample_comments", m_comments},
             {"code_object_snapshot_filenames", nlohmann::json::array()},
             {"att_filenames", nlohmann::json::array()},
         })},
        {"buffer_records",
         nlohmann::json::object({
             {"pc_sample_stochastic", std::move(stochastic_records)},
             {"pc_sample_host_trap", std::move(host_trap_records)},
             {"kernel_dispatch", std::move(kernel_dispatch)},
             {"hip_api", nlohmann::json::array()},
             {"hsa_api", nlohmann::json::array()},
             {"memory_copy", nlohmann::json::array()},
             {"marker_api", nlohmann::json::array()},
             {"rccl_api", nlohmann::json::array()},
             {"memory_allocation", nlohmann::json::array()},
             {"scratch_memory", nlohmann::json::array()},
             {"kfd", nlohmann::json::array()},
             {"rocdecode_api", nlohmann::json::array()},
             {"rocjpeg_api", nlohmann::json::array()},
         })},
    });

    auto root = nlohmann::json::object({
        {"rocprofiler-sdk-tool", nlohmann::json::array({std::move(entry)})},
    });

    return root.dump();
}

void pc_sample_writer_json_t::flush(const std::filesystem::path& output_file_path)
{
    file_writer_t::create()->write(output_file_path, get_result());
    std::clog << "[rocprofiler-compute] [" << __FUNCTION__
              << "] PC sampling data has been written to: " << output_file_path << "\n";
}
