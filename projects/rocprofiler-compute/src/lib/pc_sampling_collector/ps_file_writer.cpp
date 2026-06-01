// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "ps_file_writer.h"

#include "gsl_assert.h"

#include <fstream>
#include <iostream>
#include <utility>

using namespace rocprofiler_compute_tool;

nlohmann::json ps_file_writer_json_t::build_sample(int inst_index, const pc_sampling_record_t& rec)
{
    nlohmann::json snapshot = nlohmann::json::object();
    if (rec.stall_reason.has_value())
    {
        snapshot["stall_reason"] = rec.stall_reason.value();
    }

    nlohmann::json record = nlohmann::json::object({
        {"pc",
         nlohmann::json::object({
             {"code_object_id", rec.code_object_id},
             {"code_object_offset", rec.code_object_offset},
         })},
        {"dispatch_id", rec.dispatch_id},
        {"snapshot", std::move(snapshot)},
    });

    // Only stochastic records carry wave_issued; omit it for host-trap to match
    // the rocprofiler-sdk tool's output exactly.
    if (rec.wave_issued.has_value())
    {
        record["wave_issued"] = rec.wave_issued.value();
    }

    return nlohmann::json::object({
        {"inst_index", inst_index},
        {"record", std::move(record)},
    });
}

void ps_file_writer_json_t::add_stochastic_sample(int inst_index, const pc_sampling_record_t& rec)
{
    m_stochastic_samples.push_back(build_sample(inst_index, rec));
}

void ps_file_writer_json_t::add_host_trap_sample(int inst_index, const pc_sampling_record_t& rec)
{
    m_host_trap_samples.push_back(build_sample(inst_index, rec));
}

void ps_file_writer_json_t::set_instruction_strings(const std::vector<std::string>& instructions,
                                                    const std::vector<std::string>& comments)
{
    m_instructions = nlohmann::json::array();
    for (const auto& inst : instructions)
    {
        m_instructions.push_back(inst);
    }

    m_comments = nlohmann::json::array();
    for (const auto& comment : comments)
    {
        m_comments.push_back(comment);
    }
}

void ps_file_writer_json_t::add_kernel_symbol(uint64_t           code_object_id,
                                              const std::string& formatted_kernel_name)
{
    m_kernel_symbols.push_back(nlohmann::json::object({
        {"code_object_id", code_object_id},
        {"formatted_kernel_name", formatted_kernel_name},
    }));
}

std::string ps_file_writer_json_t::get_result()
{
    nlohmann::json tool_entry = nlohmann::json::object({
        {"buffer_records",
         nlohmann::json::object({
             {"pc_sample_stochastic", m_stochastic_samples},
             {"pc_sample_host_trap", m_host_trap_samples},
         })},
        {"strings",
         nlohmann::json::object({
             {"pc_sample_instructions", m_instructions},
             {"pc_sample_comments", m_comments},
         })},
        {"kernel_symbols", m_kernel_symbols},
    });

    return nlohmann::json{{"rocprofiler-sdk-tool", nlohmann::json::array({std::move(tool_entry)})}}.dump();
}

void ps_file_writer_json_t::flush(const std::filesystem::path& output_file_path)
{
    Expects(!output_file_path.empty());
    create_parent_dir(output_file_path);

    std::ofstream out_file(output_file_path, std::ios::out);
    if (!out_file.is_open())
    {
        std::cerr << "Failed to open output file: " << output_file_path << "\n";
        return;
    }
    out_file << get_result();
    std::clog << "[rocprofiler-compute] [" << __FUNCTION__
              << "] PC sampling data has been written to: " << output_file_path << "\n";
}

void ps_file_writer_json_t::create_parent_dir(const std::filesystem::path& output_file_path)
{
    Expects(output_file_path.has_parent_path());
    std::error_code error;
    std::filesystem::create_directories(output_file_path.parent_path(), error);
    if (error)
    {
        throw std::runtime_error("Failed to create output directory: " + output_file_path.string() +
                                 ", error: " + error.message());
    }
}
