// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "nlohmann/json.hpp"
#include "pc_sampling_record.h"

#include <filesystem>
#include <string>
#include <vector>

namespace rocprofiler_compute_tool
{
class ps_file_writer_t
{
public:
    virtual ~ps_file_writer_t()                                                         = default;
    virtual void add_stochastic_sample(int inst_index, const pc_sampling_record_t& rec) = 0;
    virtual void add_host_trap_sample(int inst_index, const pc_sampling_record_t& rec)  = 0;
    virtual void set_instruction_strings(const std::vector<std::string>& instructions,
                                         const std::vector<std::string>& comments)      = 0;
    virtual void add_kernel_symbol(uint64_t code_object_id, const std::string& formatted_kernel_name) = 0;
    virtual std::string get_result()                                         = 0;
    virtual void        flush(const std::filesystem::path& output_file_path) = 0;
};

class ps_file_writer_json_t : public ps_file_writer_t
{
public:
    void add_stochastic_sample(int inst_index, const pc_sampling_record_t& rec) override;
    void add_host_trap_sample(int inst_index, const pc_sampling_record_t& rec) override;
    void set_instruction_strings(const std::vector<std::string>& instructions,
                                 const std::vector<std::string>& comments) override;
    void add_kernel_symbol(uint64_t code_object_id, const std::string& formatted_kernel_name) override;
    std::string get_result() override;
    void        flush(const std::filesystem::path& output_file_path) override;

private:
    static void           create_parent_dir(const std::filesystem::path& output_file_path);
    static nlohmann::json build_sample(int inst_index, const pc_sampling_record_t& rec);

    nlohmann::json::array_t m_stochastic_samples = nlohmann::json::array();
    nlohmann::json::array_t m_host_trap_samples  = nlohmann::json::array();
    nlohmann::json::array_t m_instructions       = nlohmann::json::array();
    nlohmann::json::array_t m_comments           = nlohmann::json::array();
    nlohmann::json::array_t m_kernel_symbols     = nlohmann::json::array();
};
}  // namespace rocprofiler_compute_tool
