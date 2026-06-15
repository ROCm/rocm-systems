// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "pc_sampling_collector.h"

#include <filesystem>
#include <string>

namespace rocprofiler_compute_tool
{
PcSamplingMode parse_pc_sampling_mode(const std::string& mode);

class pc_sampling_feature_t
{
public:
    pc_sampling_feature_t() = default;
    pc_sampling_feature_t(PcSamplingMode        mode,
                          std::filesystem::path output_root,
                          std::filesystem::path code_obj_path,
                          std::filesystem::path pc_samples_path);
    pc_sampling_feature_t(PcSamplingMode               mode,
                          std::filesystem::path        output_root,
                          std::filesystem::path        code_obj_path,
                          std::filesystem::path        pc_samples_path,
                          pc_sampling_collector_t::ptr collector);

    bool enabled() const { return m_enabled; }

    PcSamplingMode mode() const { return m_mode; }

    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info);

    void append_sample(const pc_sample_record_t& record);
    void add_kernel_symbol(uint64_t           code_object_id,
                           const std::string& formatted_kernel_name,
                           uint64_t           kernel_id);
    void add_agent(const agent_record_t& agent);
    void append_kernel_dispatch(const kernel_dispatch_record_t& record);

    void finalize();

private:
    bool                         m_enabled = false;
    PcSamplingMode               m_mode    = PcSamplingMode::Disabled;
    std::filesystem::path        m_code_obj_path;
    std::filesystem::path        m_output_root;
    std::filesystem::path        m_pc_samples_path;
    pc_sampling_collector_t::ptr m_collector;
};
}  // namespace rocprofiler_compute_tool
