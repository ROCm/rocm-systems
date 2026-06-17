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
    pc_sampling_feature_t(PcSamplingMode mode, std::filesystem::path output_path);
    pc_sampling_feature_t(PcSamplingMode               mode,
                          std::filesystem::path        output_path,
                          pc_sampling_collector_t::ptr collector);

    bool enabled() const { return m_enabled; }

    PcSamplingMode mode() const { return m_mode; }

    const std::filesystem::path& output_path() const { return m_output_path; }

    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info);
    void finalize();

private:
    bool                         m_enabled = false;
    PcSamplingMode               m_mode    = PcSamplingMode::Disabled;
    std::filesystem::path        m_output_path;
    pc_sampling_collector_t::ptr m_collector;
};
}  // namespace rocprofiler_compute_tool
