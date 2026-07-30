// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_writer.h"
#include "pc_sampling_collector.h"
#include "source_snapshotter.h"

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
                          std::filesystem::path code_object_info_path,
                          std::filesystem::path source_snapshot_path);
    pc_sampling_feature_t(PcSamplingMode               mode,
                          std::filesystem::path        code_object_info_path,
                          std::filesystem::path        source_snapshot_path,
                          pc_sampling_collector_t::ptr collector,
                          source_snapshotter_t::ptr    snapshotter);
    pc_sampling_feature_t(PcSamplingMode               mode,
                          std::filesystem::path        code_object_info_path,
                          std::filesystem::path        source_snapshot_path,
                          pc_sampling_collector_t::ptr collector,
                          source_snapshotter_t::ptr    snapshotter,
                          code_object_writer_t::ptr    writer);

    bool enabled() const { return m_enabled; }

    PcSamplingMode mode() const { return m_mode; }

    const std::filesystem::path& code_object_info_path() const { return m_code_object_info_path; }

    const std::filesystem::path& source_snapshot_path() const { return m_source_snapshot_path; }

    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info);
    void finalize();

private:
    bool                         m_enabled = false;
    PcSamplingMode               m_mode    = PcSamplingMode::Disabled;
    std::filesystem::path        m_code_object_info_path;
    std::filesystem::path        m_source_snapshot_path;
    pc_sampling_collector_t::ptr m_collector;
    source_snapshotter_t::ptr    m_snapshotter;
    code_object_writer_t::ptr    m_writer;
};
}  // namespace rocprofiler_compute_tool
