// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_feature.h"

#include "code_object_writer.h"

using namespace rocprofiler_compute_tool;

PcSamplingMode rocprofiler_compute_tool::parse_pc_sampling_mode(const std::string& mode)
{
    if (mode == "stochastic")
        return PcSamplingMode::Stochastic;
    if (mode == "host_trap")
        return PcSamplingMode::HostTrap;
    return PcSamplingMode::Disabled;
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode mode, std::filesystem::path output_path)
    : pc_sampling_feature_t(mode, std::move(output_path), pc_sampling_collector_t::create())
{
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode               mode,
                                             std::filesystem::path        output_path,
                                             pc_sampling_collector_t::ptr collector)
    : m_enabled(true)
    , m_mode(mode)
    , m_output_path(std::move(output_path))
    , m_collector(std::move(collector))
{
}

void pc_sampling_feature_t::on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    m_collector->on_code_object_load(info);
}

void pc_sampling_feature_t::finalize()
{
    code_object_writer_json_t writer;
    m_collector->write(writer);
    writer.flush(m_output_path);
}
