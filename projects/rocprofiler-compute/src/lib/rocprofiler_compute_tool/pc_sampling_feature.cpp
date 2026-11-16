// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_feature.h"

#include "code_object_writer.h"
#include "gsl_assert.h"
#include "sdk_callbacks.h"

using namespace rocprofiler_compute_tool;

PcSamplingMode rocprofiler_compute_tool::parse_pc_sampling_mode(const std::string& mode)
{
    if (mode == "stochastic")
        return PcSamplingMode::Stochastic;
    if (mode == "host_trap")
        return PcSamplingMode::HostTrap;
    return PcSamplingMode::Disabled;
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode        mode,
                                             std::filesystem::path code_object_info_path,
                                             std::filesystem::path source_snapshot_path)
    : pc_sampling_feature_t(mode,
                            std::move(code_object_info_path),
                            std::move(source_snapshot_path),
                            pc_sampling_collector_t::create(),
                            source_snapshotter_t::create())
{
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode               mode,
                                             std::filesystem::path        code_object_info_path,
                                             std::filesystem::path        source_snapshot_path,
                                             pc_sampling_collector_t::ptr collector,
                                             source_snapshotter_t::ptr    snapshotter)
    : pc_sampling_feature_t(mode,
                            std::move(code_object_info_path),
                            std::move(source_snapshot_path),
                            std::move(collector),
                            std::move(snapshotter),
                            std::make_shared<code_object_writer_json_t>())
{
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode               mode,
                                             std::filesystem::path        code_object_info_path,
                                             std::filesystem::path        source_snapshot_path,
                                             pc_sampling_collector_t::ptr collector,
                                             source_snapshotter_t::ptr    snapshotter,
                                             code_object_writer_t::ptr    writer)
    : m_enabled(true)
    , m_mode(mode)
    , m_code_object_info_path(std::move(code_object_info_path))
    , m_source_snapshot_path(std::move(source_snapshot_path))
    , m_collector(std::move(collector))
    , m_snapshotter(std::move(snapshotter))
    , m_writer(std::move(writer))
{
}

void pc_sampling_feature_t::on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    m_collector->on_code_object_load(info);
}

void pc_sampling_feature_t::write(tool_data_t& tool_data)
{
    if (!m_enabled)
        return;

    std::set<std::pair<uint64_t, std::string>> dispatched_symbols;
    for (const auto& dispatch : tool_data.dispatch_records)
    {
        const auto found = tool_data.kernel_symbols.find(dispatch.kernel_id);
        Expects(found != tool_data.kernel_symbols.end());
        dispatched_symbols.emplace(found->second.code_object_id, found->second.mangled_name);
    }
    finalize(dispatched_symbols);
}

void pc_sampling_feature_t::finalize(const std::set<std::pair<uint64_t, std::string>>& dispatched_symbols)
{
    m_collector->finalize(*m_writer, dispatched_symbols);
    // Processes that loaded no code objects (e.g. non-GPU launchers/forks)
    // should not leave an empty artifact behind.
    if (!m_writer->empty())
    {
        m_writer->flush(m_code_object_info_path);
        m_snapshotter->snapshot(m_collector->get_source_paths(), m_source_snapshot_path);
    }
}
