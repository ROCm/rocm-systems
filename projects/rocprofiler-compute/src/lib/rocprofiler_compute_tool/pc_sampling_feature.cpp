// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "pc_sampling_feature.h"

#include "code_object_writer.h"
#include "gsl_assert.h"

#include <memory>
#include <utility>

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
    : m_enabled(true)
    , m_mode(mode)
    , m_output_path(std::move(output_path))
    , m_translator(std::make_shared<code_object_translator_impl_t>())
    , m_collector(pc_sampling_collector_t::create(m_translator))
    , m_snapshotter(std::make_shared<source_snapshotter_impl_t>())
{
}

pc_sampling_feature_t::pc_sampling_feature_t(PcSamplingMode               mode,
                                             std::filesystem::path        output_path,
                                             code_object_translator_t::ptr translator,
                                             pc_sampling_collector_t::ptr  collector,
                                             source_snapshotter_t::ptr     snapshotter)
    : m_enabled(true)
    , m_mode(mode)
    , m_output_path(std::move(output_path))
    , m_translator(std::move(translator))
    , m_collector(std::move(collector))
    , m_snapshotter(std::move(snapshotter))
{
    Expects(m_translator);
    Expects(m_collector);
    Expects(m_snapshotter);
}

void pc_sampling_feature_t::on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    m_collector->on_code_object_load(info);
}

void pc_sampling_feature_t::finalize()
{
    code_object_writer_json_t writer;
    m_collector->write(writer);
    // Processes that loaded no code objects (e.g. non-GPU launchers/forks)
    // should not leave an empty artifact behind.
    if (!writer.empty())
    {
        writer.flush(m_output_path);
        m_snapshotter->snapshot(m_translator->get_source_paths(),
                                m_output_path.parent_path() / "sources");
    }
}
