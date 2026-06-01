// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "pc_sample_record_collector.h"
#include "pc_sampling_collector.h"
#include "sdk_wrapper.h"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/fwd.h>

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

    // Path to the SDK-equivalent ps_file_results.json. The SDK consumer reads
    // exactly <workload>/ps_file_results.json, so this path carries NO pid prefix.
    void set_ps_output_path(std::filesystem::path p) { m_ps_output_path = std::move(p); }

    // PC sampling must use a context separate from counter collection.
    void set_pc_context(rocprofiler_context_id_t ctx) { m_pc_context = ctx; }

    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info);
    void finalize();

    // Query GPU agents, pick a config matching m_mode, create a buffer on the PC
    // context and configure the PC sampling service. Gracefully skips on failure.
    void configure(SdkWrapper& sdk, rocprofiler_buffer_tracing_cb_t buffer_cb, void* buffer_cb_data);

    // Start the PC sampling context (only if a buffer was actually created).
    void start(SdkWrapper& sdk);

    // Decode buffered PC sampling records and push them to the record collector.
    void on_sample_records(rocprofiler_record_header_t** headers, size_t count);

    // Stop the PC context and flush the PC buffer so all buffered samples are
    // delivered to on_sample_records before finalize_samples serializes them.
    // The SDK does not auto-flush on teardown, so without this the output file
    // would omit any samples still sitting below the buffer watermark.
    void stop_and_flush(SdkWrapper& sdk);

    // Write ps_file_results.json via ps_file_writer_json_t.
    void finalize_samples();

private:
    bool                              m_enabled = false;
    PcSamplingMode                    m_mode    = PcSamplingMode::Disabled;
    std::filesystem::path             m_output_path;
    pc_sampling_collector_t::ptr      m_collector;
    pc_sample_record_collector_t::ptr m_record_collector{};
    rocprofiler_context_id_t          m_pc_context{0};
    rocprofiler_buffer_id_t           m_pc_buffer{0};
    std::filesystem::path             m_ps_output_path{};
};
}  // namespace rocprofiler_compute_tool
