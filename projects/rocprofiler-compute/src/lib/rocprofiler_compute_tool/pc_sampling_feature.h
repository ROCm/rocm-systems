// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "pc_record_store.h"
#include "pc_sampling_collector.h"

#include <rocprofiler-sdk/rocprofiler.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

namespace rocprofiler_compute_tool
{
class SdkWrapper;

PcSamplingMode parse_pc_sampling_mode(const std::string& mode);

class pc_sampling_feature_t
{
public:
    pc_sampling_feature_t() = default;
    pc_sampling_feature_t(PcSamplingMode        mode,
                          uint64_t              interval,
                          std::string           unit,
                          std::filesystem::path code_obj_info_path,
                          std::filesystem::path ps_file_path);
    pc_sampling_feature_t(PcSamplingMode               mode,
                          uint64_t                     interval,
                          std::string                  unit,
                          std::filesystem::path        code_obj_info_path,
                          std::filesystem::path        ps_file_path,
                          pc_sampling_collector_t::ptr collector);

    bool enabled() const { return m_enabled; }

    PcSamplingMode mode() const { return m_mode; }

    const std::filesystem::path& output_path() const { return m_output_path; }

    // Configure the SDK PC sampling service for the requested mode. Returns false
    // (no SDK service configured, drains nothing) when m_mode == Disabled, no
    // agent supports the requested method/unit, or PC sampling is unavailable on
    // this runtime (the SDK seam throws, e.g. NOT_IMPLEMENTED). Never throws, so
    // a missing PC sampling capability degrades gracefully instead of aborting
    // the profiled application. Caller logs on false.
    bool configure(rocprofiler_context_id_t ctx, SdkWrapper& sdk, void* buffer_callback_user_data);

    // Buffer callback fan-in: decode PC sample records and append to the store.
    void on_pc_sample_records(rocprofiler_record_header_t** headers, size_t num_headers);

    // Drain the LOSSLESS delivery buffer so any samples still below the
    // watermark are handed to on_pc_sample_records() before finalize()
    // serializes the store. No-op when configure() set up no buffer (disabled
    // mode or no matching agent). Must run after the context is stopped and
    // before finalize().
    void flush(SdkWrapper& sdk);

    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info);
    void finalize();

private:
    // Body of configure(); may throw from the SDK seam. configure() wraps this
    // in a try/catch so no exception escapes to the SDK C-callback boundary.
    bool try_configure(rocprofiler_context_id_t ctx, SdkWrapper& sdk, void* buffer_callback_user_data);

    bool                               m_enabled = false;
    PcSamplingMode                     m_mode    = PcSamplingMode::Disabled;
    uint64_t                           m_interval{0};
    std::string                        m_unit{};
    std::filesystem::path              m_output_path;   ///< code_obj_info.json path
    std::filesystem::path              m_ps_file_path;  ///< ps_file_results.json path
    pc_sampling_collector_t::ptr       m_collector;
    std::shared_ptr<pc_record_store_t> m_record_store;
    rocprofiler_buffer_id_t            m_buffer_id{0};
};
}  // namespace rocprofiler_compute_tool
