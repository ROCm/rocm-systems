// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "code_object_writer.h"

#include <rocprofiler-sdk/rocprofiler.h>

#include <filesystem>
#include <memory>
#include <set>
#include <string_view>
#include <vector>

namespace rocprofiler_compute_tool
{

enum class PcSamplingMode : uint8_t
{
    Disabled,
    Stochastic,
    HostTrap
};

class pc_sampling_collector_t
{
public:
    using ptr = std::shared_ptr<pc_sampling_collector_t>;
    static ptr create();

    virtual ~pc_sampling_collector_t() = default;
    virtual void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) = 0;
    virtual void write(code_object_writer_t& writer) = 0;
    virtual std::set<std::filesystem::path> create_source_paths() const = 0;
};

class pc_sampling_collector_impl_t : public pc_sampling_collector_t
{
public:
    pc_sampling_collector_impl_t(code_object_translator_t::ptr translator);
    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) override;
    void write(code_object_writer_t& writer) override;
    std::set<std::filesystem::path> create_source_paths() const override;

private:
    static constexpr std::string_view source_frame_separator = " -> ";

    static bool is_source_line_token(std::string_view token);
    static std::string_view path_from_source_frame(std::string_view frame);
    static std::vector<std::filesystem::path> source_paths_from_comment(
        std::string_view comment);

    code_object_translator_t::ptr m_translator;
};
}  // namespace rocprofiler_compute_tool
