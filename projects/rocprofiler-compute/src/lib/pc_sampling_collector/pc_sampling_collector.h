// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "code_object_writer.h"

#include <rocprofiler-sdk/rocprofiler.h>

#include <memory>
#include <string>
#include <unordered_set>
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
    virtual void                     write(code_object_writer_t& writer) = 0;
    virtual std::vector<std::string> collect_source_paths()              = 0;
};

class pc_sampling_collector_impl_t : public pc_sampling_collector_t
{
public:
    pc_sampling_collector_impl_t(const std::shared_ptr<code_object_translator_t>& translator);
    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) override;
    void                     write(code_object_writer_t& writer) override;
    std::vector<std::string> collect_source_paths() override;

private:
    // Record a source path harvested from an instruction comment, preserving
    // first-seen order and de-duplicating.
    void record_source_path(const std::string& comment);

    std::shared_ptr<code_object_translator_t> m_translator;
    // Source paths harvested during write()'s disassembly walk so finalize()
    // does not need a second full traversal. collect_source_paths() falls back
    // to its own walk when write() has not run. The set de-duplicates on insert;
    // collect_source_paths() materializes a vector from it (order unspecified).
    std::unordered_set<std::string> m_source_paths;
    bool                            m_source_paths_collected = false;
};
}  // namespace rocprofiler_compute_tool
