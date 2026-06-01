// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"
#include "code_object_writer.h"

#include <rocprofiler-sdk/rocprofiler.h>

#include <memory>

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
    // Share an existing translator so a single decode of each code object can be
    // reused by both the code-object writer and the PC-sample collector.
    static ptr create(const std::shared_ptr<code_object_translator_t>& translator);

    virtual ~pc_sampling_collector_t() = default;
    virtual void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) = 0;
    virtual void write(code_object_writer_t& writer) = 0;
    // The shared translator, so collaborators can decode the same code objects.
    virtual std::shared_ptr<code_object_translator_t> translator() const = 0;
};

class pc_sampling_collector_impl_t : public pc_sampling_collector_t
{
public:
    pc_sampling_collector_impl_t(const std::shared_ptr<code_object_translator_t>& translator);
    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) override;
    void write(code_object_writer_t& writer) override;

    std::shared_ptr<code_object_translator_t> translator() const override { return m_translator; }

private:
    std::shared_ptr<code_object_translator_t> m_translator;
};
}  // namespace rocprofiler_compute_tool
