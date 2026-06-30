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
    static ptr create(code_object_translator_t::ptr translator);

    virtual ~pc_sampling_collector_t() = default;
    virtual void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) = 0;
    virtual void write(code_object_writer_t& writer) = 0;
};

class pc_sampling_collector_impl_t : public pc_sampling_collector_t
{
public:
    pc_sampling_collector_impl_t(code_object_translator_t::ptr translator);
    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) override;
    void write(code_object_writer_t& writer) override;

private:
    code_object_translator_t::ptr m_translator;
};
}  // namespace rocprofiler_compute_tool
