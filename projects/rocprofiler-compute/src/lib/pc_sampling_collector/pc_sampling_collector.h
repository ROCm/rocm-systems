// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <rocprofiler-sdk/rocprofiler.h>

#include <memory>

namespace rocm_compute
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
    using Ptr                          = std::shared_ptr<pc_sampling_collector_t>;
    virtual ~pc_sampling_collector_t() = default;
    virtual void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) = 0;
};

class pc_sampling_collector_impl_t : public pc_sampling_collector_t
{
public:
    pc_sampling_collector_impl_t();
    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) override;
};
}  // namespace rocm_compute
