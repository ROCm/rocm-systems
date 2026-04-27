// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <memory>
#include <rocprofiler-sdk/rocprofiler.h>

namespace rocprofiler_compute_tool
{
class pc_sampling_collector_t
{
public:
    using Ptr                          = std::shared_ptr<pc_sampling_collector_t>;
    virtual ~pc_sampling_collector_t() = default;
    virtual void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) = 0;
};
}  // namespace rocprofiler_compute_tool
