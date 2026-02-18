// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "library/rocm.hpp"
#include "core/config.hpp"
#include "core/dynamic_library.hpp"
#include "core/gpu.hpp"
#include "library/pmc/sampler.hpp"
#include "library/rocprofiler-sdk.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"
#include "library/tracing.hpp"

#include <timemory/backends/cpu.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/utility/types.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <tuple>

#if defined(ROCPROFSYS_USE_ROCM) && ROCPROFSYS_USE_ROCM > 0
#    include <rocprofiler-sdk/rocprofiler.h>
#endif

namespace rocprofsys
{
namespace rocm
{
std::vector<hardware_counter_info>
rocm_events()
{
    return rocprofiler_sdk::get_rocm_events_info();
}
}  // namespace rocm
}  // namespace rocprofsys
