// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/kfd/timestamp_convert.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <fmt/core.h>

#include <hsa/hsa.h>

#include <atomic>
#include <cstdint>

namespace rocprofiler
{
namespace kfd
{
namespace
{
// Require this many consecutive in-tolerance samples before trusting KFD ns, and
// the per-endpoint tolerance for a sample to count. 1 us tolerance over a ~100
// sample burst keeps a bad calibration from ever flipping the switch.
constexpr uint32_t kValidationSamples = 100;
constexpr uint64_t kToleranceNs       = 1000;

// Jointly-sampled reference point mapping the GPU tick domain to CLOCK_BOOTTIME.
struct clock_anchor
{
    uint64_t gpu_tick    = 0;
    uint64_t boottime_ns = 0;
    uint64_t freq_hz     = 0;  // 0 => no valid anchor yet
};

// Anchor is written by calibrate_clock() (reader thread, periodically) and read by
// convert_gpu_ticks_to_ns() (completion path). Synchronized per house style.
common::Synchronized<clock_anchor>&
anchor()
{
    static auto*& _v = common::static_object<common::Synchronized<clock_anchor>>::construct();
    return *_v;
}

std::atomic<uint32_t>&
validation_samples()
{
    static auto*& _v = common::static_object<std::atomic<uint32_t>>::construct();
    return *_v;
}

std::atomic<bool>&
conversion_ok()
{
    static auto*& _v = common::static_object<std::atomic<bool>>::construct();
    return *_v;
}
}  // namespace

void
calibrate_clock()
{
    const auto* core = hsa::get_core_table();
    if(core == nullptr || core->hsa_system_get_info_fn == nullptr) return;

    uint64_t freq_hz = 0;
    if(core->hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &freq_hz) !=
           HSA_STATUS_SUCCESS ||
       freq_hz == 0)
        return;

    // Sample the GPU tick and the host boottime as close together as possible.
    uint64_t gpu_tick    = 0;
    uint64_t boottime_ns = common::timestamp_ns();
    if(core->hsa_system_get_info_fn(HSA_SYSTEM_INFO_TIMESTAMP, &gpu_tick) != HSA_STATUS_SUCCESS)
        return;

    anchor().wlock([&](clock_anchor& a) {
        a.gpu_tick    = gpu_tick;
        a.boottime_ns = boottime_ns;
        a.freq_hz     = freq_hz;
    });
}

uint64_t
convert_gpu_ticks_to_ns(uint64_t gpu_tick)
{
    return anchor().rlock([&](const clock_anchor& a) -> uint64_t {
        if(a.freq_hz == 0) return 0;
        // Signed delta so ticks before the anchor convert correctly.
        int64_t delta = static_cast<int64_t>(gpu_tick - a.gpu_tick);
        double  ns    = static_cast<double>(delta) / static_cast<double>(a.freq_hz) * 1e9;
        return a.boottime_ns + static_cast<uint64_t>(static_cast<int64_t>(ns));
    });
}

bool
validate_conversion(uint64_t kfd_start_ns,
                    uint64_t hsa_start_ns,
                    uint64_t kfd_end_ns,
                    uint64_t hsa_end_ns)
{
    if(conversion_ok().load(std::memory_order_acquire)) return true;

    auto     absdiff     = [](uint64_t a, uint64_t b) { return a > b ? a - b : b - a; };
    uint64_t delta_start = absdiff(kfd_start_ns, hsa_start_ns);
    uint64_t delta_end   = absdiff(kfd_end_ns, hsa_end_ns);

    if(delta_start <= kToleranceNs && delta_end <= kToleranceNs)
    {
        if(validation_samples().fetch_add(1, std::memory_order_relaxed) + 1 >= kValidationSamples)
        {
            conversion_ok().store(true, std::memory_order_release);
            ROCP_INFO << "KFD dispatch-log: timestamp conversion validated, switching to KFD "
                         "timestamps";
        }
    }
    else
    {
        // One bad sample resets the streak and re-anchors: a stale anchor (drift
        // since last calibrate) is the likely cause, so resample before retrying.
        validation_samples().store(0, std::memory_order_relaxed);
        ROCP_WARNING << fmt::format(
            "KFD dispatch-log: conversion out of tolerance (delta_start={}ns delta_end={}ns), "
            "re-calibrating",
            delta_start,
            delta_end);
        calibrate_clock();
    }
    return false;
}

bool
conversion_validated()
{
    return conversion_ok().load(std::memory_order_acquire);
}
}  // namespace kfd
}  // namespace rocprofiler
