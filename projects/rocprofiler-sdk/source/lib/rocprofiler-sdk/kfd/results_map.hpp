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

#pragma once

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/kfd/correlation_types.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <unordered_map>

// ResultsMap: the handoff from the KFD reader thread to the completion path.
// The reader pairs a firmware start+eop record and deposits raw GPU ticks keyed
// by correlation_key; get_dispatch_time() takes them (converting ticks->ns
// there, so validation can compare against the HSA result on the same call).
// evict_stale() reclaims entries the completion path never took (dispatch
// completed via HSA fallback before its record arrived) to prevent leaks.

namespace rocprofiler
{
namespace kfd
{
// Raw firmware timing for one dispatch. Ticks are converted to CLOCK_BOOTTIME
// ns later (in get_dispatch_time); deposited_at_ns is a host CLOCK_BOOTTIME
// stamp used only for stale eviction.
struct kfd_timing_result
{
    uint64_t start_gpu_ticks = 0;
    uint64_t end_gpu_ticks   = 0;
    uint64_t deposited_at_ns = 0;
};

class ResultsMap
{
public:
    ResultsMap()  = default;
    ~ResultsMap() = default;

    ResultsMap(const ResultsMap&)     = delete;
    ResultsMap(ResultsMap&&) noexcept = delete;
    ResultsMap& operator=(const ResultsMap&) = delete;
    ResultsMap& operator=(ResultsMap&&) noexcept = delete;

    // Deposit a paired result (KFD reader thread). Overwrites any existing entry.
    void deposit(const correlation_key& key, const kfd_timing_result& result);

    // Atomically find + erase. nullopt if not present.
    std::optional<kfd_timing_result> take(const correlation_key& key);

    // Remove entries whose deposited_at_ns is older than max_age_ns relative to
    // now_ns. now_ns is passed in (not sampled) so the function is deterministic
    // and unit-testable. Returns the number of entries evicted.
    size_t evict_stale(uint64_t now_ns, uint64_t max_age_ns);

    // Current entry count (diagnostics/tests).
    size_t size() const;

private:
    using map_t = std::unordered_map<correlation_key, kfd_timing_result, correlation_key_hash>;
    common::Synchronized<map_t> m_data = {};
};
}  // namespace kfd
}  // namespace rocprofiler
