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

#include <rocprofiler-sdk/fwd.h>

#include <cstdint>
#include <optional>
#include <unordered_map>

// CorrelationTable: links an in-flight dispatch (keyed by correlation_key) to
// the SDK metadata needed to emit its record. Written at enqueue time (capture
// phase); consumed/erased at completion time inside get_dispatch_time().

namespace rocprofiler
{
namespace kfd
{
// SDK-side metadata carried alongside a dispatch until completion.
struct correlation_entry
{
    rocprofiler_dispatch_id_t sdk_dispatch_id = 0;
    rocprofiler_kernel_id_t   kernel_id       = 0;
    rocprofiler_queue_id_t    queue_id        = {};
    rocprofiler_timestamp_t   enqueue_ts      = 0;
};

class CorrelationTable
{
public:
    CorrelationTable()  = default;
    ~CorrelationTable() = default;

    CorrelationTable(const CorrelationTable&)     = delete;
    CorrelationTable(CorrelationTable&&) noexcept = delete;
    CorrelationTable& operator=(const CorrelationTable&) = delete;
    CorrelationTable& operator=(CorrelationTable&&) noexcept = delete;

    // Insert at enqueue time. Overwrites any existing entry for the same key.
    void insert(const correlation_key& key, const correlation_entry& entry);

    // Atomically find + erase. nullopt if the key is not present.
    std::optional<correlation_entry> take(const correlation_key& key);

    // Unconditional erase. Idempotent: erasing a missing key is a no-op. Called
    // from the completion path even when the HSA fallback is used, so no entry
    // leaks past AsyncSignalHandler.
    void erase(const correlation_key& key);

    // Current number of in-flight entries (diagnostics/tests).
    size_t size() const;

private:
    using map_t = std::unordered_map<correlation_key, correlation_entry, correlation_key_hash>;
    common::Synchronized<map_t> m_data = {};
};
}  // namespace kfd
}  // namespace rocprofiler
