// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "counter_config_manager.hpp"
#include "types.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace python
{
/**
 * @brief Thread-safe collector for counter records
 *
 * This class collects counter records from rocprofiler buffer callbacks
 * and stores them in a thread-safe manner for later retrieval.
 */
class RecordCollector
{
public:
    RecordCollector();
    ~RecordCollector() = default;

    // Non-copyable, non-movable
    RecordCollector(const RecordCollector&) = delete;
    RecordCollector& operator=(const RecordCollector&) = delete;
    RecordCollector(RecordCollector&&)                 = delete;
    RecordCollector& operator=(RecordCollector&&) = delete;

    /**
     * @brief Add a dispatch header record
     * @param record The dispatch counting service record
     * @param kernel_name The resolved kernel name
     */
    void add_dispatch_header(const rocprofiler_dispatch_counting_service_record_t& record,
                             const std::string&                                    kernel_name);

    /**
     * @brief Add a counter value record
     * @param record The counter record from the buffer
     * @param config_mgr The config manager for dimension lookup
     */
    void add_counter_record(const rocprofiler_counter_record_t& record,
                            CounterConfigManager&               config_mgr);

    /**
     * @brief Get all collected counter records
     * @return Vector of counter records
     */
    std::vector<CounterRecord> get_records() const;

    /**
     * @brief Clear all collected records
     */
    void clear();

    /**
     * @brief Get the number of collected records
     */
    size_t record_count() const;

private:
    mutable std::mutex mutex_;

    // Dispatch ID -> DispatchInfo for correlating counter records
    std::unordered_map<uint64_t, DispatchInfo> dispatches_;

    // All collected counter records
    std::vector<CounterRecord> records_;
};

}  // namespace python
}  // namespace rocprofiler
