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

#include "record_collector.hpp"

#include <rocprofiler-sdk/counters.h>

namespace rocprofiler
{
namespace python
{
RecordCollector::RecordCollector() = default;

void
RecordCollector::add_dispatch_header(const rocprofiler_dispatch_counting_service_record_t& record,
                                     const std::string& kernel_name)
{
    std::lock_guard lock(mutex_);

    DispatchInfo info;
    info.dispatch_id    = record.dispatch_info.dispatch_id;
    info.kernel_id      = record.dispatch_info.kernel_id;
    info.correlation_id = record.correlation_id.internal;
    info.queue_id       = record.dispatch_info.queue_id.handle;
    info.agent_id       = record.dispatch_info.agent_id.handle;
    info.kernel_name    = kernel_name;

    dispatches_[info.dispatch_id] = info;
}

void
RecordCollector::add_counter_record(const rocprofiler_counter_record_t& record,
                                    CounterConfigManager&               config_mgr)
{
    std::lock_guard lock(mutex_);

    CounterRecord counter_rec;
    counter_rec.dispatch_id = record.dispatch_id;
    counter_rec.value       = record.counter_value;

    // Get counter ID from record ID
    rocprofiler_counter_id_t counter_id = {.handle = 0};
    rocprofiler_query_record_counter_id(record.id, &counter_id);
    counter_rec.counter_id = counter_id.handle;

    // Get counter name
    try
    {
        auto info                = CounterConfigManager::get_counter_info(counter_id);
        counter_rec.counter_name = info.name;
    } catch(...)
    {
        counter_rec.counter_name = "<unknown>";
    }

    // Get kernel name and agent from dispatch info
    auto dispatch_it = dispatches_.find(record.dispatch_id);
    if(dispatch_it != dispatches_.end())
    {
        counter_rec.kernel_name = dispatch_it->second.kernel_name;
        counter_rec.agent_id    = dispatch_it->second.agent_id;
    }
    else
    {
        counter_rec.kernel_name = "<unknown>";
        counter_rec.agent_id    = 0;
    }

    // Get dimension information
    auto dimensions = config_mgr.get_dimensions(counter_id);
    for(const auto& dim : dimensions)
    {
        size_t pos = 0;
        rocprofiler_query_record_dimension_position(record.id, dim.id, &pos);
        counter_rec.dimensions.emplace_back(dim.name, pos);
    }

    records_.push_back(std::move(counter_rec));
}

std::vector<CounterRecord>
RecordCollector::get_records() const
{
    std::lock_guard lock(mutex_);
    return records_;
}

void
RecordCollector::clear()
{
    std::lock_guard lock(mutex_);
    dispatches_.clear();
    records_.clear();
}

size_t
RecordCollector::record_count() const
{
    std::lock_guard lock(mutex_);
    return records_.size();
}

}  // namespace python
}  // namespace rocprofiler
