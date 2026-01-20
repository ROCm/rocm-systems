// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "common/defines.h"

#include "common/synchronized.hpp"
#include "core/debug.hpp"
#include "core/perfetto.hpp"
#include "core/timemory.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"

#include <timemory/operations/types.hpp>
#include <timemory/utility/types.hpp>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/name_info.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

#include <memory>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
struct counter_dispatch_record
{
    const rocprofiler_dispatch_counting_service_data_t* dispatch_data  = nullptr;
    rocprofiler_dispatch_id_t                           dispatch_id    = 0;
    rocprofiler_counter_id_t                            counter_id     = {};
    rocprofiler_record_counter_t                        record_counter = {};
};

struct counter_data_tag
{};

using counter_data_tracker = component::data_tracker<double, counter_data_tag>;
using counter_storage_type = typename counter_data_tracker::storage_type;
using counter_bundle_t     = tim::lightweight_tuple<counter_data_tracker>;
using counter_track_type   = ::perfetto::CounterTrack;

struct counter_event
{
    ROCPROFSYS_DEFAULT_OBJECT(counter_event)

    explicit counter_event(counter_dispatch_record&& _v)
    : record{ _v }
    {}

    void operator()(const client_data* tool_data, counter_track_type*,
                    const std::string& track_name, timing_interval _timing,
                    scope::config _scope) const;

    counter_dispatch_record record = {};
};

struct counter_storage
{
    const client_data*                    tool_data          = nullptr;
    uint64_t                              device_id          = 0;
    int64_t                               index              = 0;
    std::string                           metric_name        = {};
    std::string                           metric_description = {};
    std::string                           storage_name       = {};
    std::string                           track_name         = {};
    std::unique_ptr<counter_storage_type> storage            = {};
    std::unique_ptr<counter_track_type>   track              = {};

    counter_storage(const client_data* _tool_data, uint64_t _devid, size_t _idx,
                    std::string_view _name);

    ~counter_storage()                                 = default;
    counter_storage(const counter_storage&)            = delete;
    counter_storage(counter_storage&&)                 = default;
    counter_storage& operator=(const counter_storage&) = delete;
    counter_storage& operator=(counter_storage&&)      = default;

    friend bool operator<(const counter_storage& lhs, const counter_storage& rhs)
    {
        return std::tie(lhs.storage_name, lhs.device_id, lhs.index) <
               std::tie(rhs.storage_name, rhs.device_id, rhs.index);
    }

    void operator()(const counter_event& _event, timing_interval _timing,
                    scope::config _scope = scope::get_default()) const;

    static void write(counter_storage_type* storage, const std::string& metric_name,
                      const std::string& metric_description);
};
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

namespace tim
{
namespace operation
{

template <>
struct set_storage<::rocprofsys::rocprofiler_sdk::counter_data_tracker>
: public dynamic_storage_base<
      storage<::rocprofsys::rocprofiler_sdk::counter_data_tracker>*,
      ROCPROFSYS_MAX_THREADS>
{
    static constexpr size_t max_threads = ROCPROFSYS_MAX_THREADS;
    using type            = ::rocprofsys::rocprofiler_sdk::counter_data_tracker;
    using base_type       = dynamic_storage_base<storage<type>*, max_threads>;
    using storage_array_t = typename base_type::storage_array_t;

    ROCPROFSYS_DEFAULT_OBJECT(set_storage)

    auto operator()(storage<type>* _v, size_t _idx) const
    {
        base_type::ensure_capacity(get(), _idx);
        get().at(_idx) = _v;
    }
    auto operator()(type&, size_t) const {}
    auto operator()(storage<type>* _v) const
    {
        std::fill(get().begin(), get().end(), _v);
    }

    static storage_array_t& get()
    {
        static storage_array_t _v(max_threads, nullptr);
        return _v;
    }
};

template <>
struct get_storage<::rocprofsys::rocprofiler_sdk::counter_data_tracker>
{
    using type = ::rocprofsys::rocprofiler_sdk::counter_data_tracker;

    ROCPROFSYS_DEFAULT_OBJECT(get_storage)

    auto operator()(const type&) const
    {
        return operation::set_storage<type>::get().at(0);
    }

    auto operator()() const
    {
        type _obj{};
        return (*this)(_obj);
    }

    auto operator()(size_t _idx) const
    {
        // Thread-safe read using atomic capacity
        const size_t current_capacity = operation::set_storage<type>::get_capacity();
        if(_idx >= current_capacity) return static_cast<storage<type>*>(nullptr);
        return operation::set_storage<type>::get().at(_idx);
    }

    auto operator()(type&, size_t _idx) const { return (*this)(_idx); }
};
}  // namespace operation
}  // namespace tim

// Add columns for MIN, MAX, VAR, STDDEV
TIMEMORY_STATISTICS_TYPE(rocprofsys::rocprofiler_sdk::counter_data_tracker, double)
// Hide DEPTH, UNITS, and %SELF columns since they are not relevant
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_depth, rocprofiler_sdk::counter_data_tracker,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_units, rocprofiler_sdk::counter_data_tracker,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_self, rocprofiler_sdk::counter_data_tracker,
                                 false_type)
