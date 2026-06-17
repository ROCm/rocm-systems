// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/synchronized.hpp"
#include "core/perfetto.hpp"
#include "core/timemory.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"
#include <cstdint>

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
    std::uint64_t                         device_id          = 0;
    std::int64_t                          index              = 0;
    std::string                           metric_name        = {};
    std::string                           metric_description = {};
    std::string                           storage_name       = {};
    std::string                           track_name         = {};
    tim::manager::pointer_t               manager            = {};
    std::unique_ptr<counter_storage_type> storage            = {};
    std::unique_ptr<counter_track_type>   track              = {};

    counter_storage(const client_data* _tool_data, std::uint64_t _devid, size_t _idx,
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

    void write_zero(rocprofiler_timestamp_t timestamp) const;

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
{
    static constexpr size_t max_threads = 4096;
    using type            = ::rocprofsys::rocprofiler_sdk::counter_data_tracker;
    using storage_array_t = std::array<storage<type>*, max_threads>;
    friend struct get_storage<rocprofsys::rocprofiler_sdk::counter_data_tracker>;

    auto operator()(storage<type>* _v, size_t _idx) const { get().at(_idx) = _v; }
    auto operator()(type&, size_t) const {}
    auto operator()(storage<type>* _v) const { get().fill(_v); }

private:
    static storage_array_t& get()
    {
        static storage_array_t _v = { nullptr };
        return _v;
    }
};

template <>
struct get_storage<::rocprofsys::rocprofiler_sdk::counter_data_tracker>
{
    using type = ::rocprofsys::rocprofiler_sdk::counter_data_tracker;

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
