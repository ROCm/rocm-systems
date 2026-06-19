// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/rocprofiler_sdk/wrapper.hpp"
#include <cstdint>
// All rocprofiler-sdk headers come transitively through fwd.hpp -> backend shim.
#include "common/synchronized.hpp"
#include "core/agent_manager.hpp"
#include "core/demangler.hpp"
#include "core/perfetto.hpp"
#include "core/timemory.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"
#include "logger/debug.hpp"

#include <timemory/utility/types.hpp>

#include <memory>
#include <regex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler_sdk
{

// ─── counter_dispatch_record<Wrapper> ────────────────────────────────────────

template <typename Wrapper>
struct counter_dispatch_record
{
    const typename Wrapper::dispatch_counting_data* dispatch_data  = nullptr;
    typename Wrapper::dispatch_id_t                 dispatch_id    = 0;
    typename Wrapper::counter_id                    counter_id     = {};
    typename Wrapper::counter_record                record_counter = {};
};

struct counter_data_tag
{};

using counter_data_tracker = component::data_tracker<double, counter_data_tag>;
using counter_storage_type = typename counter_data_tracker::storage_type;
using counter_bundle_t     = tim::lightweight_tuple<counter_data_tracker>;
using counter_track_type   = ::perfetto::CounterTrack;

// ─── counter_event<Wrapper> ──────────────────────────────────────────────────

template <typename Wrapper>
struct counter_event
{
    explicit counter_event(counter_dispatch_record<Wrapper>&& _v)
    : record{ std::move(_v) }
    {}

    void operator()(const client_data<Wrapper>* tool_data, counter_track_type*,
                    const std::string& track_name, timing_interval<Wrapper> _timing,
                    scope::config _scope) const;

    counter_dispatch_record<Wrapper> record = {};
};

// ─── counter_storage<Wrapper> ────────────────────────────────────────────────

template <typename Wrapper>
struct counter_storage
{
    const client_data<Wrapper>*           tool_data          = nullptr;
    std::uint64_t                         device_id          = 0;
    std::int64_t                          index              = 0;
    std::string                           metric_name        = {};
    std::string                           metric_description = {};
    std::string                           storage_name       = {};
    std::string                           track_name         = {};
    tim::manager::pointer_t               manager            = {};
    std::unique_ptr<counter_storage_type> storage            = {};
    std::unique_ptr<counter_track_type>   track              = {};

    counter_storage(const client_data<Wrapper>* _tool_data, std::uint64_t _devid,
                    size_t _idx, std::string_view _name);

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

    void operator()(const counter_event<Wrapper>& _event,
                    timing_interval<Wrapper>      _timing,
                    scope::config                 _scope = scope::get_default()) const;

    void write_zero(typename Wrapper::timestamp_t timestamp) const;

    static void write(counter_storage_type* storage, const std::string& metric_name,
                      const std::string& metric_description);

private:
    static std::string get_counter_description(const client_data<Wrapper>* tool_data,
                                               std::string_view            _v);
    static void        metadata_initialize_counter_category();
    static void        metadata_initialize_counter_track(const char* name);
    static void metadata_initialize_counters_pmc(size_t dev_id, const std::string& name,
                                                 const std::string& metric_description);
};

// ─── Production aliases ───────────────────────────────────────────────────────

using counter_dispatch_record_t = counter_dispatch_record<backend>;
using counter_event_t           = counter_event<backend>;
using counter_storage_t         = counter_storage<backend>;

}  // namespace rocprofiler_sdk
}  // namespace rocprofsys

// ─── timemory trait specialisations ──────────────────────────────────────────

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

TIMEMORY_STATISTICS_TYPE(rocprofsys::rocprofiler_sdk::counter_data_tracker, double)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_depth, rocprofiler_sdk::counter_data_tracker,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_units, rocprofiler_sdk::counter_data_tracker,
                                 false_type)
ROCPROFSYS_DEFINE_CONCRETE_TRAIT(report_self, rocprofiler_sdk::counter_data_tracker,
                                 false_type)

// ─── Template implementations ────────────────────────────────────────────────

namespace rocprofsys::rocprofiler_sdk
{

// ─── counter_storage private helpers ─────────────────────────────────────────

template <typename Wrapper>
std::string
counter_storage<Wrapper>::get_counter_description(const client_data<Wrapper>* tool_data,
                                                  std::string_view            _v)
{
    const auto& _info = tool_data->events_info;
    for(const auto& itr : _info)
    {
        if(itr.symbol().find(_v) == 0 || itr.short_description().find(_v) == 0)
            return itr.long_description();
    }
    return std::string{};
}

template <typename Wrapper>
void
counter_storage<Wrapper>::metadata_initialize_counter_category()
{
    trace_cache::get_metadata_registry().add_string(
        trait::name<category::rocm_counter_collection>::value);
}

template <typename Wrapper>
void
counter_storage<Wrapper>::metadata_initialize_counter_track(const char* name)
{
    trace_cache::get_metadata_registry().add_track({ name, std::nullopt, "{}" });
}

template <typename Wrapper>
void
counter_storage<Wrapper>::metadata_initialize_counters_pmc(
    size_t dev_id, const std::string& name, const std::string& metric_description)
{
    const size_t EVENT_CODE       = 0;
    const size_t INSTANCE_ID      = 0;
    const char*  LONG_DESCRIPTION = "";
    const char*  COMPONENT        = "";
    const char*  BLOCK            = "";
    const char*  EXPRESSION       = "";
    const auto*  TARGET_ARCH      = "GPU";

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, dev_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID, name.c_str(),
          name.c_str(), metric_description.c_str(), LONG_DESCRIPTION, COMPONENT,
          "Unit Count", rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0,
          "{}" });
}

// ─── counter_event::operator() ───────────────────────────────────────────────

template <typename Wrapper>
void
counter_event<Wrapper>::operator()(const client_data<Wrapper>* tool_data,
                                   ::perfetto::CounterTrack*   _track,
                                   const std::string&          track_name,
                                   timing_interval<Wrapper>    _timing,
                                   scope::config               _scope) const
{
    if(!record.dispatch_data) return;

    const auto& _dispatch_info = record.dispatch_data->dispatch_info;
    const auto* _kern_sym_data =
        tool_data->get_kernel_symbol_info(_dispatch_info.kernel_id);

    auto _bundle =
        counter_bundle_t{ rocprofsys::utility::demangle(_kern_sym_data->kernel_name),
                          _scope };

    _bundle.push(_dispatch_info.queue_id.handle)
        .start()
        .store(record.record_counter.counter_value);

    _bundle.stop().pop(_dispatch_info.queue_id.handle);

    if(_track && _timing.start > 0 && _timing.end > _timing.start)
    {
        TRACE_COUNTER(trait::name<category::rocm_counter_collection>::value, *_track,
                      _timing.start, record.record_counter.counter_value);
        TRACE_COUNTER(trait::name<category::rocm_counter_collection>::value, *_track,
                      _timing.end, 0);

        const std::string event_metadata  = "{}";
        const size_t      stack_id        = 0;
        const size_t      parent_stack_id = 0;
        const size_t      correlation_id  = 0;
        const std::string call_stack      = "{}";
        const std::string line_info       = "{}";
        const size_t      agent_handle    = record.record_counter.agent_id.handle;
        const size_t      value           = record.record_counter.counter_value;

        auto agent = get_agent_manager_instance().get_agent_by_handle(agent_handle);

        trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
            static_cast<size_t>(
                category_enum_id<category::rocm_counter_collection>::value),
            track_name.c_str(), _timing.start, event_metadata.c_str(), stack_id,
            parent_stack_id, correlation_id, call_stack.c_str(), line_info.c_str(),
            static_cast<std::uint32_t>(agent.device_type_index),
            static_cast<std::uint8_t>(agent.type), track_name.c_str(),
            static_cast<double>(value), std::nullopt });

        trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
            static_cast<size_t>(
                category_enum_id<category::rocm_counter_collection>::value),
            track_name.c_str(), _timing.end, event_metadata.c_str(), stack_id,
            parent_stack_id, correlation_id, call_stack.c_str(), line_info.c_str(),
            static_cast<std::uint32_t>(agent.device_type_index),
            static_cast<std::uint8_t>(agent.type), track_name.c_str(),
            static_cast<double>(0), std::nullopt });
    }
}

// ─── counter_storage constructor ─────────────────────────────────────────────

template <typename Wrapper>
counter_storage<Wrapper>::counter_storage(const client_data<Wrapper>* _tool_data,
                                          std::uint64_t _devid, size_t _idx,
                                          std::string_view _name)
: tool_data{ _tool_data }
, device_id{ _devid }
, index{ static_cast<std::int64_t>(_idx) }
, metric_name{ _name }
, metric_description{ get_counter_description(_tool_data, metric_name) }
{
    auto _metric_name = std::string{ _name };
    _metric_name =
        std::regex_replace(_metric_name, std::regex{ "(.*)\\[([0-9]+)\\]" }, "$1_$2");
    storage_name = fmt::format("rocprof-device-{}-{}", device_id, _metric_name);
    manager      = tim::manager::instance();
    storage = std::make_unique<counter_storage_type>(tim::standalone_storage{}, index,
                                                     storage_name);
    if(manager)
    {
        manager->add_cleanup(storage_name + "cleanup",
                             [storage_ptr = storage.get(), metric_name = metric_name,
                              metric_description = metric_description]() {
                                 if(storage_ptr)
                                     counter_storage::write(storage_ptr, metric_name,
                                                            metric_description);
                             });
    }
    else
    {
        LOG_WARNING("{} counter_data_tracker is disabled. Can't add cleanup.",
                    metric_name);
    }

    {
        constexpr auto _unit = ::perfetto::CounterTrack::Unit::UNIT_COUNT;
        track_name           = fmt::format("GPU {} [{}]", _metric_name, device_id);
        track                = std::make_unique<counter_track_type>(
            ::perfetto::StaticString(track_name.c_str()));

        metadata_initialize_counter_category();
        metadata_initialize_counters_pmc(device_id, track_name.c_str(),
                                         metric_description);
        metadata_initialize_counter_track(track_name.c_str());
        track->set_is_incremental(false);
        track->set_unit(_unit);
        track->set_unit_multiplier(1);
    }
}

// ─── counter_storage::operator() ─────────────────────────────────────────────

template <typename Wrapper>
void
counter_storage<Wrapper>::operator()(const counter_event<Wrapper>& _event,
                                     timing_interval<Wrapper>      _timing,
                                     scope::config                 _scope) const
{
    operation::set_storage<counter_data_tracker>{}(storage.get());
    _event(tool_data, track.get(), track_name, _timing, _scope);
}

// ─── counter_storage::write_zero ─────────────────────────────────────────────

template <typename Wrapper>
void
counter_storage<Wrapper>::write_zero(typename Wrapper::timestamp_t timestamp) const
{
    if(!track || timestamp == 0) return;

    TRACE_COUNTER(trait::name<category::rocm_counter_collection>::value, *track,
                  timestamp, 0);

    trace_cache::get_buffer_storage().store(trace_cache::pmc_event_with_sample{
        static_cast<size_t>(category_enum_id<category::rocm_counter_collection>::value),
        track_name.c_str(), timestamp, "{}", 0, 0, 0, "{}", "{}",
        static_cast<std::uint32_t>(device_id), static_cast<std::uint8_t>(agent_type::GPU),
        track_name.c_str(), 0.0, std::nullopt });
}

// ─── counter_storage::write (static) ─────────────────────────────────────────

template <typename Wrapper>
void
counter_storage<Wrapper>::write(counter_storage_type* storage,
                                const std::string&    metric_name,
                                const std::string&    metric_description)
{
    if(!trait::runtime_enabled<counter_data_tracker>::get())
    {
        LOG_WARNING("{} counter_data_tracker is disabled. Can't write storage.",
                    metric_name);
        return;
    }

    operation::set_storage<counter_data_tracker>{}(storage);
    counter_data_tracker::label()       = metric_name;
    counter_data_tracker::description() = metric_description;
    storage->write();
}

}  // namespace rocprofsys::rocprofiler_sdk
