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

#include "library/rocprofiler-sdk/counters.hpp"
#include "library/rocprofiler-sdk/fwd.hpp"

#include <memory>
#include <timemory/utility/types.hpp>

namespace rocprofsys
{
namespace rocprofiler_sdk
{
namespace
{
std::string
get_counter_description(const client_data* tool_data, std::string_view _v)
{
    const auto& _info = tool_data->events_info;
    for(const auto& itr : _info)
    {
        if(itr.symbol().find(_v) == 0 || itr.short_description().find(_v) == 0)
        {
            return itr.long_description();
        }
    }
    return std::string{};
}
}  // namespace

void
counter_event::operator()(const client_data* tool_data, ::perfetto::CounterTrack* _track,
                          timing_interval _timing, scope::config _scope) const
{
    if(!record.dispatch_data) return;

    const auto& _dispatch_info = record.dispatch_data->dispatch_info;
    const auto* _kern_sym_data =
        tool_data->get_kernel_symbol_info(_dispatch_info.kernel_id);

    auto _bundle = counter_bundle_t{ tim::demangle(_kern_sym_data->kernel_name), _scope };

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
    }
}

counter_storage::counter_storage(const client_data* _tool_data, uint64_t _devid,
                                 size_t _idx, std::string_view _name)
: tool_data{ _tool_data }
, device_id{ _devid }
, index{ static_cast<int64_t>(_idx) }
, metric_name{ _name }
, metric_description{ get_counter_description(_tool_data, metric_name) }
{
    auto _metric_name = std::string{ _name };
    _metric_name =
        std::regex_replace(_metric_name, std::regex{ "(.*)\\[([0-9]+)\\]" }, "$1_$2");
    storage_name = JOIN('-', "rocprof", "device", device_id, _metric_name);
    storage = std::make_unique<counter_storage_type>(tim::standalone_storage{}, index,
                                                     storage_name);
    tim::manager::instance()->add_cleanup(
        storage_name + "cleanup", [storage_ptr = storage.get(), metric_name = metric_name,
                                   metric_description = metric_description]() {
            if(storage_ptr)
                counter_storage::write(storage_ptr, metric_name, metric_description);
        });
    {
        constexpr auto _unit = ::perfetto::CounterTrack::Unit::UNIT_COUNT;
        track_name = JOIN(" ", "GPU", _metric_name, JOIN("", '[', device_id, ']'));
        track      = std::make_unique<counter_track_type>(
            ::perfetto::StaticString(track_name.c_str()));
        track->set_is_incremental(false);
        track->set_unit(_unit);
        track->set_unit_multiplier(1);
    }
}

void
counter_storage::operator()(const counter_event& _event, timing_interval _timing,
                            scope::config _scope) const
{
    operation::set_storage<counter_data_tracker>{}(storage.get());
    _event(tool_data, track.get(), _timing, _scope);
}

void
counter_storage::write(counter_storage_type* storage, std::string metric_name,
                       std::string metric_description)
{
    if(!trait::runtime_enabled<counter_data_tracker>::get())
    {
        ROCPROFSYS_WARNING_F(
            1, "%s counter_data_tracker is disabled. Can't write storage.\n",
            metric_name.c_str());
        return;
    }

    operation::set_storage<counter_data_tracker>{}(storage);
    counter_data_tracker::label()       = metric_name;
    counter_data_tracker::description() = metric_description;
    storage->write();
}
}  // namespace rocprofiler_sdk
}  // namespace rocprofsys
