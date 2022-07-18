// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/components/rocprofiler.hpp"
#include "library/common.hpp"
#include "library/components/pthread_create_gotcha.hpp"
#include "library/components/pthread_gotcha.hpp"
#include "library/config.hpp"
#include "library/debug.hpp"
#include "library/defines.hpp"
#include "library/dynamic_library.hpp"
#include "library/perfetto.hpp"
#include "library/redirect.hpp"
#include "library/rocprofiler.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"

#include <timemory/storage/types.hpp>
#include <timemory/utility/types.hpp>
#include <timemory/variadic/functional.hpp>
#include <timemory/variadic/lightweight_tuple.hpp>

#include <rocprofiler.h>

#include <cstdint>
#include <string_view>
#include <type_traits>

TIMEMORY_STATISTICS_TYPE(component::rocm_data_tracker, component::rocm_feature_value)
TIMEMORY_DEFINE_CONCRETE_TRAIT(report_units, component::rocm_data_tracker, false_type)

namespace tim
{
namespace component
{
namespace
{
auto&
rocprofiler_activity_count()
{
    static std::atomic<int64_t> _v{ 0 };
    return _v;
}
}  // namespace

omnitrace::unique_ptr_t<rocm_data_t>&
rocm_data(int64_t _tid)
{
    using thread_data_t = omnitrace::thread_data<rocm_data_t, rocm_event>;
    static auto& _v     = thread_data_t::instances(thread_data_t::construct_on_init{});
    return _v.at(_tid);
}

rocm_event::rocm_event(uint32_t _dev, uint32_t _thr, uint32_t _queue,
                       std::string _event_name, rocm_metric_type _begin,
                       rocm_metric_type _end, uint32_t _feature_count, void* _features_v)
: device_id{ _dev }
, thread_id{ _thr }
, queue_id{ _queue }
, entry{ _begin }
, exit{ _end }
, name(std::move(_event_name))
{
    feature_values.reserve(_feature_count);
    feature_names.reserve(_feature_count);
    auto* _features = static_cast<rocprofiler_feature_t*>(_features_v);
    for(uint32_t i = 0; i < _feature_count; ++i)
    {
        const rocprofiler_feature_t* p = &_features[i];
        feature_names.emplace_back(p->name);
        switch(p->data.kind)
        {
            // Output metrics results
            case ROCPROFILER_DATA_KIND_UNINIT: break;
            case ROCPROFILER_DATA_KIND_BYTES:
                feature_values.emplace_back(
                    rocm_feature_value{ p->data.result_bytes.size });
                break;
            case ROCPROFILER_DATA_KIND_INT32:
                feature_values.emplace_back(rocm_feature_value{ p->data.result_int32 });
                break;
            case ROCPROFILER_DATA_KIND_FLOAT:
                feature_values.emplace_back(rocm_feature_value{ p->data.result_float });
                break;
            case ROCPROFILER_DATA_KIND_DOUBLE:
                feature_values.emplace_back(rocm_feature_value{ p->data.result_double });
                break;
            case ROCPROFILER_DATA_KIND_INT64:
                feature_values.emplace_back(rocm_feature_value{ p->data.result_int64 });
                break;
        }
    }
}

std::string
rocm_event::as_string() const
{
    std::stringstream _ss{};
    _ss << name << ", device: " << device_id << ", queue: " << queue_id
        << ", thread: " << thread_id << ", entry: " << entry << ", exit = " << exit;
    _ss.precision(3);
    _ss << std::fixed;
    for(size_t i = 0; i < feature_names.size(); ++i)
    {
        _ss << ", " << feature_names.at(i) << " = ";
        auto _as_string = [&_ss](auto&& itr) { _ss << std::setw(4) << itr; };
        std::visit(_as_string, feature_values.at(i));
    }
    return _ss.str();
}

void
rocprofiler::preinit()
{
    rocprofiler_data::label()       = "rocprofiler";
    rocprofiler_data::description() = "ROCm hardware counters";
}

void
rocprofiler::start()
{
    if(tracker_type::start() == 0) setup();
}

void
rocprofiler::stop()
{
    if(tracker_type::stop() == 0) shutdown();
}

bool
rocprofiler::is_setup()
{
    return omnitrace::rocprofiler::is_setup();
}

void
rocprofiler::add_setup(const std::string&, std::function<void()>&&)
{}

void
rocprofiler::add_shutdown(const std::string&, std::function<void()>&&)
{}

void
rocprofiler::remove_setup(const std::string&)
{}

void
rocprofiler::remove_shutdown(const std::string&)
{}

void
rocprofiler::setup()
{
    OMNITRACE_VERBOSE_F(1, "rocprofiler is setup\n");
}

void
rocprofiler::shutdown()
{
    omnitrace::rocprofiler::post_process();
    omnitrace::rocprofiler::rocm_cleanup();
    /*
    using storage_type = typename rocprofiler_data::storage_type;
    using bundle_t     = rocprofiler_data;
    using tag_t        = api::omnitrace;

    auto    _data   = omnitrace::rocprofiler::get_data();
    auto    _labels = omnitrace::rocprofiler::get_data_labels();
    auto    _info   = omnitrace::rocprofiler::rocm_metrics();
    int64_t _idx    = 0;
    auto    _scope  = tim::scope::get_default();

    auto _get_metric_desc = [_info](std::string_view _v) {
        for(auto itr : _info)
        {
            if(itr.symbol().find(_v) == 0 || itr.short_description().find(_v) == 0)
                return std::make_pair(itr.short_description(), itr.long_description());
        }
        return std::make_pair(std::string{}, std::string{});
    };

    auto _debug       = settings::debug();
    settings::debug() = true;

    struct hw_counters
    {};

    using rocm_counter = omnitrace::rocprofiler::rocm_counter;

    struct perfetto_rocm_event
    {
        rocm_counter      entry = {};
        rocm_counter      exit  = {};
        rocprofiler_value value = {};

        bool operator<(const perfetto_rocm_event& _v) const
        {
            return (entry.at(0) == _v.entry.at(0)) ? exit.at(0) < _v.exit.at(0)
                                                   : entry.at(0) < _v.entry.at(0);
        }
    };

    // contains the necessary info for export to perfetto
    auto _perfetto_raw_data =
        std::map<int64_t, std::map<int64_t, std::vector<perfetto_rocm_event>>>{};
    // contains the time-stamp regions for the counter tracks
    auto _perfetto_time_regions =
        std::map<int64_t, std::map<int64_t, std::set<uint64_t>>>{};

    // create a layout compatible for exporting to perfetto
    for(const auto& itr : _labels)
    {
        auto _dev_id   = itr.first;
        auto _dev_name = JOIN("", '[', _dev_id, ']');

        for(size_t i = 0; i < itr.second.size(); ++i)
        {
            auto _metric_name = itr.second.at(i);
            auto _idx         = perfetto_counter_track<hw_counters>::emplace(
                _dev_id, JOIN(' ', "Device", _metric_name, _dev_name));
            auto& _raw = _perfetto_raw_data[_dev_id][_idx];
            auto& _reg = _perfetto_time_regions[_dev_id][_idx];
            for(const auto& ditr : _data)
            {
                _raw.emplace_back(
                    perfetto_rocm_event{ ditr.entry, ditr.exit, ditr.data.at(i) });
            }
            std::sort(_raw.begin(), _raw.end());
            for(auto ritr : _raw)
            {
                if(pthread_create_gotcha::is_valid_execution_time(0, ritr.entry.at(0)))
                    _reg.emplace(ritr.entry.at(0));
                if(pthread_create_gotcha::is_valid_execution_time(0, ritr.exit.at(0)))
                    _reg.emplace(ritr.exit.at(0));
            }
        }
    }

    for(auto& ditr : _perfetto_time_regions)
        for(auto& citr : ditr.second)
        {
            for(auto _ts = citr.second.begin(); _ts != citr.second.end(); ++_ts)
            {
                rocprofiler_value _v    = {};
                auto              _curr = _ts;
                auto              _next = std::next(_ts);
                if(_next == citr.second.end()) continue;
                auto _min_ts = *_curr;
                auto _max_ts = (_next == citr.second.end()) ? *_curr : *_next;
                for(auto itr : _perfetto_raw_data[ditr.first][citr.first])
                {
                    if(itr.entry[0] >= _min_ts && itr.exit[0] <= _max_ts)
                    {
                        using namespace tim::stl;
                        _v += itr.value;
                    }
                }

                auto _write_counter = [&](auto _v) {
                    if(_min_ts == _max_ts)
                    {
                        using value_type = std::remove_reference_t<
                            std::remove_cv_t<decay_t<decltype(_v)>>>;
                        _v = static_cast<value_type>(0);
                    }

                    TRACE_COUNTER(
                        "hardware_counter",
                        perfetto_counter_track<hw_counters>::at(ditr.first, citr.first),
                        _min_ts, _v);
                };
                std::visit(_write_counter, _v);
            }
        }

    for(const auto& itr : _labels)
    {
        for(size_t i = 0; i < itr.second.size(); ++i)
        {
            auto _metric_name         = itr.second.at(i);
            auto _metric_desc         = _get_metric_desc(_metric_name).second;
            rocprofiler_data::label() = _metric_name;
            if(!_metric_desc.empty())
                rocprofiler_data::description() = JOIN(" - ", "rocprof", _metric_desc);
            auto _dev_id = itr.first;
            auto _label  = JOIN('-', "rocprofiler", _metric_name, "device", _dev_id);
            storage_type          _storage{ standalone_storage{}, ++_idx, _label };
            std::vector<bundle_t> _bundles = {};
            _bundles.reserve(_data.size());
            for(const auto& ditr : _data)
            {
                auto _hash = add_hash_id(ditr.name);
                auto _v    = ditr.data.at(i);
                auto _obj  = std::tie(_bundles.emplace_back(bundle_t{}));
                invoke::reset<tag_t>(_obj);
                invoke::push<tag_t>(_obj, _scope, _hash, &_storage, _dev_id);
                invoke::start<tag_t>(_obj);
                invoke::store<tag_t>(_obj, _v);
                invoke::stop<tag_t>(_obj);
                invoke::pop<tag_t>(_obj, &_storage, _dev_id);
            }

            _storage.write(_label);
        }
    }
    settings::debug() = _debug;
    */

    OMNITRACE_VERBOSE_F(1, "rocprofiler is shutdown\n");
}

scope::transient_destructor
rocprofiler::protect_flush_activity()
{
    return scope::transient_destructor([]() { --rocprofiler_activity_count(); },
                                       []() { ++rocprofiler_activity_count(); });
}
}  // namespace component
}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(rocprofiler, false, void)
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(rocprofiler_data, true,
                                      tim::component::rocprofiler_value)
