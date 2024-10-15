// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/rocprofiler.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/gpu.hpp"
#include "core/perfetto.hpp"
#include "library/rocm.hpp"
#include "library/rocm/hsa_rsrc_factory.hpp"

#include <timemory/backends/hardware_counters.hpp>
#include <timemory/manager.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/storage/types.hpp>
#include <timemory/utility/types.hpp>

#include <rocprofiler.h>

#include <atomic>
#include <cstdlib>
#include <dlfcn.h>
#include <hsa.h>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string.h>
#include <string_view>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace rocprofsys
{
namespace rocprofiler
{
namespace
{
using ::rocprofiler::util::AgentInfo;
using ::rocprofiler::util::HsaRsrcFactory;

auto&
get_event_names()
{
    static auto _v = std::map<uint32_t, std::vector<rocprofiler_feature_t>>{};
    return _v;
}
}  // namespace

// Error handler
void
fatal(const std::string& msg)
{
    ROCPROFSYS_PRINT_F("\n");
    ROCPROFSYS_PRINT_F("%s\n", msg.c_str());
    abort();
}

// Check returned HSA API status
const char*
rocm_error_string(hsa_status_t _status)
{
    const char* _err_string = nullptr;
    if(_status != HSA_STATUS_SUCCESS) rocprofiler_error_string(&_err_string);
    return _err_string;
}

// Check returned HSA API status
bool
rocm_check_status(hsa_status_t _status, const std::set<hsa_status_t>& _nonfatal = {})
{
    if(_status != HSA_STATUS_SUCCESS)
    {
        if(_nonfatal.count(_status) == 0)
            fatal(JOIN(" :: ", "ERROR", rocm_error_string(_status)));

        ROCPROFSYS_PRINT_F("Warning! %s\n", rocm_error_string(_status));
        return false;
    }
    return true;
}

// Context stored entry type
struct context_entry_t
{
    bool                        valid;
    hsa_agent_t                 agent;
    rocprofiler_group_t         group;
    rocprofiler_callback_data_t data;
};

// Context callback arg
struct callbacks_arg_t
{
    rocprofiler_pool_t** pools;
};

// Handler callback arg
struct handler_arg_t
{
    rocprofiler_feature_t* features;
    unsigned               feature_count;
};

bool&
is_setup()
{
    static bool _v = false;
    return _v;
}

std::map<uint32_t, std::vector<std::string_view>>
get_data_labels()
{
    auto _v = std::map<uint32_t, std::vector<std::string_view>>{};
    for(const auto& itr : get_event_names())
    {
        _v[itr.first] = {};
        for(auto vitr : itr.second)
            _v[itr.first].emplace_back(std::string_view{ vitr.name });
    }
    return _v;
}

// Dump stored context entry
void
rocm_dump_context_entry(context_entry_t* entry, rocprofiler_feature_t* features,
                        unsigned feature_count)
{
    volatile std::atomic<bool>* valid =
        reinterpret_cast<std::atomic<bool>*>(&entry->valid);
    while(valid->load() == false)
        sched_yield();

    const rocprofiler_dispatch_record_t* record = entry->data.record;

    if(!record) return;  // there is nothing to do here.

    auto _queue_id    = entry->data.queue_id;
    auto _thread_id   = entry->data.thread_id;
    auto _dev_id      = HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index;
    auto _kernel_name = std::string{ entry->data.kernel_name };
    auto _pos         = _kernel_name.find_last_of(')');
    if(_pos != std::string::npos) _kernel_name = _kernel_name.substr(0, _pos + 1);

    rocprofiler_group_t& group = entry->group;
    if(group.context == nullptr)
    {
        fatal("context is nullptr\n");
    }

    if(feature_count > 0)
    {
        rocm_check_status(rocprofiler_group_get_data(&group));
        rocm_check_status(rocprofiler_get_metrics(group.context));
    }

    auto _evt =
        component::rocm_event{ _dev_id,       _thread_id,  _queue_id,     _kernel_name,
                               record->begin, record->end, feature_count, features };

    component::rocm_data()->emplace_back(_evt);
}

// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool
rocm_context_handler(const rocprofiler_pool_entry_t* entry, void* arg)
{
    // Context entry
    context_entry_t* ctx_entry   = reinterpret_cast<context_entry_t*>(entry->payload);
    handler_arg_t*   handler_arg = reinterpret_cast<handler_arg_t*>(arg);

    // rocm::lock_t _lk{ rocm::rocm_mutex, std::defer_lock };
    // if(!_lk.owns_lock()) _lk.lock();

    rocm_dump_context_entry(ctx_entry, handler_arg->features, handler_arg->feature_count);

    return true;
}

// Kernel disoatch callback
hsa_status_t
rocm_dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* arg,
                       rocprofiler_group_t* group)
{
    // Passed tool data
    hsa_agent_t agent = callback_data->agent;

    // Open profiling context
    const unsigned   gpu_id = HsaRsrcFactory::Instance().GetAgentInfo(agent)->dev_index;
    callbacks_arg_t* callbacks_arg = reinterpret_cast<callbacks_arg_t*>(arg);
    rocprofiler_pool_t*      pool  = callbacks_arg->pools[gpu_id];
    rocprofiler_pool_entry_t pool_entry{};
    rocm_check_status(rocprofiler_pool_fetch(pool, &pool_entry));
    // Profiling context entry
    rocprofiler_t*   context = pool_entry.context;
    context_entry_t* entry   = reinterpret_cast<context_entry_t*>(pool_entry.payload);

    // Get group[0]
    rocm_check_status(rocprofiler_get_group(context, 0, group));

    // Fill profiling context entry
    entry->agent            = agent;
    entry->group            = *group;
    entry->data             = *callback_data;
    entry->data.kernel_name = strdup(callback_data->kernel_name);
    reinterpret_cast<std::atomic<bool>*>(&entry->valid)->store(true);

    return HSA_STATUS_SUCCESS;
}

unsigned
metrics_input(unsigned _device, rocprofiler_feature_t** ret)
{
    // Profiling feature objects
    auto                     _events = tim::delimit(config::get_rocm_events(), ", ;\t\n");
    std::vector<std::string> _features    = {};
    auto                     _this_device = JOIN("", ":device=", _device);
    for(auto itr : _events)
    {
        ROCPROFSYS_VERBOSE_F(3, "Processing feature '%s' for device %u...\n", itr.c_str(),
                             _device);
        auto _pos = itr.find(":device=");
        if(_pos != std::string::npos)
        {
            if(itr.find(_this_device) != std::string::npos)
            {
                _features.emplace_back(itr.substr(0, _pos));
            }
        }
        else
        {
            _features.emplace_back(itr);
        }
    }
    const unsigned         feature_count = _features.size();
    rocprofiler_feature_t* features      = new rocprofiler_feature_t[feature_count];
    memset(features, 0, feature_count * sizeof(rocprofiler_feature_t));

    // PMC events
    for(unsigned i = 0; i < feature_count; ++i)
    {
        ROCPROFSYS_VERBOSE_F(3, "Adding feature '%s' for device %u...\n",
                             _features.at(i).c_str(), _device);
        features[i].kind            = ROCPROFILER_FEATURE_KIND_METRIC;
        features[i].name            = strdup(_features.at(i).c_str());
        features[i].parameters      = nullptr;
        features[i].parameter_count = 0;
    }

    *ret = features;
    return feature_count;
}

using info_data = std::vector<component::rocm_info_entry>;

hsa_status_t
info_data_callback(const rocprofiler_info_data_t info, void* arg)
{
    using qualifier_t     = tim::hardware_counters::qualifier;
    using qualifier_vec_t = std::vector<qualifier_t>;
    auto* _data           = static_cast<info_data*>(arg);
    auto  _dev_index      = info.agent_index;

    switch(info.kind)
    {
        case ROCPROFILER_INFO_KIND_METRIC:
        {
            auto _device_qualifier_sym = JOIN("", ":device=", _dev_index);
            auto _device_qualifier =
                tim::hardware_counters::qualifier{ true, static_cast<int>(_dev_index),
                                                   _device_qualifier_sym,
                                                   JOIN(" ", "Device", _dev_index) };
            auto _long_desc = std::string{ info.metric.description };
            auto _units     = std::string{};
            auto _pysym     = std::string{};
            if(info.metric.expr != nullptr)
            {
                auto _sym        = JOIN("", info.metric.name, _device_qualifier_sym);
                auto _short_desc = JOIN("", "Derived counter: ", info.metric.expr);
                _data->emplace_back(component::rocm_info_entry(
                    true, tim::hardware_counters::api::rocm, _data->size(), 0, _sym,
                    _pysym, _short_desc, _long_desc, _units,
                    qualifier_vec_t{ _device_qualifier }));
            }
            else
            {
                if(info.metric.instances == 1)
                {
                    auto _sym = JOIN("", info.metric.name, _device_qualifier_sym);
                    auto _short_desc =
                        JOIN("", info.metric.name, " on device ", _dev_index);
                    _data->emplace_back(component::rocm_info_entry(
                        true, tim::hardware_counters::api::rocm, _data->size(), 0, _sym,
                        _pysym, _short_desc, _long_desc, _units,
                        qualifier_vec_t{ _device_qualifier }));
                }
                else
                {
                    for(uint32_t i = 0; i < info.metric.instances; ++i)
                    {
                        auto _instance_qualifier_sym = JOIN("", '[', i, ']');
                        auto _instance_qualifier =
                            tim::hardware_counters::qualifier{ true, static_cast<int>(i),
                                                               _instance_qualifier_sym,
                                                               JOIN(" ", "Instance", i) };
                        auto _sym = JOIN("", info.metric.name, _instance_qualifier_sym,
                                         _device_qualifier_sym);
                        auto _short_desc = JOIN("", info.metric.name, " instance ", i,
                                                " on device ", _dev_index);
                        _data->emplace_back(component::rocm_info_entry(
                            true, tim::hardware_counters::api::rocm, _data->size(), 0,
                            _sym, _pysym, _short_desc, _long_desc, _units,
                            qualifier_vec_t{ _device_qualifier, _instance_qualifier }));
                    }
                }
            }
            break;
        }
        default: printf("wrong info kind %u\n", info.kind); return HSA_STATUS_ERROR;
    }
    return HSA_STATUS_SUCCESS;
}

std::vector<component::rocm_info_entry>
rocm_metrics()
{
    std::vector<component::rocm_info_entry> _data = {};
    try
    {
        (void) HsaRsrcFactory::Instance();
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE_F(0, "%s\n", _e.what());
        return _data;
    }

    // Available GPU agents
    const unsigned gpu_count = HsaRsrcFactory::Instance().GetCountOfGpuAgents();

    std::vector<AgentInfo*> _gpu_agents(gpu_count, nullptr);
    for(unsigned i = 0; i < gpu_count; ++i)
    {
        const AgentInfo*  _agent   = _gpu_agents[i];
        const AgentInfo** _agent_p = &_agent;
        HsaRsrcFactory::Instance().GetGpuAgentInfo(i, _agent_p);

        if(!rocm_check_status(rocprofiler_iterate_info(
                                  &_agent->dev_id, ROCPROFILER_INFO_KIND_METRIC,
                                  info_data_callback, reinterpret_cast<void*>(&_data)),
                              { HSA_STATUS_ERROR_NOT_INITIALIZED }))
        {
            ROCPROFSYS_WARNING_F(-1, "rocprofiler_iterate_info failed for gpu agent %u\n",
                                 i);
        }
    }

    if(gpu_count > 0 && _data.empty())
    {
        if(!rocm_check_status(rocprofiler_iterate_info(
                                  nullptr, ROCPROFILER_INFO_KIND_METRIC,
                                  info_data_callback, reinterpret_cast<void*>(&_data)),
                              { HSA_STATUS_ERROR_NOT_INITIALIZED }))
        {
            ROCPROFSYS_WARNING_F(
                -1, "rocprofiler_iterate_info failed for %i gpu agents\n", gpu_count);
        }
    }

    auto _settings = tim::settings::shared_instance();
    if(_settings)
    {
        auto ritr = _settings->find("ROCPROFSYS_ROCM_EVENTS");
        if(ritr != _settings->end())
        {
            auto _rocm_events = ritr->second;
            if(_rocm_events->get_choices().empty())
            {
                std::vector<std::string> _choices = {};
                _choices.reserve(_data.size());
                for(auto itr : _data)
                {
                    if(!itr.symbol().empty()) _choices.emplace_back(itr.symbol());
                }
                _rocm_events->set_choices(_choices);
            }
        }
    }

    return _data;
}

void
rocm_initialize()
{
    // Available GPU agents
    const unsigned gpu_count = HsaRsrcFactory::Instance().GetCountOfGpuAgents();

    (void) rocm_metrics();

    // Adding dispatch observer
    callbacks_arg_t* callbacks_arg = new callbacks_arg_t{};
    callbacks_arg->pools           = new rocprofiler_pool_t*[gpu_count];
    for(unsigned gpu_id = 0; gpu_id < gpu_count; gpu_id++)
    {
        // Getting profiling features
        rocprofiler_feature_t* features      = nullptr;
        unsigned               feature_count = metrics_input(gpu_id, &features);

        if(features)
        {
            get_event_names()[gpu_id].clear();
            get_event_names()[gpu_id].reserve(feature_count);
            for(unsigned i = 0; i < feature_count; ++i)
                get_event_names().at(gpu_id).emplace_back(features[i]);
        }

        // Handler arg
        handler_arg_t* handler_arg = new handler_arg_t{};
        handler_arg->features      = features;
        handler_arg->feature_count = feature_count;

        // Context properties
        rocprofiler_pool_properties_t properties{};
        properties.num_entries   = 100;
        properties.payload_bytes = sizeof(context_entry_t);
        properties.handler       = rocm_context_handler;
        properties.handler_arg   = handler_arg;

        // Getting GPU device info
        const AgentInfo* agent_info = nullptr;
        if(HsaRsrcFactory::Instance().GetGpuAgentInfo(gpu_id, &agent_info) == false)
        {
            fprintf(stderr, "GetGpuAgentInfo failed\n");
            abort();
        }

        // Open profiling pool
        rocprofiler_pool_t* pool = nullptr;
        uint32_t            mode = 0;  // ROCPROFILER_MODE_SINGLEGROUP
        rocm_check_status(rocprofiler_pool_open(agent_info->dev_id, features,
                                                feature_count, &pool, mode, &properties));
        callbacks_arg->pools[gpu_id] = pool;
    }

    rocprofiler_queue_callbacks_t callbacks_ptrs{};
    callbacks_ptrs.dispatch = rocm_dispatch_callback;
    int err = rocprofiler_set_queue_callbacks(callbacks_ptrs, callbacks_arg);
    ROCPROFSYS_VERBOSE_F(3, "err=%d, rocprofiler_set_queue_callbacks\n", err);

    is_setup() = true;
}

void
rocm_cleanup()
{
    // Unregister dispatch callback
    rocm_check_status(rocprofiler_remove_queue_callbacks());
    // close profiling pool
    // rocm_check_status(rocprofiler_pool_flush(pool));
    // rocm_check_status(rocprofiler_pool_close(pool));
}

namespace
{
using rocm_event         = component::rocm_event;
using rocm_data_t        = component::rocm_data_t;
using rocm_metric_type   = component::rocm_metric_type;
using rocm_feature_value = component::rocm_feature_value;
using rocm_data_tracker  = component::rocm_data_tracker;

void
post_process_perfetto()
{
    using counter_track = perfetto_counter_track<rocm_event>;

    static bool _once = false;
    if(_once) return;

    auto _data          = rocm_data_t{};
    auto _device_data   = std::map<uint32_t, std::vector<rocm_event*>>{};
    auto _device_fields = std::map<uint32_t, std::vector<std::string_view>>{};
    auto _device_range  = std::map<uint32_t, std::set<rocm_metric_type>>{};

    for(size_t i = 0; i < ROCPROFSYS_MAX_THREADS; ++i)
    {
        auto& _v = component::rocm_data(i);
        if(_v)
        {
            _data.reserve(_data.size() + _v->size());
            for(auto& itr : *_v)
                _data.emplace_back(itr);
        }
    }

    if(_data.empty()) return;
    _once = true;

    std::sort(_data.begin(), _data.end());

    auto _get_events = [](std::vector<rocm_event*>& _inp, rocm_metric_type _ts) {
        auto _v = std::vector<rocm_event*>{};
        for(const auto& itr : _inp)
        {
            if(_ts >= itr->entry && _ts <= itr->exit) _v.emplace_back(itr);
            if(_ts > itr->exit) break;
        }
        return _v;
    };

    {
        auto _device_time = std::map<uint32_t, std::set<rocm_metric_type>>{};
        for(auto& itr : _data)
        {
            _device_data[itr.device_id].emplace_back(&itr);
            _device_time[itr.device_id].emplace(itr.entry);
            _device_time[itr.device_id].emplace(itr.exit);
            auto _dev_id = itr.device_id;
            if(get_use_perfetto() && !counter_track::exists(_dev_id))
            {
                auto addendum = [&](auto&& _v) {
                    return JOIN(" ", "Device", _v, JOIN("", '[', _dev_id, ']'));
                };
                for(auto nitr : itr.feature_names)
                {
                    auto _name = get_data_labels().at(itr.device_id).at(nitr);
                    counter_track::emplace(_dev_id, addendum(_name));
                }
            }
        }

        for(auto& ditr : _device_time)
        {
            for(auto itr = ditr.second.begin(); itr != ditr.second.end(); ++itr)
            {
                auto _next = std::next(itr);
                if(_next == ditr.second.end()) continue;
                _device_range[ditr.first].emplace(((*_next / 2) + (*itr / 2)));
            }
        }
    }

    for(auto& ditr : _device_range)
    {
        auto _dev_id         = ditr.first;
        auto _values         = std::vector<rocm_feature_value>{};
        auto _ts_sorted_data = _device_data[_dev_id];
        std::sort(_ts_sorted_data.begin(), _ts_sorted_data.end(),
                  [](auto* _l, auto* _r) { return _l->exit < _r->exit; });
        for(const auto& itr : ditr.second)
        {
            auto     _v  = _get_events(_ts_sorted_data, itr);
            uint64_t _ts = itr;
            for(auto* vitr : _v)
            {
                size_t _n = vitr->feature_values.size();
                if(_values.empty())
                {
                    _values.reserve(_n);
                    for(size_t i = 0; i < _n; ++i)
                    {
                        _values.emplace_back(vitr->feature_values.at(i));
                    }
                }
                else
                {
                    for(size_t i = 0; i < _n; ++i)
                    {
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wdouble-promotion"
#endif
                        auto _plus = [](auto& _lhs, auto&& _rhs) { _lhs += _rhs; };
                        std::visit(_plus, _values.at(i), vitr->feature_values.at(i));
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif
                    }
                }
            }

            for(size_t i = 0; i < _values.size(); ++i)
            {
                auto _trace_counter = [_dev_id, i, _ts](auto&& _val) {
                    TRACE_COUNTER("kernel_hardware_counter",
                                  counter_track::at(_dev_id, i), _ts, _val);
                };
                std::visit(_trace_counter, _values.at(i));
            }
        }
    }
}

void
post_process_timemory()
{
    static bool _once = false;
    if(_once) return;

    auto _data          = rocm_data_t{};
    auto _device_data   = std::map<uint32_t, std::vector<rocm_event*>>{};
    auto _device_fields = std::map<uint32_t, std::vector<std::string_view>>{};
    auto _device_range  = std::map<uint32_t, std::set<rocm_metric_type>>{};

    for(size_t i = 0; i < ROCPROFSYS_MAX_THREADS; ++i)
    {
        auto& _v = component::rocm_data(i);
        if(_v)
        {
            _data.reserve(_data.size() + _v->size());
            for(auto& itr : *_v)
                _data.emplace_back(itr);
        }
    }

    if(_data.empty()) return;
    _once = true;

    std::sort(_data.begin(), _data.end());

    for(auto& itr : _data)
    {
        _device_data[itr.device_id].emplace_back(&itr);
    }

    for(auto& itr : _device_data)
    {
        // sort according to when it exited
        std::sort(itr.second.begin(), itr.second.end(),
                  [](auto* _lhs, auto* _rhs) { return _lhs->exit < _rhs->exit; });
    }

    using storage_type = typename rocm_data_tracker::storage_type;
    using bundle_type  = tim::lightweight_tuple<rocm_data_tracker>;

    auto        _info            = rocm_metrics();
    static auto _get_description = [&_info](std::string_view _v) {
        for(auto& itr : _info)
        {
            if(itr.symbol().find(_v) == 0 || itr.short_description().find(_v) == 0)
            {
                return itr.long_description();
            }
        }
        return std::string{};
    };

    struct local_event
    {
        rocm_event*                      parent   = nullptr;
        mutable std::vector<local_event> children = {};

        ROCPROFSYS_DEFAULT_OBJECT(local_event)

        explicit local_event(rocm_event* _v)
        : parent{ _v }
        {}

        bool operator()(rocm_event* _v)
        {
            if(!parent) return false;
            if(_v->device_id != parent->device_id) return false;
            if(_v->entry > parent->entry && _v->exit <= parent->exit)
            {
                children.emplace_back(_v);
                return true;
            }
            return false;
        }

        bool operator<(const local_event& _v) const
        {
            if(!parent && _v.parent) return true;
            if(parent && !_v.parent) return false;
            return *parent < *_v.parent;
        }

        void operator()(int64_t _index, scope::config _scope) const
        {
            if(!parent) return;
            bundle_type _bundle{ parent->name, _scope };
            _bundle.push(parent->queue_id)
                .start()
                .store(parent->feature_values.at(_index));

            std::sort(children.begin(), children.end());
            for(const auto& itr : children)
                itr(_index, _scope);

            _bundle.stop().pop(parent->queue_id);
        }
    };

    struct local_storage
    {
        int64_t                       index              = 0;
        std::string                   metric_name        = {};
        std::string                   metric_description = {};
        std::unique_ptr<storage_type> storage            = {};

        local_storage(uint32_t _devid, size_t _idx, std::string_view _name)
        : index{ static_cast<int64_t>(_idx) }
        , metric_name{ _name }
        , metric_description{ _get_description(metric_name) }
        {
            auto _metric_name = std::string{ _name };
            _metric_name      = std::regex_replace(
                _metric_name, std::regex{ "(.*)\\[([0-9]+)\\]" }, "$1_$2");
            storage = std::make_unique<storage_type>(
                tim::standalone_storage{}, index,
                JOIN('-', "rocprof", "device", _devid, _metric_name));
        }

        void operator()(const local_event& _event, scope::config _scope) const
        {
            operation::set_storage<rocm_data_tracker>{}(storage.get());
            _event(index, _scope);
        }

        void write() const
        {
            rocm_data_tracker::label()       = metric_name;
            rocm_data_tracker::description() = metric_description;
            storage->write();
        }
    };

    auto _local_data = std::map<uint32_t, std::vector<local_event>>{};
    auto _scope      = scope::get_default();

    for(auto& ditr : _device_data)
    {
        ROCPROFSYS_VERBOSE_F(1, "Post-processing %zu entries for device %u...\n",
                             ditr.second.size(), ditr.first);
        auto _storage = std::vector<local_storage>{};
        for(auto& itr : ditr.second)
        {
            auto _n = itr->feature_names.size();
            if(_n > _storage.size())
            {
                _storage.reserve(_n);
                for(size_t i = _storage.size(); i < _n; ++i)
                    _storage.emplace_back(
                        ditr.first, i,
                        get_data_labels().at(ditr.first).at(itr->feature_names.at(i)));
            }
        }

        auto& _local = _local_data[ditr.first];
        _local.reserve(ditr.second.size());
        double _avg = 0.0;
        for(auto& itr : ditr.second)
        {
            if(_local.empty() || itr->entry >= _local.back().parent->exit)
            {
                _local.emplace_back(itr);
            }
            else
            {
                size_t _n     = 0;
                bool   _found = false;
                for(auto litr = _local.rbegin(); litr != _local.rend(); ++litr)
                {
                    ++_n;
                    if((*litr)(itr))
                    {
                        _found = true;
                        break;
                    }
                }
                if(!_found) _local.emplace_back(itr);
                _avg += _n;
            }
        }

        ROCPROFSYS_VERBOSE_F(3, "Average # of iterations before match: %.1f\n",
                             _avg / ditr.second.size() * 100.0);

        for(auto& sitr : _storage)
        {
            for(auto& itr : _local)
                sitr(itr, _scope);
        }

        for(auto& itr : _storage)
            itr.write();
    }

    tim::trait::runtime_enabled<rocprofsys::rocprofiler::rocm_data_tracker>::set(false);
}
}  // namespace

void
post_process()
{
    if(get_use_perfetto()) post_process_perfetto();

    if(get_use_timemory())
    {
        auto _manager = tim::manager::master_instance();
        if(_manager)
        {
            _manager->add_cleanup("rocprofiler", &post_process_timemory);
        }
        else
        {
            post_process_timemory();
        }
    }
}
}  // namespace rocprofiler
}  // namespace rocprofsys
