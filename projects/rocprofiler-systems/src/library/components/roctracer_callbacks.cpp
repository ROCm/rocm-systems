// Copyright (c) 2018 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// with the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// * Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimers.
//
// * Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimers in the
// documentation and/or other materials provided with the distribution.
//
// * Neither the names of Advanced Micro Devices, Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this Software without specific prior written permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
// THE SOFTWARE.

#include "library/components/roctracer_callbacks.hpp"
#include "library.hpp"
#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/thread_data.hpp"
#include "timemory/backends/threading.hpp"

#include <cstdint>

TIMEMORY_DEFINE_API(roctracer)
namespace omnitrace
{
namespace api = tim::api;

std::unordered_set<uint64_t>&
get_roctracer_kernels()
{
    static auto _v = std::unordered_set<uint64_t>{};
    return _v;
}

auto&
get_roctracer_hip_data(int64_t _tid = threading::get_id())
{
    using data_t        = std::unordered_map<uint64_t, roctracer_bundle_t>;
    using thread_data_t = omnitrace_thread_data<data_t, api::roctracer>;
    static auto& _v     = thread_data_t::instances(thread_data_t::construct_on_init{});
    return _v.at(_tid);
}

std::unordered_map<uint64_t, const char*>&
get_roctracer_key_data()
{
    static auto _v = std::unordered_map<uint64_t, const char*>{};
    return _v;
}

std::unordered_map<uint64_t, int64_t>&
get_roctracer_tid_data()
{
    static auto _v = std::unordered_map<uint64_t, int64_t>{};
    return _v;
}

using cid_tuple_t = std::tuple<uint64_t, uint64_t, uint16_t>;
std::unordered_map<uint64_t, cid_tuple_t>&
get_roctracer_cid_data()
{
    static auto _v = std::unordered_map<uint64_t, cid_tuple_t>{};
    return _v;
}

auto&
get_hip_activity_callbacks(int64_t _tid = threading::get_id())
{
    using thread_data_t =
        omnitrace_thread_data<std::vector<std::function<void()>>, api::roctracer>;
    static auto& _v = thread_data_t::instances(thread_data_t::construct_on_init{});
    return _v.at(_tid);
}

std::unique_ptr<hsa_timer_t>&
get_hsa_timer()
{
    static auto _v = std::unique_ptr<hsa_timer_t>{};
    return _v;
}

using hip_activity_mutex_t = std::decay_t<decltype(get_hip_activity_callbacks())>;
using key_data_mutex_t     = std::decay_t<decltype(get_roctracer_key_data())>;
using hip_data_mutex_t     = std::decay_t<decltype(get_roctracer_hip_data())>;
using cid_data_mutex_t     = std::decay_t<decltype(get_roctracer_cid_data())>;

auto&
get_hip_activity_mutex(int64_t _tid = threading::get_id())
{
    return tim::type_mutex<hip_activity_mutex_t, api::roctracer, max_supported_threads>(
        _tid);
}

// HSA API callback function
void
hsa_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    (void) arg;
    const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
    OMNITRACE_DEBUG("<%-30s id(%u)\tcorrelation_id(%lu) %s>\n",
                    roctracer_op_string(domain, cid, 0), cid, data->correlation_id,
                    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");

    static thread_local timestamp_t begin_timestamp = 0;
    static auto&                    timer           = get_hsa_timer();
    static auto                     _scope          = []() {
        auto _v = scope::config{};
        if(get_roctracer_timeline_profile()) _v += scope::timeline{};
        if(get_roctracer_flat_profile()) _v += scope::flat{};
        return _v;
    }();

    if(!timer) return;

    switch(cid)
    {
        case HSA_API_ID_hsa_init:
        case HSA_API_ID_hsa_shut_down:
        case HSA_API_ID_hsa_agent_get_exception_policies:
        case HSA_API_ID_hsa_agent_get_info:
        case HSA_API_ID_hsa_amd_agent_iterate_memory_pools:
        case HSA_API_ID_hsa_amd_agent_memory_pool_get_info:
        case HSA_API_ID_hsa_amd_coherency_get_type:
        case HSA_API_ID_hsa_amd_memory_pool_get_info:
        case HSA_API_ID_hsa_amd_pointer_info:
        case HSA_API_ID_hsa_amd_pointer_info_set_userdata:
        case HSA_API_ID_hsa_amd_profiling_async_copy_enable:
        case HSA_API_ID_hsa_amd_profiling_get_async_copy_time:
        case HSA_API_ID_hsa_amd_profiling_get_dispatch_time:
        case HSA_API_ID_hsa_amd_profiling_set_profiler_enabled:
        case HSA_API_ID_hsa_cache_get_info:
        case HSA_API_ID_hsa_code_object_get_info:
        case HSA_API_ID_hsa_code_object_get_symbol:
        case HSA_API_ID_hsa_code_object_get_symbol_from_name:
        case HSA_API_ID_hsa_code_object_reader_create_from_memory:
        case HSA_API_ID_hsa_code_symbol_get_info:
        case HSA_API_ID_hsa_executable_create_alt:
        case HSA_API_ID_hsa_executable_freeze:
        case HSA_API_ID_hsa_executable_get_info:
        case HSA_API_ID_hsa_executable_get_symbol:
        case HSA_API_ID_hsa_executable_get_symbol_by_name:
        case HSA_API_ID_hsa_executable_symbol_get_info:
        case HSA_API_ID_hsa_extension_get_name:
        case HSA_API_ID_hsa_ext_image_data_get_info:
        case HSA_API_ID_hsa_ext_image_data_get_info_with_layout:
        case HSA_API_ID_hsa_ext_image_get_capability:
        case HSA_API_ID_hsa_ext_image_get_capability_with_layout:
        case HSA_API_ID_hsa_isa_get_exception_policies:
        case HSA_API_ID_hsa_isa_get_info:
        case HSA_API_ID_hsa_isa_get_info_alt:
        case HSA_API_ID_hsa_isa_get_round_method:
        case HSA_API_ID_hsa_region_get_info:
        case HSA_API_ID_hsa_system_extension_supported:
        case HSA_API_ID_hsa_system_get_extension_table:
        case HSA_API_ID_hsa_system_get_info:
        case HSA_API_ID_hsa_system_get_major_extension_table:
        case HSA_API_ID_hsa_wavefront_get_info: break;
        default:
        {
            if(data->phase == ACTIVITY_API_PHASE_ENTER)
            {
                begin_timestamp = timer->timestamp_fn_ns();
            }
            else
            {
                const auto*       _name         = roctracer_op_string(domain, cid, 0);
                const timestamp_t end_timestamp = (cid == HSA_API_ID_hsa_shut_down)
                                                      ? begin_timestamp
                                                      : timer->timestamp_fn_ns();

                if(begin_timestamp > end_timestamp) return;

                if(get_use_perfetto())
                {
                    TRACE_EVENT_BEGIN("device", perfetto::StaticString{ _name },
                                      begin_timestamp);
                    TRACE_EVENT_END("device", end_timestamp);
                }

                if(get_use_timemory())
                {
                    std::unique_lock<std::mutex> _lk{ tasking::get_roctracer_mutex() };
                    auto                         _begin_ns = begin_timestamp;
                    auto                         _end_ns   = end_timestamp;
                    tasking::get_roctracer_task_group().exec(
                        [_name, _begin_ns, _end_ns]() {
                            roctracer_hsa_bundle_t _bundle{ _name, _scope };
                            _bundle.start()
                                .store(std::plus<double>{},
                                       static_cast<double>(_end_ns - _begin_ns))
                                .stop();
                        });
                }
                // timemory is disabled in this callback because collecting data in this
                // thread causes strange segmentation faults
            }
        }
    }
}

void
hsa_activity_callback(uint32_t op, activity_record_t* record, void* arg)
{
    static const char* copy_op_name     = "hsa_async_copy";
    static const char* dispatch_op_name = "hsa_dispatch";
    static const char* barrier_op_name  = "hsa_barrier";
    const char**       _name            = nullptr;

    switch(op)
    {
        case HSA_OP_ID_DISPATCH: _name = &dispatch_op_name; break;
        case HSA_OP_ID_COPY: _name = &copy_op_name; break;
        case HSA_OP_ID_BARRIER: _name = &barrier_op_name; break;
        default: break;
    }

    if(!_name) return;

    auto        _begin_ns = record->begin_ns;
    auto        _end_ns   = record->end_ns;
    static auto _scope    = []() {
        auto _v = scope::config{};
        if(get_roctracer_timeline_profile()) _v += scope::timeline{};
        if(get_roctracer_flat_profile()) _v += scope::flat{};
        return _v;
    }();

    auto _func = [_begin_ns, _end_ns, _name]() {
        if(get_use_perfetto())
        {
            TRACE_EVENT_BEGIN("device", perfetto::StaticString{ *_name }, _begin_ns);
            TRACE_EVENT_END("device", _end_ns);
        }
        if(get_use_timemory())
        {
            roctracer_hsa_bundle_t _bundle{ *_name, _scope };
            _bundle.start()
                .store(std::plus<double>{}, static_cast<double>(_end_ns - _begin_ns))
                .stop();
        }
    };

    std::unique_lock<std::mutex> _lk{ tasking::get_roctracer_mutex() };
    tasking::get_roctracer_task_group().exec(_func);

    // timemory is disabled in this callback because collecting data in this thread
    // causes strange segmentation faults
    tim::consume_parameters(arg);
}

void
hip_exec_activity_callbacks(int64_t _tid)
{
    // ROCTRACER_CALL(roctracer_flush_activity());
    tim::auto_lock_t _lk{ get_hip_activity_mutex(_tid) };
    auto&            _async_ops = get_hip_activity_callbacks(_tid);
    for(auto& itr : *_async_ops)
        itr();
    _async_ops->clear();
}

namespace
{
thread_local std::unordered_map<size_t, size_t> gpu_cids = {};
}

// HIP API callback function
void
hip_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    using Device = critical_trace::Device;
    using Phase  = critical_trace::Phase;

    const char* op_name = roctracer_op_string(domain, cid, 0);
    if(op_name == nullptr) op_name = hip_api_name(cid);
    if(op_name == nullptr) return;

    const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
    OMNITRACE_DEBUG("<%-30s id(%u)\tcorrelation_id(%lu) %s>\n", op_name, cid,
                    data->correlation_id,
                    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");

    switch(cid)
    {
        case HIP_API_ID___hipPushCallConfiguration:
        case HIP_API_ID___hipPopCallConfiguration:
        case HIP_API_ID_hipDeviceEnablePeerAccess:
        case HIP_API_ID_hipImportExternalMemory:
        case HIP_API_ID_hipDestroyExternalMemory: return;
        default: break;
    }

    if(data->phase == ACTIVITY_API_PHASE_ENTER)
    {
        switch(cid)
        {
            case HIP_API_ID_hipLaunchKernel:
            case HIP_API_ID_hipLaunchCooperativeKernel:
            {
                const char* _name =
                    hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address,
                                          data->args.hipLaunchKernel.stream);
                if(_name != nullptr)
                {
                    if(get_use_perfetto() || get_use_timemory())
                    {
                        tim::auto_lock_t _lk{ tim::type_mutex<key_data_mutex_t>() };
                        get_roctracer_key_data().emplace(data->correlation_id, _name);
                        get_roctracer_tid_data().emplace(data->correlation_id,
                                                         threading::get_id());
                    }
                }
                break;
            }
            case HIP_API_ID_hipModuleLaunchKernel:
            {
                const char* _name = hipKernelNameRef(data->args.hipModuleLaunchKernel.f);
                if(_name != nullptr)
                {
                    if(get_use_perfetto() || get_use_timemory())
                    {
                        tim::auto_lock_t _lk{ tim::type_mutex<key_data_mutex_t>() };
                        get_roctracer_key_data().emplace(data->correlation_id, _name);
                        get_roctracer_tid_data().emplace(data->correlation_id,
                                                         threading::get_id());
                    }
                }
                break;
            }
            default:
            {
                if(get_use_perfetto() || get_use_timemory())
                {
                    tim::auto_lock_t _lk{ tim::type_mutex<key_data_mutex_t>() };
                    get_roctracer_key_data().emplace(data->correlation_id, op_name);
                    get_roctracer_tid_data().emplace(data->correlation_id,
                                                     threading::get_id());
                }
            }
        }

        if(get_use_perfetto())
        {
            TRACE_EVENT_BEGIN("device", perfetto::StaticString{ op_name });
        }
        if(get_use_timemory())
        {
            auto itr = get_roctracer_hip_data()->emplace(data->correlation_id,
                                                         roctracer_bundle_t{ op_name });
            if(itr.second)
            {
                itr.first->second.start();
            }
            else if(itr.first != get_roctracer_hip_data()->end())
            {
                itr.first->second.stop();
                get_roctracer_hip_data()->erase(itr.first);
            }
        }
        if(get_use_critical_trace())
        {
            auto     _cid        = get_cpu_cid()++;
            uint16_t _depth      = (get_cpu_cid_stack()->empty())
                                       ? get_cpu_cid_stack(0)->size()
                                       : get_cpu_cid_stack()->size() - 1;
            auto     _parent_cid = (get_cpu_cid_stack()->empty())
                                       ? get_cpu_cid_stack(0)->back()
                                       : get_cpu_cid_stack()->back();
            int64_t  _ts         = comp::wall_clock::record();
            add_critical_trace<Device::GPU, Phase::BEGIN>(
                threading::get_id(), _cid, data->correlation_id, _parent_cid, _ts, 0,
                critical_trace::add_hash_id(op_name), _depth);
            tim::auto_lock_t _lk{ tim::type_mutex<cid_data_mutex_t>() };
            get_roctracer_cid_data().emplace(data->correlation_id,
                                             cid_tuple_t{ _cid, _parent_cid, _depth });
        }

        hip_exec_activity_callbacks(threading::get_id());
    }
    else if(data->phase == ACTIVITY_API_PHASE_EXIT)
    {
        hip_exec_activity_callbacks(threading::get_id());

        if(get_use_perfetto())
        {
            TRACE_EVENT_END("device");
        }
        if(get_use_timemory())
        {
            auto _stop = [data](int64_t _tid) {
                auto& _data = get_roctracer_hip_data(_tid);
                auto  itr   = _data->find(data->correlation_id);
                if(itr != get_roctracer_hip_data()->end())
                {
                    itr->second.stop();
                    _data->erase(itr);
                    return true;
                }
                return false;
            };
            if(!_stop(threading::get_id()))
            {
                for(size_t i = 0; i < max_supported_threads; ++i)
                {
                    if(_stop(i)) break;
                }
            }
        }
        if(get_use_critical_trace())
        {
            uint16_t _depth      = 0;
            uint64_t _cid        = 0;
            uint64_t _parent_cid = 0;
            {
                tim::auto_lock_t _lk{ tim::type_mutex<cid_data_mutex_t>() };
                std::tie(_cid, _parent_cid, _depth) =
                    get_roctracer_cid_data().at(data->correlation_id);
            }
            int64_t _ts = comp::wall_clock::record();
            add_critical_trace<Device::GPU, Phase::END>(
                threading::get_id(), _cid, data->correlation_id, _parent_cid, _ts, _ts,
                critical_trace::add_hash_id(op_name), _depth);
        }
    }
    tim::consume_parameters(arg);
}

// Activity tracing callback
void
hip_activity_callback(const char* begin, const char* end, void*)
{
    using Device = critical_trace::Device;
    using Phase  = critical_trace::Phase;

    if(!trait::runtime_enabled<comp::roctracer>::get()) return;
    static auto _kernel_names        = std::unordered_map<const char*, std::string>{};
    static auto _indexes             = std::unordered_map<uint64_t, int>{};
    const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
    const roctracer_record_t* end_record =
        reinterpret_cast<const roctracer_record_t*>(end);

    OMNITRACE_DEBUG("Activity records:\n");

    while(record < end_record)
    {
        const char* op_name =
            roctracer_op_string(record->domain, record->correlation_id, 0);
        if(op_name == nullptr) op_name = hip_api_name(record->correlation_id);

        if(op_name != nullptr)
        {
            OMNITRACE_DEBUG("\t%-30s\tcorrelation_id(%6lu) time_ns(%12lu:%12lu) "
                            "delta_ns(%12lu) device_id(%d) "
                            "stream_id(%lu)\n",
                            op_name, record->correlation_id, record->begin_ns,
                            record->end_ns, (record->end_ns - record->begin_ns),
                            record->device_id, record->queue_id);
        }

        auto        _begin_ns = record->begin_ns;
        auto        _end_ns   = record->end_ns;
        auto        _corr_id  = record->correlation_id;
        static auto _scope    = []() {
            auto _v = scope::config{};
            if(get_roctracer_timeline_profile()) _v += scope::timeline{};
            if(get_roctracer_flat_profile()) _v += scope::flat{};
            return _v;
        }();

        auto& _keys = get_roctracer_key_data();
        auto& _cids = get_roctracer_cid_data();
        auto& _tids = get_roctracer_tid_data();

        int16_t     _depth          = 0;                     // depth of kernel launch
        int64_t     _tid            = 0;                     // thread id
        uint64_t    _cid            = 0;                     // correlation id
        uint64_t    _pcid           = 0;                     // parent corr_id
        auto        _laps           = _indexes[_corr_id]++;  // see note #1
        const char* _name           = nullptr;
        bool        _found          = false;
        bool        _critical_trace = get_use_critical_trace();

        {
            tim::auto_lock_t _lk{ tim::type_mutex<key_data_mutex_t>() };
            if(_tids.find(_corr_id) != _tids.end())
            {
                _found   = true;
                _tid     = _tids.at(_corr_id);
                auto itr = _keys.find(_corr_id);
                if(itr != _keys.end()) _name = itr->second;
            }
        }

        if(_critical_trace)
        {
            tim::auto_lock_t _lk{ tim::type_mutex<cid_data_mutex_t>() };
            if(_cids.find(_corr_id) != _cids.end())
                std::tie(_cid, _pcid, _depth) = _cids.at(_corr_id);
            else
                _critical_trace = false;
        }

        auto _func = [_critical_trace, _depth, _tid, _cid, _laps, _begin_ns, _end_ns,
                      _corr_id, _name]() {
            // NOTE #1: we get two measurements for 1 kernel so we need to
            // tweak the number of laps for the wall-clock component
            if(_name != nullptr)
            {
                if(get_use_perfetto())
                {
                    if(_kernel_names.find(_name) == _kernel_names.end())
                        _kernel_names.emplace(_name, tim::demangle(_name));
                    TRACE_EVENT_BEGIN(
                        "device",
                        perfetto::StaticString{ _kernel_names.at(_name).c_str() },
                        _begin_ns);
                    TRACE_EVENT_END("device", _end_ns);
                }
                if(get_use_timemory())
                {
                    roctracer_bundle_t _bundle{ _name, _scope };
                    _bundle.start()
                        .store(std::plus<double>{},
                               static_cast<double>(_end_ns - _begin_ns))
                        .stop()
                        .get<comp::wall_clock>([&](comp::wall_clock* wc) {
                            wc->set_value(_end_ns - _begin_ns);
                            wc->set_accum(_end_ns - _begin_ns);
                            if(_laps % 2 == 1)
                            {
                                // below is a hack bc we get two measurements for 1 kernel
                                wc->set_laps(0);

                                auto itr = wc->get_iterator();
                                if(itr && itr->data().get_laps() == 0)
                                {
                                    wc->set_is_invalid(true);
                                    itr->data().set_is_invalid(true);
                                }
                            }
                            return wc;
                        });
                    _bundle.pop();
                }
                if(_critical_trace)
                {
                    auto     _hash = critical_trace::add_hash_id(_name);
                    uint16_t _prio = _laps + 1;  // priority
                    add_critical_trace<Device::GPU, Phase::DELTA, false>(
                        _tid, _cid, _corr_id, _cid, _begin_ns, _end_ns, _hash, _depth + 1,
                        _prio);
                }
            }
        };

        if(_found)
        {
            auto&            _async_ops = get_hip_activity_callbacks(_tid);
            tim::auto_lock_t _lk{ get_hip_activity_mutex(_tid) };
            _async_ops->emplace_back(std::move(_func));
        }

        ROCTRACER_CALL(roctracer_next_record(record, &record));
    }
}

bool&
roctracer_is_setup()
{
    static bool _v = false;
    return _v;
}

using roctracer_functions_t = std::vector<std::pair<std::string, std::function<void()>>>;

roctracer_functions_t&
roctracer_setup_routines()
{
    static auto _v = roctracer_functions_t{};
    return _v;
}

roctracer_functions_t&
roctracer_tear_down_routines()
{
    static auto _v = roctracer_functions_t{};
    return _v;
}
}  // namespace omnitrace
