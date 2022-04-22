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

#include "library/components/roctracer_callbacks.hpp"
#include "library.hpp"
#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/debug.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"

#include <timemory/backends/threading.hpp>

#include <chrono>
#include <cstdint>

#include <roctracer_ext.h>
#include <roctracer_hcc.h>
#include <roctracer_hip.h>

#define AMD_INTERNAL_BUILD 1
#include <roctracer_hsa.h>

TIMEMORY_DEFINE_API(roctracer)
namespace omnitrace
{
namespace api = tim::api;

int64_t
get_clock_skew()
{
    static auto _v = []() {
        // synchronize timestamps
        // We'll take a CPU timestamp before and after taking a GPU timestmp, then
        // take the average of those two, hoping that it's roughly at the same time
        // as the GPU timestamp.
        auto _now = []() {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now().time_since_epoch())
                .count();
        };
        uint64_t _cpu_ts = _now();
        uint64_t _gpu_ts;
        roctracer_get_timestamp(&_gpu_ts);
        _cpu_ts += _now();
        _cpu_ts /= 2;

        // assume CPU timestamp is greater than GPU
        OMNITRACE_BASIC_VERBOSE_F(2, "CPU timestamp: %lu\n", _cpu_ts);
        OMNITRACE_BASIC_VERBOSE_F(2, "HIP timestamp: %lu\n", _gpu_ts);
        auto _diff = static_cast<int64_t>(_cpu_ts) - static_cast<int64_t>(_gpu_ts);
        OMNITRACE_BASIC_VERBOSE_F(2, "CPU/HIP timestamp skew: %li\n", _diff);
        return _diff;
    }();
    return _v;
}

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
    using thread_data_t = thread_data<data_t, api::roctracer>;
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
    using thread_data_t = thread_data<std::vector<std::function<void()>>, api::roctracer>;
    static auto& _v     = thread_data_t::instances(thread_data_t::construct_on_init{});
    return _v.at(_tid);
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

    static thread_local int64_t begin_timestamp = 0;
    static auto                 _scope          = []() {
        auto _v = scope::config{};
        if(get_roctracer_timeline_profile()) _v += scope::timeline{};
        if(get_roctracer_flat_profile()) _v += scope::flat{};
        return _v;
    }();

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
                begin_timestamp = comp::wall_clock::record();
            }
            else
            {
                const auto* _name         = roctracer_op_string(domain, cid, 0);
                const auto  end_timestamp = (cid == HSA_API_ID_hsa_shut_down)
                                                ? begin_timestamp
                                                : comp::wall_clock::record();

                if(begin_timestamp > end_timestamp) return;

                if(get_use_perfetto())
                {
                    TRACE_EVENT_BEGIN("device", perfetto::StaticString{ _name },
                                      static_cast<uint64_t>(begin_timestamp));
                    TRACE_EVENT_END("device", static_cast<uint64_t>(end_timestamp));
                }

                if(get_use_timemory())
                {
                    std::unique_lock<std::mutex> _lk{ tasking::get_roctracer_mutex() };
                    auto                         _beg_ns = begin_timestamp;
                    auto                         _end_ns = end_timestamp;
                    if(tasking::get_roctracer_task_group().pool())
                        tasking::get_roctracer_task_group().exec(
                            [_name, _beg_ns, _end_ns]() {
                                roctracer_hsa_bundle_t _bundle{ _name, _scope };
                                _bundle.start()
                                    .store(std::plus<double>{},
                                           static_cast<double>(_end_ns - _beg_ns))
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
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    static const char* copy_op_name     = "hsa_async_copy";
    static const char* dispatch_op_name = "hsa_dispatch";
    static const char* barrier_op_name  = "hsa_barrier";
    const char**       _name            = nullptr;

    static thread_local auto _once = (threading::set_thread_name("omni.roctracer"), true);
    (void) _once;

    switch(op)
    {
        case HSA_OP_ID_DISPATCH: _name = &dispatch_op_name; break;
        case HSA_OP_ID_COPY: _name = &copy_op_name; break;
        case HSA_OP_ID_BARRIER: _name = &barrier_op_name; break;
        default: break;
    }

    if(!_name) return;

    auto        _beg_ns = record->begin_ns + get_clock_skew();
    auto        _end_ns = record->end_ns + get_clock_skew();
    static auto _scope  = []() {
        auto _v = scope::config{};
        if(get_roctracer_timeline_profile()) _v += scope::timeline{};
        if(get_roctracer_flat_profile()) _v += scope::flat{};
        return _v;
    }();

    auto _func = [_beg_ns, _end_ns, _name]() {
        if(get_use_perfetto())
        {
            TRACE_EVENT_BEGIN("device", perfetto::StaticString{ *_name },
                              static_cast<uint64_t>(_beg_ns));
            TRACE_EVENT_END("device", static_cast<uint64_t>(_end_ns));
        }
        if(get_use_timemory())
        {
            roctracer_hsa_bundle_t _bundle{ *_name, _scope };
            _bundle.start()
                .store(std::plus<double>{}, static_cast<double>(_end_ns - _beg_ns))
                .stop();
        }
    };

    std::unique_lock<std::mutex> _lk{ tasking::get_roctracer_mutex() };
    if(tasking::get_roctracer_task_group().pool())
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

    (void) get_clock_skew();

    using Device = critical_trace::Device;
    using Phase  = critical_trace::Phase;

    assert(domain == ACTIVITY_DOMAIN_HIP_API);
    const char* op_name = roctracer_op_string(domain, cid, 0);
    if(op_name == nullptr) op_name = hip_api_name(cid);
    if(op_name == nullptr) return;
    assert(std::string{ op_name } == std::string{ hip_api_name(cid) });

    switch(cid)
    {
        case HIP_API_ID___hipPushCallConfiguration:
        case HIP_API_ID___hipPopCallConfiguration:
        case HIP_API_ID_hipDeviceEnablePeerAccess:
#if OMNITRACE_HIP_VERSION_MAJOR > 4 ||                                                   \
    (OMNITRACE_HIP_VERSION_MAJOR == 4 && OMNITRACE_HIP_VERSION_MINOR >= 3)
        case HIP_API_ID_hipImportExternalMemory:
        case HIP_API_ID_hipDestroyExternalMemory:
#endif
            return;
        default: break;
    }

    const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
    OMNITRACE_DEBUG("<%-30s id(%u)\tcorrelation_id(%lu) %s>\n", op_name, cid,
                    data->correlation_id,
                    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");

    int64_t  _ts         = comp::wall_clock::record();
    auto     _tid        = threading::get_id();
    uint64_t _cid        = 0;
    uint64_t _parent_cid = 0;
    uint16_t _depth      = 0;
    auto     _corr_id    = data->correlation_id;

    if(data->phase == ACTIVITY_API_PHASE_ENTER)
    {
        const char* _name = nullptr;
        switch(cid)
        {
            case HIP_API_ID_hipLaunchKernel:
            case HIP_API_ID_hipLaunchCooperativeKernel:
            {
                _name = hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address,
                                              data->args.hipLaunchKernel.stream);
                break;
            }
            case HIP_API_ID_hipHccModuleLaunchKernel:
            {
                _name = hipKernelNameRef(data->args.hipHccModuleLaunchKernel.f);
                break;
            }
            case HIP_API_ID_hipModuleLaunchKernel:
            {
                _name = hipKernelNameRef(data->args.hipModuleLaunchKernel.f);
                break;
            }
            case HIP_API_ID_hipExtModuleLaunchKernel:
            {
                _name = hipKernelNameRef(data->args.hipExtModuleLaunchKernel.f);
                break;
            }
            case HIP_API_ID_hipExtLaunchKernel:
            {
                _name =
                    hipKernelNameRefByPtr(data->args.hipExtLaunchKernel.function_address,
                                          data->args.hipLaunchKernel.stream);
                break;
            }
            default: break;
        }

        if(_name != nullptr)
        {
            if(get_use_perfetto() || get_use_timemory() || get_use_rocm_smi())
            {
                tim::auto_lock_t _lk{ tim::type_mutex<key_data_mutex_t>() };
                get_roctracer_key_data().emplace(_corr_id, _name);
                get_roctracer_tid_data().emplace(_corr_id, _tid);
            }
        }

        std::tie(_cid, _parent_cid, _depth) = create_cpu_cid_entry();

        if(get_use_perfetto())
        {
            TRACE_EVENT_BEGIN(
                "host", perfetto::StaticString{ op_name }, static_cast<uint64_t>(_ts),
                perfetto::Flow::ProcessScoped(_cid), "pcid", _parent_cid, "cid", _cid,
                "tid", _tid, "depth", _depth, "corr_id", _corr_id);
        }
        if(get_use_timemory())
        {
            auto itr = get_roctracer_hip_data()->emplace(_corr_id,
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
        if(get_use_critical_trace() || get_use_rocm_smi())
        {
            add_critical_trace<Device::CPU, Phase::BEGIN>(
                _tid, _cid, _corr_id, _parent_cid, _ts, 0,
                critical_trace::add_hash_id(op_name), _depth);
        }

        {
            tim::auto_lock_t _lk{ tim::type_mutex<cid_data_mutex_t>() };
            get_roctracer_cid_data().emplace(_corr_id,
                                             cid_tuple_t{ _cid, _parent_cid, _depth });
        }

        hip_exec_activity_callbacks(_tid);
    }
    else if(data->phase == ACTIVITY_API_PHASE_EXIT)
    {
        hip_exec_activity_callbacks(_tid);

        {
            tim::auto_lock_t _lk{ tim::type_mutex<cid_data_mutex_t>() };
            std::tie(_cid, _parent_cid, _depth) = get_roctracer_cid_data().at(_corr_id);
        }

        if(get_use_perfetto())
        {
            TRACE_EVENT_END("host", static_cast<uint64_t>(_ts));
        }
        if(get_use_timemory())
        {
            auto _stop = [&_corr_id](int64_t _tid) {
                auto& _data = get_roctracer_hip_data(_tid);
                auto  itr   = _data->find(_corr_id);
                if(itr != get_roctracer_hip_data()->end())
                {
                    itr->second.stop();
                    _data->erase(itr);
                    return true;
                }
                return false;
            };
            if(!_stop(_tid))
            {
                for(size_t i = 0; i < max_supported_threads; ++i)
                {
                    if(_stop(i)) break;
                }
            }
        }
        if(get_use_critical_trace() || get_use_rocm_smi())
        {
            add_critical_trace<Device::CPU, Phase::END>(
                _tid, _cid, _corr_id, _parent_cid, _ts, _ts,
                critical_trace::add_hash_id(op_name), _depth);
        }
    }
    tim::consume_parameters(arg);
}

// Activity tracing callback
void
hip_activity_callback(const char* begin, const char* end, void*)
{
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    sampling::block_signals();

    static thread_local auto _once = (threading::set_thread_name("omni.roctracer"), true);
    (void) _once;

    using Device = critical_trace::Device;
    using Phase  = critical_trace::Phase;

    if(!trait::runtime_enabled<comp::roctracer>::get()) return;
    static auto _kernel_names        = std::unordered_map<const char*, std::string>{};
    static auto _indexes             = std::unordered_map<uint64_t, int>{};
    const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
    const roctracer_record_t* end_record =
        reinterpret_cast<const roctracer_record_t*>(end);

    auto&& _advance_record = [&record]() {
        ROCTRACER_CALL(roctracer_next_record(record, &record));
    };

    while(record < end_record)
    {
        // make sure every iteration advances regardless of where return point happens
        scope::destructor _next_dtor{ _advance_record };

        // OMNITRACE_CI will enable these asserts and should fail if something relevant
        // changes
        assert(HIP_OP_ID_DISPATCH == 0);
        assert(HIP_OP_ID_COPY == 1);
        assert(HIP_OP_ID_BARRIER == 2);
        assert(record->domain == ACTIVITY_DOMAIN_HIP_OPS);

        if(record->domain != ACTIVITY_DOMAIN_HIP_OPS) continue;
        if(record->op > HIP_OP_ID_BARRIER) continue;

        const char* op_name =
            roctracer_op_string(record->domain, record->op, record->kind);

        uint64_t    _beg_ns  = record->begin_ns + get_clock_skew();
        uint64_t    _end_ns  = record->end_ns + get_clock_skew();
        auto        _corr_id = record->correlation_id;
        static auto _scope   = []() {
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
        bool        _critical_trace = get_use_critical_trace() || get_use_rocm_smi();

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

        if(_name == nullptr && op_name == nullptr) continue;
        if(_name == nullptr) _name = op_name;

        if(_critical_trace)
        {
            tim::auto_lock_t _lk{ tim::type_mutex<cid_data_mutex_t>() };
            if(_cids.find(_corr_id) != _cids.end())
                std::tie(_cid, _pcid, _depth) = _cids.at(_corr_id);
            else
                _critical_trace = false;
        }

        {
            static size_t _n = 0;
            OMNITRACE_VERBOSE_F(
                2,
                "%4zu :: %-20s :: %-20s :: correlation_id(%6lu) time_ns(%12lu:%12lu) "
                "delta_ns(%12lu) device_id(%d) stream_id(%lu) proc_id(%u) thr_id(%lu)\n",
                _n++, op_name, _name, record->correlation_id, _beg_ns, _end_ns,
                (_end_ns - _beg_ns), record->device_id, record->queue_id,
                record->process_id, _tid);
        }

        // execute this on this thread bc of how perfetto visualization works
        if(get_use_perfetto())
        {
            static auto _op_id_names =
                std::array<const char*, 3>{ "DISPATCH", "COPY", "BARRIER" };

            if(_kernel_names.find(_name) == _kernel_names.end())
                _kernel_names.emplace(_name, tim::demangle(_name));

            assert(_end_ns > _beg_ns);
            TRACE_EVENT_BEGIN(
                "device", perfetto::StaticString{ _kernel_names.at(_name).c_str() },
                _beg_ns, perfetto::Flow::ProcessScoped(_cid), "corr_id",
                record->correlation_id, "device", record->device_id, "queue",
                record->queue_id, "op", _op_id_names.at(record->op));
            TRACE_EVENT_END("device", _end_ns);
        }

        auto _func = [_critical_trace, _depth, _tid, _cid, _laps, _beg_ns, _end_ns,
                      _corr_id, _name]() {
            // NOTE #1: we get two measurements for 1 kernel so we need to
            // tweak the number of laps for the wall-clock component
            if(_name != nullptr)
            {
                if(get_use_timemory())
                {
                    roctracer_bundle_t _bundle{ _name, _scope };
                    _bundle.start()
                        .store(std::plus<double>{},
                               static_cast<double>(_end_ns - _beg_ns))
                        .stop()
                        .get<comp::wall_clock>([&](comp::wall_clock* wc) {
                            wc->set_value(_end_ns - _beg_ns);
                            wc->set_accum(_end_ns - _beg_ns);
                            return wc;
                        });
                    _bundle.pop();
                }
                if(_critical_trace)
                {
                    auto     _hash = critical_trace::add_hash_id(_name);
                    uint16_t _prio = _laps + 1;  // priority
                    add_critical_trace<Device::GPU, Phase::DELTA, false>(
                        _tid, _cid, _corr_id, _cid, _beg_ns, _end_ns, _hash, _depth + 1,
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
roctracer_shutdown_routines()
{
    static auto _v = roctracer_functions_t{};
    return _v;
}
}  // namespace omnitrace

#include "library/components/rocm_smi.hpp"

using namespace omnitrace;

extern "C" bool
omnitrace_init_tooling_hidden() OMNITRACE_HIDDEN_API;

// HSA-runtime tool on-load method
extern "C"
{
    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names) OMNITRACE_VISIBILITY("default");
    void OnUnload() OMNITRACE_VISIBILITY("default");

    bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                const char* const* failed_tool_names)
    {
        pthread_gotcha::push_enable_sampling_on_child_threads(false);
        OMNITRACE_CONDITIONAL_BASIC_PRINT_F(get_debug_env() || get_verbose_env() > 0,
                                            "\n");
        tim::consume_parameters(table, runtime_version, failed_tool_count,
                                failed_tool_names);

        if(get_state() < State::Active) omnitrace_init_tooling_hidden();

        auto _setup = [=]() {
            try
            {
                OMNITRACE_CONDITIONAL_BASIC_PRINT_F(get_debug() || get_verbose() > 1,
                                                    "setting up HSA...\n");

                // const char* output_prefix = getenv("ROCP_OUTPUT_DIR");
                const char* output_prefix = nullptr;

                bool trace_hsa_api = get_trace_hsa_api();

                // Enable HSA API callbacks/activity
                if(trace_hsa_api)
                {
                    std::vector<std::string> hsa_api_vec =
                        tim::delimit(get_trace_hsa_api_types());

                    // initialize HSA tracing
                    roctracer_set_properties(ACTIVITY_DOMAIN_HSA_API, (void*) table);

                    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug() || get_verbose() > 1,
                                                      "    HSA-trace(");
                    if(!hsa_api_vec.empty())
                    {
                        for(const auto& itr : hsa_api_vec)
                        {
                            uint32_t    cid = HSA_API_ID_NUMBER;
                            const char* api = itr.c_str();
                            ROCTRACER_CALL(roctracer_op_code(ACTIVITY_DOMAIN_HSA_API, api,
                                                             &cid, nullptr));
                            ROCTRACER_CALL(roctracer_enable_op_callback(
                                ACTIVITY_DOMAIN_HSA_API, cid, hsa_api_callback, nullptr));

                            OMNITRACE_CONDITIONAL_BASIC_PRINT(
                                get_debug() || get_verbose() > 1, " %s", api);
                        }
                    }
                    else
                    {
                        ROCTRACER_CALL(roctracer_enable_domain_callback(
                            ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, nullptr));
                    }
                    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug() || get_verbose() > 1,
                                                      "\n");
                }

                bool trace_hsa_activity = get_trace_hsa_activity();
                // Enable HSA GPU activity
                if(trace_hsa_activity)
                {
                    // initialize HSA tracing
                    ::roctracer::hsa_ops_properties_t ops_properties{
                        table,
                        reinterpret_cast<activity_async_callback_t>(
                            hsa_activity_callback),
                        nullptr, output_prefix
                    };
                    roctracer_set_properties(ACTIVITY_DOMAIN_HSA_OPS, &ops_properties);

                    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug() || get_verbose() > 1,
                                                      "    HSA-activity-trace()\n");
                    ROCTRACER_CALL(roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS,
                                                                HSA_OP_ID_COPY));
                }
            } catch(std::exception& _e)
            {
                OMNITRACE_BASIC_PRINT("Exception was thrown in HSA setup: %s\n",
                                      _e.what());
            }
        };

        auto _shutdown = []() {
            OMNITRACE_DEBUG_F("roctracer_disable_domain_callback\n");
            ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));

            OMNITRACE_DEBUG_F("roctracer_disable_op_activity\n");
            ROCTRACER_CALL(
                roctracer_disable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY));
        };

        comp::roctracer::add_setup("hsa", std::move(_setup));
        comp::roctracer::add_shutdown("hsa", std::move(_shutdown));

        rocm_smi::set_state(State::Active);
        comp::roctracer::setup();

        pthread_gotcha::pop_enable_sampling_on_child_threads();
        return true;
    }

    // HSA-runtime on-unload method
    void OnUnload()
    {
        OMNITRACE_DEBUG_F("\n");
        rocm_smi::set_state(State::Finalized);
        comp::roctracer::shutdown();
    }
}
