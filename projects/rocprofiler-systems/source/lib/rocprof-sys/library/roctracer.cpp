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

#include "library/roctracer.hpp"
#include "binary/analysis.hpp"
#include "core/components/fwd.hpp"
#include "core/concepts.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/locking.hpp"
#include "library/components/category_region.hpp"
#include "library/runtime.hpp"
#include "library/sampling.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"
#include "library/tracing.hpp"

#include <timemory/backends/cpu.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/hash/types.hpp>
#include <timemory/utility/types.hpp>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <tuple>

#include <roctracer_ext.h>
#include <roctracer_hip.h>
#include <roctracer_roctx.h>

#if ROCPROFSYS_HIP_VERSION < 50300
#    include <roctracer_hcc.h>
#endif

#define AMD_INTERNAL_BUILD 1
#include <roctracer_hsa.h>

#if __has_include(<hip/amd_detail/hip_prof_str.h>) || (defined(ROCPROFSYS_USE_HIP) && ROCPROFSYS_USE_HIP > 0)
#    include <hip/amd_detail/hip_prof_str.h>
#    define ROCPROFSYS_HIP_API_ARGS 1
#else
#    define ROCPROFSYS_HIP_API_ARGS 0
#endif

TIMEMORY_DEFINE_API(roctracer)
namespace rocprofsys
{
namespace
{
template <typename Tp>
auto&
roctracer_type_mutex()
{
    return tim::type_mutex<Tp, category::roctracer, max_supported_threads,
                           locking::atomic_mutex>();
}

std::string
hip_api_string(hip_api_id_t id, const hip_api_data_t* data)
{
#if ROCPROFSYS_HIP_API_ARGS > 0
    std::string _v = hipApiString(id, data);
    if(_v.empty()) return _v;
    auto _pbeg = _v.find('(');
    if(_pbeg == std::string::npos) return _v;
    auto _pend = _v.find_last_of(')');
    if(_pend == std::string::npos || _pbeg >= _pend) return _v;
    auto _n = (_pend - _pbeg - 1);
    return _v.substr(_pbeg + 1, _n);
#else
    tim::consume_parameters(id, data);
#endif
}

int&
get_current_device()
{
    static thread_local int _v = 1;
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
    using data_t        = std::unordered_map<uint64_t, roctracer_hip_bundle_t>;
    using thread_data_t = thread_data<data_t, category::roctracer>;
    return thread_data_t::instance(construct_on_thread{ _tid });
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

auto&
get_hip_activity_callbacks(int64_t _tid = threading::get_id())
{
    using thread_data_t =
        thread_data<std::vector<std::function<void()>>, category::roctracer>;
    return thread_data_t::instance(construct_on_thread{ _tid });
}

size_t
get_hip_activity_callbacks_size()
{
    using thread_data_t =
        thread_data<std::vector<std::function<void()>>, category::roctracer>;
    return thread_data_t::size();
}

using hip_activity_mutex_t = std::decay_t<decltype(get_hip_activity_callbacks())>;
using key_data_mutex_t     = std::decay_t<decltype(get_roctracer_key_data())>;

auto&
get_hip_activity_mutex(int64_t _tid = threading::get_id())
{
    return tim::type_mutex<hip_activity_mutex_t, category::roctracer,
                           max_supported_threads, locking::atomic_mutex>(
        _tid % max_supported_threads);
}
}  // namespace

//
int64_t
get_clock_skew()
{
    static auto _use = tim::get_env("ROCPROFSYS_USE_ROCTRACER_CLOCK_SKEW", true);
    if(!_use) return 0;
    static auto _v = []() {
        namespace cpu = tim::cpu;
        // synchronize timestamps
        // We'll take a CPU timestamp before and after taking a GPU timestmp, then
        // take the average of those two, hoping that it's roughly at the same time
        // as the GPU timestamp.
        static auto _cpu_now = []() {
            cpu::fence();
            return comp::wall_clock::record();
        };

        static auto _gpu_now = []() {
            cpu::fence();
            uint64_t _ts = 0;
            ROCPROFSYS_ROCTRACER_CALL(roctracer_get_timestamp(&_ts));
            return _ts;
        };

        do
        {
            // warm up cache and allow for any static initialization
            (void) _cpu_now();
            (void) _gpu_now();
        } while(false);

        auto _compute = [](volatile uint64_t& _cpu_ts, volatile uint64_t& _gpu_ts) {
            _cpu_ts = 0;
            _gpu_ts = 0;
            _cpu_ts += _cpu_now() / 2;
            _gpu_ts += _gpu_now() / 1;
            _cpu_ts += _cpu_now() / 2;
            return static_cast<int64_t>(_cpu_ts) - static_cast<int64_t>(_gpu_ts);
        };
        constexpr int64_t _n       = 10;
        int64_t           _cpu_ave = 0;
        int64_t           _gpu_ave = 0;
        int64_t           _diff    = 0;
        for(int64_t i = 0; i < _n; ++i)
        {
            volatile uint64_t _cpu_ts = 0;
            volatile uint64_t _gpu_ts = 0;
            _diff += _compute(_cpu_ts, _gpu_ts);
            _cpu_ave += _cpu_ts / _n;
            _gpu_ave += _gpu_ts / _n;
        }
        ROCPROFSYS_BASIC_VERBOSE(2, "CPU timestamp: %li\n", _cpu_ave);
        ROCPROFSYS_BASIC_VERBOSE(2, "HIP timestamp: %li\n", _gpu_ave);
        ROCPROFSYS_BASIC_VERBOSE(1, "CPU/HIP timestamp skew: %li (used: %s)\n", _diff,
                                 _use ? "yes" : "no");
        _diff /= _n;
        return _diff;
    }();
    return _v;
}

// HSA API callback function
void
hsa_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    (void) arg;
    const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
    ROCPROFSYS_CONDITIONAL_PRINT_F(
        get_debug() && get_verbose() >= 2, "<%-30s id(%u)\tcorrelation_id(%lu) %s>\n",
        roctracer_op_string(domain, cid, 0), cid, data->correlation_id,
        (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");

    static thread_local int64_t begin_timestamp = 0;

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
                    uint64_t _beg_ts = begin_timestamp;
                    uint64_t _end_ts = end_timestamp;
                    tracing::push_perfetto_ts(category::rocm_hsa{}, _name, _beg_ts,
                                              [&](::perfetto::EventContext ctx) {
                                                  if(config::get_perfetto_annotations())
                                                  {
                                                      tracing::add_perfetto_annotation(
                                                          ctx, "begin_ns", _beg_ts);
                                                  }
                                              });
                    tracing::pop_perfetto_ts(category::rocm_hsa{}, _name, _end_ts,
                                             [&](::perfetto::EventContext ctx) {
                                                 if(config::get_perfetto_annotations())
                                                 {
                                                     tracing::add_perfetto_annotation(
                                                         ctx, "end_ns", _end_ts);
                                                 }
                                             });
                }

                if(get_use_timemory())
                {
                    auto _beg_ns = begin_timestamp;
                    auto _end_ns = end_timestamp;
                    if(tasking::roctracer::get_task_group().pool())
                        tasking::roctracer::get_task_group().exec(
                            [_name, _beg_ns, _end_ns]() {
                                roctracer_hsa_bundle_t _bundle{ _name };
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
hsa_activity_callback(uint32_t op, const void* vrecord, void* arg)
{
    const auto* record = static_cast<const activity_record_t*>(vrecord);

    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto&& _protect = comp::roctracer::protect_flush_activity();
    (void) _protect;

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

    ROCPROFSYS_CI_FAIL(_name == nullptr, "Error! HSA operation type not handled: %u\n",
                       op);

    if(!_name) return;

    auto _beg_ns = record->begin_ns + get_clock_skew();
    auto _end_ns = record->end_ns + get_clock_skew();

    if(get_use_perfetto())
    {
        uint64_t _beg = _beg_ns;
        uint64_t _end = _end_ns;
        tracing::push_perfetto_ts(
            category::device_hsa{}, *_name, _beg, [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                {
                    tracing::add_perfetto_annotation(ctx, "begin_ns", _beg);
                }
            });
        tracing::pop_perfetto_ts(
            category::device_hsa{}, *_name, _end, [&](::perfetto::EventContext ctx) {
                if(config::get_perfetto_annotations())
                {
                    tracing::add_perfetto_annotation(ctx, "end_ns", _end);
                }
            });
    }

    auto _func = [_beg_ns, _end_ns, _name]() {
        if(get_use_timemory())
        {
            roctracer_hsa_bundle_t _bundle{ *_name };
            _bundle.start()
                .store(std::plus<double>{}, static_cast<double>(_end_ns - _beg_ns))
                .stop();
        }
    };

    if(tasking::roctracer::get_task_group().pool())
        tasking::roctracer::get_task_group().exec(_func);

    // timemory is disabled in this callback because collecting data in this thread
    // causes strange segmentation faults
    tim::consume_parameters(arg);
}

void
hip_exec_activity_callbacks(int64_t _tid)
{
    // guard against initialization of structure when trying to exec
    if(static_cast<size_t>(_tid) >= get_hip_activity_callbacks_size()) return;

    // ROCPROFSYS_ROCTRACER_CALL(roctracer_flush_activity());
    locking::atomic_lock _lk{ get_hip_activity_mutex(_tid) };
    auto&                _async_ops = get_hip_activity_callbacks(_tid);
    if(!_async_ops) return;
    for(auto& itr : *_async_ops)
    {
        if(itr) itr();
    }
    _async_ops->clear();
}

namespace
{
thread_local std::unordered_map<size_t, size_t> gpu_crit_cids = {};
}

void
roctx_api_callback(uint32_t domain, uint32_t cid, const void* callback_data,
                   void* /*arg*/)
{
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    if(domain != ACTIVITY_DOMAIN_ROCTX) return;

    static auto _range_map  = std::unordered_map<roctx_range_id_t, std::string>{};
    static auto _range_lock = locking::atomic_mutex{};
    const auto* _data       = reinterpret_cast<const roctx_api_data_t*>(callback_data);
    static thread_local auto _range_stack = std::vector<std::string>{};

    switch(cid)
    {
        case ROCTX_API_ID_roctxRangePushA:
        {
            if(_data->args.message)
            {
                auto& itr = _range_stack.emplace_back(std::string{ _data->args.message });
                component::category_region<category::rocm_roctx>::start(itr.c_str());
            }
            break;
        }
        case ROCTX_API_ID_roctxRangePop:
        {
            if(!_range_stack.empty())
            {
                auto& itr = _range_stack.back();
                component::category_region<category::rocm_roctx>::stop(itr.c_str());
                _range_stack.pop_back();
            }
            else
            {
                ROCPROFSYS_THROW("Error! roctxRangePop stack is empty! Expected "
                                 "roctxRangePush/roctxRangePop on same thread\n");
            }
            break;
        }
        case ROCTX_API_ID_roctxRangeStartA:
        {
            {
                locking::atomic_lock _lk{ _range_lock, std::defer_lock };
                if(!_lk.owns_lock()) _lk.lock();
                _range_map.emplace(roctx_range_id_t{ _data->args.id },
                                   std::string{ _data->args.message });
            }

            component::category_region<category::rocm_roctx>::start(_data->args.message);
            break;
        }
        case ROCTX_API_ID_roctxRangeStop:
        {
            std::string_view _message = {};
            {
                locking::atomic_lock _lk{ _range_lock, std::defer_lock };
                if(!_lk.owns_lock()) _lk.lock();
                auto itr = _range_map.find(roctx_range_id_t{ _data->args.id });
                ROCPROFSYS_CI_THROW(itr == _range_map.end(),
                                    "Error! could not find range with id %lu\n",
                                    _data->args.id);
                if(itr == _range_map.end())
                {
                    ROCPROFSYS_VERBOSE(0, "Warning! could not find range with id %lu\n",
                                       _data->args.id);
                    return;
                }
                else
                {
                    _message = itr->second;
                }
            }

            if(!_message.empty())
            {
                component::category_region<category::rocm_roctx>::stop(_message.data());
            }

            break;
        }
        case ROCTX_API_ID_roctxMarkA:
        {
            if(_data->args.message)
            {
                component::category_region<category::rocm_roctx>::mark(
                    _data->args.message);
            }
            break;
        }
        default: break;
    }
}

// HIP API callback function
void
hip_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

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
#if ROCPROFSYS_HIP_VERSION_MAJOR > 4 ||                                                  \
    (ROCPROFSYS_HIP_VERSION_MAJOR == 4 && ROCPROFSYS_HIP_VERSION_MINOR >= 3)
        case HIP_API_ID_hipImportExternalMemory:
        case HIP_API_ID_hipDestroyExternalMemory:
#endif
            return;
        default: break;
    }

    const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
    ROCPROFSYS_CONDITIONAL_PRINT_F(
        get_debug() && get_verbose() >= 2, "<%-30s id(%u)\tcorrelation_id(%lu) %s>\n",
        op_name, cid, data->correlation_id,
        (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");

    int64_t  _ts              = comp::wall_clock::record();
    auto     _tid             = threading::get_id();
    uint64_t _crit_cid        = 0;
    uint64_t _parent_crit_cid = 0;
    uint32_t _depth           = 0;
    auto     _roct_cid        = data->correlation_id;

    auto& _device_id = get_current_device();

    if(data->phase == ACTIVITY_API_PHASE_ENTER)
    {
        if(cid == HIP_API_ID_hipSetDevice)
            get_current_device() =
                reinterpret_cast<int>(data->args.hipSetDevice.deviceId) + 1;

        const char* _name = nullptr;
        switch(cid)
        {
            case HIP_API_ID_hipLaunchKernel:
            {
                _name = hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address,
                                              data->args.hipLaunchKernel.stream);
                break;
            }
            case HIP_API_ID_hipLaunchCooperativeKernel:
            {
                _name =
                    hipKernelNameRefByPtr(data->args.hipLaunchCooperativeKernel.f,
                                          data->args.hipLaunchCooperativeKernel.stream);
                if(!_name)
                {
                    _name =
                        hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address,
                                              data->args.hipLaunchKernel.stream);
                }
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
                locking::atomic_lock _lk{ roctracer_type_mutex<key_data_mutex_t>() };
                get_roctracer_key_data().emplace(_roct_cid, _name);
                get_roctracer_tid_data().emplace(_roct_cid, _tid);
            }
        }

        std::tie(_crit_cid, _parent_crit_cid, _depth) = create_cpu_cid_entry();

        if(get_use_perfetto())
        {
            static auto _compact_annotations =
                config::get_setting_value<bool>(
                    "ROCPROFSYS_PERFETTO_COMPACT_ROCTRACER_ANNOTATIONS")
                    .value_or(false);

            static auto _enable_backtraces =
                config::get_setting_value<bool>("ROCPROFSYS_ROCTRACER_HIP_API_BACKTRACE")
                    .value_or(false);

            constexpr size_t bt_stack_depth       = 16;
            constexpr size_t bt_ignore_depth      = 3;
            constexpr bool   bt_with_signal_frame = true;

            using backtrace_entry_vec_t = std::vector<tim::unwind::processed_entry>;
            auto _bt_data               = std::optional<backtrace_entry_vec_t>{};
            if(_enable_backtraces && config::get_perfetto_annotations())
            {
                auto _backtrace = tim::get_unw_stack<bt_stack_depth, bt_ignore_depth,
                                                     bt_with_signal_frame>();
                _bt_data        = backtrace_entry_vec_t{};
                _bt_data->reserve(_backtrace.size());
                for(auto itr : _backtrace)
                {
                    if(itr)
                    {
                        if(auto _val = binary::lookup_ipaddr_entry<false>(itr->address());
                           _val)
                        {
                            _bt_data->emplace_back(std::move(*_val));
                        }
                    }
                }
            }

            auto _api_id = static_cast<hip_api_id_t>(cid);
            tracing::push_perfetto_ts(
                category::rocm_hip{}, op_name, _ts,
                ::perfetto::Flow::ProcessScoped(_roct_cid),
                [&](::perfetto::EventContext ctx) {
                    if(config::get_perfetto_annotations())
                    {
                        tracing::add_perfetto_annotation(ctx, "begin_ns", _ts);
                        tracing::add_perfetto_annotation(ctx, "cid", _crit_cid);
                        tracing::add_perfetto_annotation(ctx, "pcid", _parent_crit_cid);
                        tracing::add_perfetto_annotation(ctx, "device", _device_id);
                        tracing::add_perfetto_annotation(ctx, "tid", _tid);
                        tracing::add_perfetto_annotation(ctx, "depth", _depth);
                        tracing::add_perfetto_annotation(ctx, "corr_id", _roct_cid);
                        if(_compact_annotations)
                        {
                            tracing::add_perfetto_annotation(
                                ctx, "args", hip_api_string(_api_id, data));
                        }
                        else
                        {
                            auto _args = std::string{ hip_api_string(_api_id, data) };
                            if(!_args.empty())
                            {
                                for(auto itr : tim::delimit(_args, ","))
                                {
                                    if(itr.empty()) continue;
                                    auto _bpos = itr.find_first_not_of(' ');
                                    auto _epos = itr.find_last_not_of(' ');
                                    if(_epos > _bpos)
                                        itr = itr.substr(_bpos, (_epos - _bpos) + 1);
                                    auto _pos = itr.find('=');
                                    if(_pos != std::string::npos)
                                        tracing::add_perfetto_annotation(
                                            ctx, itr.substr(0, _pos),
                                            itr.substr(_pos + 1));
                                }
                            }
                        }

                        if(_enable_backtraces && _bt_data && !_bt_data->empty())
                        {
                            const std::string _unk    = "??";
                            size_t            _bt_cnt = 0;
                            for(const auto& itr : *_bt_data)
                            {
                                const auto* _func =
                                    (itr.name.empty()) ? &_unk : &itr.name;
                                const auto* _loc =
                                    (itr.location.empty()) ? &_unk : &itr.location;
                                auto _line  = (itr.lineno == 0) ? std::string{ "?" }
                                                                : join("", itr.lineno);
                                auto _entry = join("", demangle(*_func), " @ ",
                                                   join(':', *_loc, _line));
                                if(_bt_cnt < 10)
                                {
                                    // Prepend zero for better ordering in UI.
                                    // Only one zero is ever necessary since stack depth
                                    // is limited to 16.
                                    tracing::add_perfetto_annotation(
                                        ctx, join("", "frame#0", _bt_cnt++), _entry);
                                }
                                else
                                {
                                    tracing::add_perfetto_annotation(
                                        ctx, join("", "frame#", _bt_cnt++), _entry);
                                }
                            }
                        }
                    }
                });
        }
        if(get_use_timemory())
        {
            auto itr = get_roctracer_hip_data()->emplace(
                _roct_cid, roctracer_hip_bundle_t{ op_name });
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

        hip_exec_activity_callbacks(_tid);
    }
    else if(data->phase == ACTIVITY_API_PHASE_EXIT)
    {
        hip_exec_activity_callbacks(_tid);

        if(get_use_perfetto())
        {
            tracing::pop_perfetto_ts(
                category::rocm_hip{}, op_name, _ts, [&](::perfetto::EventContext ctx) {
                    if(config::get_perfetto_annotations())
                    {
                        tracing::add_perfetto_annotation(ctx, "end_ns", _ts);
                    }
                });
        }
        if(get_use_timemory())
        {
            auto _stop = [&_roct_cid](int64_t _tid_v) {
                auto& _data = get_roctracer_hip_data(_tid_v);
                auto  itr   = _data->find(_roct_cid);
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
                for(size_t i = 0; i < thread_info::get_peak_num_threads(); ++i)
                {
                    if(_stop(i)) break;
                }
            }
        }
    }
    tim::consume_parameters(arg);
}

// Activity tracing callback
void
hip_activity_callback(const char* begin, const char* end, void* arg)
{
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    ROCPROFSYS_SCOPED_THREAD_STATE(ThreadState::Internal);

    auto&& _protect = comp::roctracer::protect_flush_activity();
    (void) _protect;

    if(!trait::runtime_enabled<comp::roctracer>::get()) return;
    static auto _kernel_names = std::unordered_map<const char*, std::string>{};
    static auto _indexes      = std::unordered_map<uint64_t, int>{};
    static auto _skip_barrier_packets =
        config::get_setting_value<bool>("ROCPROFSYS_ROCTRACER_DISCARD_BARRIERS")
            .value_or(false);
    const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
    const roctracer_record_t* end_record =
        reinterpret_cast<const roctracer_record_t*>(end);

    auto&& _advance_record = [&record]() {
        ROCPROFSYS_ROCTRACER_CALL(roctracer_next_record(record, &record));
    };

    while(record < end_record)
    {
        // make sure every iteration advances regardless of where return point happens
        scope::destructor _next_dtor{ _advance_record };

        // ROCPROFSYS_CI will enable these asserts and should fail if something relevant
        // changes
        assert(HIP_OP_ID_DISPATCH == 0);
        assert(HIP_OP_ID_COPY == 1);
        assert(HIP_OP_ID_BARRIER == 2);

        if(record->domain == ACTIVITY_DOMAIN_HSA_OPS)
        {
            hsa_activity_callback(record->op, record, arg);
            continue;
        }
        if(record->domain != ACTIVITY_DOMAIN_HIP_OPS) continue;
        if(record->op > HIP_OP_ID_BARRIER) continue;
        if(_skip_barrier_packets && record->op == HIP_OP_ID_BARRIER) continue;

        const char* op_name =
            roctracer_op_string(record->domain, record->op, record->kind);
        auto     _ns_skew  = get_clock_skew();
        uint64_t _beg_ns   = record->begin_ns + _ns_skew;
        uint64_t _end_ns   = record->end_ns + _ns_skew;
        auto     _roct_cid = record->correlation_id;

        auto& _keys = get_roctracer_key_data();
        auto& _tids = get_roctracer_tid_data();

        int64_t     _tid   = 0;                  // thread id
        int32_t     _devid = record->device_id;  // device id
        int64_t     _queid = record->queue_id;   // queue id
        uintptr_t   _queue = 0;                  // Host queue (stream)
        const char* _name  = nullptr;
        bool        _found = false;

        {
            locking::atomic_lock _lk{ roctracer_type_mutex<key_data_mutex_t>() };
            if(_tids.find(_roct_cid) != _tids.end())
            {
                _found   = true;
                _tid     = _tids.at(_roct_cid);
                auto itr = _keys.find(_roct_cid);
                if(itr != _keys.end()) _name = itr->second;
            }
        }

        if(_name == nullptr && op_name == nullptr) continue;
        if(_name == nullptr) _name = op_name;

        static auto _op_id_names =
            std::array<const char*, 3>{ "DISPATCH", "COPY", "BARRIER" };

        if(_end_ns < _beg_ns)
        {
            auto          _verbose = []() { return get_verbose() >= 0 || get_debug(); };
            static size_t _n       = 0;
            static size_t _nmax =
                get_env<size_t>("ROCPROFSYS_ROCTRACER_DISCARD_INVALID", 0);
            if(_nmax == 0) std::swap(_end_ns, _beg_ns);
            ROCPROFSYS_WARNING_IF_F(
                _n < _nmax && _verbose(),
                "%4zu :: Discarding kernel roctracer activity record which ended before "
                "it started :: %-20s :: %-20s :: cid=%lu, time_ns=(%12lu:%12lu) "
                "delta=%li, device=%d, queue=%lu, pid=%u, tid=%lu, op=%s\n",
                _n, op_name, _name, record->correlation_id, _beg_ns, _end_ns,
                (static_cast<int64_t>(_end_ns) - static_cast<int64_t>(_beg_ns)), _devid,
                _queid, record->process_id, _tid, _op_id_names.at(record->op));
            ROCPROFSYS_WARNING_IF_F(
                _nmax > 0 && _n == _nmax && _verbose(),
                "Suppressing future messages about discarding kernel roctracer activity "
                "record which ended before it started. Set "
                "ROCPROFSYS_ROCTRACER_DISCARD_INVALID=N to increase/decrease the number "
                "of messages. If N is set to 0, data will be included after swapping the "
                "begin and end values\n");
            if(_end_ns < _beg_ns)
            {
                ++_n;
                continue;
            }
        }

        // execute this on this thread bc of how perfetto visualization works
        if(get_use_perfetto())
        {
            if(_kernel_names.find(_name) == _kernel_names.end())
                _kernel_names.emplace(_name, tim::demangle(_name));

            auto _track_desc = [](int32_t _device_id, int64_t _queue_id) {
                if(config::get_perfetto_roctracer_per_stream())
                    return JOIN("", "HIP Activity Device ", _device_id, ", Queue ",
                                _queue_id);
                return JOIN("", "HIP Activity Device ", _device_id);
            };

            const auto _track = tracing::get_perfetto_track(
                category::device_hip{}, _track_desc, _devid,
                (get_perfetto_roctracer_per_stream()) ? _queid : 0);

            assert(_end_ns >= _beg_ns);
            tracing::push_perfetto_track(
                category::device_hip{}, _kernel_names.at(_name).c_str(), _track, _beg_ns,
                ::perfetto::Flow::ProcessScoped(_roct_cid),
                [&](::perfetto::EventContext ctx) {
                    if(config::get_perfetto_annotations())
                    {
                        tracing::add_perfetto_annotation(ctx, "begin_ns", _beg_ns);
                        tracing::add_perfetto_annotation(ctx, "end_ns", _end_ns);
                        tracing::add_perfetto_annotation(ctx, "corr_id", _roct_cid);
                        tracing::add_perfetto_annotation(ctx, "device", _devid);
                        tracing::add_perfetto_annotation(ctx, "queue", _queid);
                        tracing::add_perfetto_annotation(ctx, "tid", _tid);
                        tracing::add_perfetto_annotation(
                            ctx, "stream", JOIN("", "0x", std::hex, _queue));
                        tracing::add_perfetto_annotation(ctx, "op",
                                                         _op_id_names.at(record->op));
                    }
                });
            tracing::pop_perfetto_track(category::device_hip{}, "", _track, _end_ns);
        }

        if(_found && _name != nullptr && get_use_timemory())
        {
            auto _func = [_beg_ns, _end_ns, _name]() {
                roctracer_hip_bundle_t _bundle{ _name };
                _bundle.start()
                    .store(std::plus<double>{}, static_cast<double>(_end_ns - _beg_ns))
                    .stop()
                    .get<comp::wall_clock>([&](comp::wall_clock* wc) {
                        wc->set_value(_end_ns - _beg_ns);
                        wc->set_accum(_end_ns - _beg_ns);
                        return wc;
                    });
                _bundle.pop();
            };

            auto&                _async_ops = get_hip_activity_callbacks(_tid);
            locking::atomic_lock _lk{ get_hip_activity_mutex(_tid) };
            _async_ops->emplace_back(std::move(_func));
        }
    }

    // ensures that all the updates are written
    if(get_use_perfetto()) ::perfetto::TrackEvent::Flush();
}

bool&
roctracer_is_init()
{
    static bool _v = tim::get_env("ROCPROFSYS_ROCTRACER_IS_INIT", false);
    return _v;
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
}  // namespace rocprofsys
