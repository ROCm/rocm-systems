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

#if defined(NDEBUG)
#    undef NDEBUG
#endif

#include "library/components/rocm_smi.hpp"
#include "library/common.hpp"
#include "library/components/fwd.hpp"
#include "library/config.hpp"
#include "library/critical_trace.hpp"
#include "library/debug.hpp"
#include "library/gpu.hpp"
#include "library/perfetto.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/locking.hpp>

#include <rocm_smi/rocm_smi.h>

#include <cassert>
#include <chrono>
#include <ios>
#include <sstream>
#include <sys/resource.h>
#include <thread>

#define OMNITRACE_ROCM_SMI_CALL(ERROR_CODE) ::omnitrace::rocm_smi::check_error(ERROR_CODE)

namespace omnitrace
{
namespace rocm_smi
{
using tim::type_mutex;
using auto_lock_t       = tim::auto_lock_t;
using bundle_t          = std::deque<data>;
using sampler_instances = thread_data<bundle_t, api::rocm_smi>;

namespace
{
bool&
is_initialized()
{
    static bool _v = false;
    return _v;
}

void
check_error(rsmi_status_t ec)
{
    if(ec == RSMI_STATUS_SUCCESS) return;
    const char* _msg = nullptr;
    auto        _err = rsmi_status_string(ec, &_msg);
    if(_err != RSMI_STATUS_SUCCESS)
        OMNITRACE_THROW(
            "rsmi_status_string(%i, ...) failed. No error message available\n", (int) ec);
    OMNITRACE_THROW("%s", _msg);
}

std::atomic<State>&
get_rocm_smi_state()
{
    static std::atomic<State> _v{ State::PreInit };
    return _v;
}
}  // namespace

//--------------------------------------------------------------------------------------//

size_t                           data::device_count     = 0;
std::set<uint32_t>               data::device_list      = {};
std::unique_ptr<data::promise_t> data::polling_finished = {};

data::data(uint32_t _dev_id) { sample(_dev_id); }

void
data::sample(uint32_t _dev_id)
{
    auto _ts = tim::get_clock_real_now<size_t, std::nano>();
    assert(_ts < std::numeric_limits<int64_t>::max());

    m_dev_id = _dev_id;
    m_ts     = _ts;

    rsmi_dev_busy_percent_get(_dev_id, &m_busy_perc);
    rsmi_dev_temp_metric_get(_dev_id, RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &m_temp);
    rsmi_dev_power_ave_get(_dev_id, 0, &m_power);
    rsmi_dev_memory_usage_get(_dev_id, RSMI_MEM_TYPE_VRAM, &m_mem_usage);
}

void
data::print(std::ostream& _os) const
{
    std::stringstream _ss{};
    _ss << "device: " << m_dev_id << ", busy = " << m_busy_perc << "%, temp = " << m_temp
        << ", power = " << m_power << ", memory usage = " << m_mem_usage;
    _os << _ss.str();
}

namespace
{
std::vector<std::unique_ptr<bundle_t>*> _bundle_data{};
}

void
config()
{
    _bundle_data.resize(data::device_count, nullptr);
    for(size_t i = 0; i < data::device_count; ++i)
    {
        if(data::device_list.count(i) > 0)
        {
            _bundle_data.at(i) = &sampler_instances::instances().at(i);
            if(!*_bundle_data.at(i)) *_bundle_data.at(i) = std::make_unique<bundle_t>();
        }
    }

    data::get_initial().resize(data::device_count);
    for(auto itr : data::device_list)
        data::get_initial().at(itr).sample(itr);
}

void
sample()
{
    if(get_rocm_smi_state() != State::Active) return;

    for(auto itr : data::device_list)
    {
        OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug(),
                                          "Polling rocm-smi for device %u...\n", itr);
        auto& _data = *_bundle_data.at(itr);
        if(!_data) continue;
        _data->emplace_back(data{ itr });
        OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug(), "    %s\n",
                                          TIMEMORY_JOIN("", _data->back()).c_str());
    }
}

void
set_state(State _v)
{
    get_rocm_smi_state().store(_v);
}

/*
void
data::poll(std::atomic<State>* _state, nsec_t _interval, promise_t* _ready)
{
    threading::set_thread_name("omni.rocm_smi");

    // notify thread started
    if(_ready) _ready->set_value();

    std::vector<std::unique_ptr<bundle_t>*> _bundle_data{};
    _bundle_data.resize(device_count, nullptr);
    for(size_t i = 0; i < device_count; ++i)
    {
        if(device_list.count(i) > 0)
        {
            _bundle_data.at(i) = &sampler_instances::instances().at(i);
            if(!*_bundle_data.at(i)) *_bundle_data.at(i) = std::make_unique<bundle_t>();
        }
    }

    OMNITRACE_CONDITIONAL_BASIC_PRINT(
        get_verbose() > 0 || get_debug(),
        "Polling rocm-smi for %zu device(s) at an interval of %f seconds...\n",
        device_list.size(),
        std::chrono::duration_cast<std::chrono::duration<double>>(_interval).count());

    get_initial().resize(device_count);
    for(auto itr : device_list)
        get_initial().at(itr).sample(itr);

    auto _now = std::chrono::steady_clock::now();
    while(_state && _state->load() != State::Finalized && get_state() != State::Finalized)
    {
        std::this_thread::sleep_until(_now);
        if(_state->load() != State::Active) continue;
        for(auto itr : device_list)
        {
            OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug(),
                                              "Polling rocm-smi for device %u...\n", itr);
            auto& _data = *_bundle_data.at(itr);
            if(!_data) continue;
            _data->emplace_back(data{ itr });
            OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug(), "    %s\n",
                                              TIMEMORY_JOIN("", _data->back()).c_str());
        }
        while(_now < std::chrono::steady_clock::now())
            _now += _interval;
    }
    OMNITRACE_CONDITIONAL_BASIC_PRINT(get_debug(), "Polling rocm-smi completed...\n");

    if(polling_finished) polling_finished->set_value();
}
*/

std::vector<data>&
data::get_initial()
{
    static std::vector<data> _v{};
    return _v;
}

bool
data::setup()
{
    perfetto_counter_track<data>::init();
    set_state(State::PreInit);
    return true;
}

bool
data::shutdown()
{
    OMNITRACE_DEBUG("Shutting down rocm-smi...\n");
    set_state(State::Finalized);
    return true;
}

#define GPU_METRIC(COMPONENT, ...)                                                       \
    if constexpr(tim::trait::is_available<COMPONENT>::value)                             \
    {                                                                                    \
        auto* _val = _v.get<COMPONENT>();                                                \
        if(_val)                                                                         \
        {                                                                                \
            _val->set_value(itr.__VA_ARGS__);                                            \
            _val->set_accum(itr.__VA_ARGS__);                                            \
        }                                                                                \
    }

void
data::post_process(uint32_t _dev_id)
{
    OMNITRACE_CONDITIONAL_PRINT(get_debug() || get_verbose() > 0,
                                "Post-processing rocm-smi data for device %u\n", _dev_id);

    using component::sampling_gpu_busy;
    using component::sampling_gpu_memory;
    using component::sampling_gpu_power;
    using component::sampling_gpu_temp;
    using bundle_t = tim::lightweight_tuple<sampling_gpu_busy, sampling_gpu_temp,
                                            sampling_gpu_power, sampling_gpu_memory>;

    if(device_count < _dev_id) return;

    auto& _rocm_smi_v = sampler_instances::instances().at(_dev_id);
    auto  _rocm_smi   = (_rocm_smi_v) ? *_rocm_smi_v : std::deque<rocm_smi::data>{};

    auto _process_perfetto = [&]() {
        for(auto& itr : _rocm_smi)
        {
            using counter_track = perfetto_counter_track<data>;
            if(itr.m_dev_id != _dev_id) continue;
            if(!counter_track::exists(_dev_id))
            {
                auto _devname = TIMEMORY_JOIN("", "[GPU ", _dev_id, "] ");
                auto addendum = [&](const char* _v) {
                    return _devname + std::string{ _v };
                };
                counter_track::emplace(_dev_id, addendum("Busy"), "%");
                counter_track::emplace(_dev_id, addendum("Temperature"), "deg C");
                counter_track::emplace(_dev_id, addendum("Power"), "watts");
                counter_track::emplace(_dev_id, addendum("Memory Usage"), "megabytes");
            }
            uint64_t _ts    = itr.m_ts;
            double   _busy  = itr.m_busy_perc;
            double   _temp  = itr.m_temp / 1.0e3;
            double   _power = itr.m_power / 1.0e6;
            double   _usage = itr.m_mem_usage / static_cast<double>(units::megabyte);
            TRACE_COUNTER("rocm_smi", counter_track::at(_dev_id, 0), _ts, _busy);
            TRACE_COUNTER("rocm_smi", counter_track::at(_dev_id, 1), _ts, _temp);
            TRACE_COUNTER("rocm_smi", counter_track::at(_dev_id, 2), _ts, _power);
            TRACE_COUNTER("rocm_smi", counter_track::at(_dev_id, 3), _ts, _usage);
        }
    };

    if(get_use_perfetto()) _process_perfetto();

    if(!get_use_timemory()) return;

    for(auto& itr : _rocm_smi)
    {
        using entry_t = critical_trace::entry;
        auto _ts      = itr.m_ts;
        auto _entries = critical_trace::get_entries(_ts, [](const entry_t& _e) {
            return _e.device == critical_trace::Device::GPU;
        });

        std::vector<bundle_t> _tc{};
        _tc.reserve(_entries.size());
        for(auto& eitr : _entries)
        {
            auto& _v = _tc.emplace_back(eitr.first);
            _v.push();
            _v.start();
            _v.stop();

            GPU_METRIC(sampling_gpu_busy, m_busy_perc)
            GPU_METRIC(sampling_gpu_temp, m_temp / 1.0e3)  // provided in milli-degree C
            GPU_METRIC(sampling_gpu_power,
                       m_power * units::microwatt / static_cast<double>(units::watt))
            GPU_METRIC(sampling_gpu_memory,
                       m_mem_usage / static_cast<double>(units::megabyte))

            _v.pop();
        }
    }
}

//--------------------------------------------------------------------------------------//

void
setup()
{
    auto_lock_t _lk{ type_mutex<api::rocm_smi>() };

    if(is_initialized() || !get_use_rocm_smi()) return;

    auto _enable_samp = pthread_gotcha::enable_sampling_on_child_threads();
    pthread_gotcha::enable_sampling_on_child_threads() = false;

    // assign the data value to determined by rocm-smi
    data::device_count = device_count();

    auto _devices_v = get_rocm_smi_devices();
    for(auto& itr : _devices_v)
        itr = tolower(itr);
    bool _all_devices = _devices_v.find("all") != std::string::npos || _devices_v.empty();
    bool _no_devices  = _devices_v.find("none") != std::string::npos;

    std::set<uint32_t> _devices{};
    if(_all_devices)
    {
        for(uint32_t i = 0; i < data::device_count; ++i)
            _devices.emplace(i);
    }
    else if(!_no_devices)
    {
        for(auto&& itr : tim::delimit(get_rocm_smi_devices()))
        {
            uint32_t idx = std::stoul(itr);
            if(idx < data::device_count) _devices.emplace(idx);
        }
    }

    data::device_list = _devices;

    for(auto itr : _devices)
    {
        uint16_t dev_id = 0;
        OMNITRACE_ROCM_SMI_CALL(rsmi_dev_id_get(itr, &dev_id));
        // dev_id holds the device ID of device i, upon a successful call
    }

    is_initialized() = true;

    data::setup();

    pthread_gotcha::enable_sampling_on_child_threads() = _enable_samp;
}

void
shutdown()
{
    auto_lock_t _lk{ type_mutex<api::rocm_smi>() };

    if(!is_initialized()) return;

    if(data::shutdown())
    {
        OMNITRACE_ROCM_SMI_CALL(rsmi_shut_down());
    }

    is_initialized() = false;
}

void
post_process()
{
    for(auto itr : data::device_list)
        data::post_process(itr);
}

uint32_t
device_count()
{
    uint32_t _num_devices = 0;
    try
    {
        static auto _rsmi_init_once = []() { OMNITRACE_ROCM_SMI_CALL(rsmi_init(0)); };
        static std::once_flag _once{};
        std::call_once(_once, _rsmi_init_once);

        OMNITRACE_ROCM_SMI_CALL(rsmi_num_monitor_devices(&_num_devices));
    } catch(const std::exception& _e)
    {
        OMNITRACE_BASIC_PRINT("Exception: %s\n", _e.what());
    }
    return _num_devices;
}
}  // namespace rocm_smi
}  // namespace omnitrace

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_gpu_busy>), true,
    double)

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_gpu_temp>), true,
    double)

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_gpu_power>), true,
    double)

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, omnitrace::component::backtrace_gpu_memory>), true,
    double)
