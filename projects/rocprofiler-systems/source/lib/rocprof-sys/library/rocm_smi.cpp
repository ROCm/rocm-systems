// Copyright (c) 2018-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/rocm_smi.hpp"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/gpu.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/locking.hpp>

#include <rocm_smi/rocm_smi.h>

#include <cassert>
#include <chrono>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>

#define ROCPROFSYS_ROCM_SMI_CALL(...)                                                    \
    ::rocprofsys::rocm_smi::check_error(__FILE__, __LINE__, __VA_ARGS__)

namespace rocprofsys
{
namespace rocm_smi
{
using bundle_t          = std::deque<data>;
using sampler_instances = thread_data<bundle_t, category::rocm_smi>;

namespace
{
auto&
get_settings(uint32_t _dev_id)
{
    static auto _v = std::unordered_map<uint32_t, rocm_smi::settings>{};
    return _v[_dev_id];
}

bool&
is_initialized()
{
    static bool _v = false;
    return _v;
}

void
check_error(const char* _file, int _line, rsmi_status_t _code, bool* _option = nullptr)
{
    if(_code == RSMI_STATUS_SUCCESS)
        return;
    else if(_code == RSMI_STATUS_NOT_SUPPORTED && _option)
    {
        *_option = false;
        return;
    }

    const char* _msg = nullptr;
    auto        _err = rsmi_status_string(_code, &_msg);
    if(_err != RSMI_STATUS_SUCCESS)
        ROCPROFSYS_THROW("rsmi_status_string failed. No error message available. "
                         "Error code %i originated at %s:%i\n",
                         static_cast<int>(_code), _file, _line);
    ROCPROFSYS_THROW("[%s:%i] Error code %i :: %s", _file, _line, static_cast<int>(_code),
                     _msg);
}

std::atomic<State>&
get_state()
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

    auto _state = get_state().load();

    if(_state != State::Active) return;

    m_dev_id = _dev_id;
    m_ts     = _ts;

#define ROCPROFSYS_RSMI_GET(OPTION, FUNCTION, ...)                                       \
    if(OPTION)                                                                           \
    {                                                                                    \
        try                                                                              \
        {                                                                                \
            ROCPROFSYS_ROCM_SMI_CALL(FUNCTION(__VA_ARGS__), &OPTION);                    \
        } catch(std::runtime_error & _e)                                                 \
        {                                                                                \
            ROCPROFSYS_VERBOSE_F(                                                        \
                0, "[%s] Exception: %s. Disabling future samples from rocm-smi...\n",    \
                #FUNCTION, _e.what());                                                   \
            get_state().store(State::Disabled);                                          \
        }                                                                                \
    }

    ROCPROFSYS_RSMI_GET(get_settings(m_dev_id).busy, rsmi_dev_busy_percent_get, _dev_id,
                        &m_busy_perc);
    ROCPROFSYS_RSMI_GET(get_settings(m_dev_id).temp, rsmi_dev_temp_metric_get, _dev_id,
                        RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_CURRENT, &m_temp);
    ROCPROFSYS_RSMI_GET(get_settings(m_dev_id).power, rsmi_dev_power_ave_get, _dev_id, 0,
                        &m_power);
    ROCPROFSYS_RSMI_GET(get_settings(m_dev_id).mem_usage, rsmi_dev_memory_usage_get,
                        _dev_id, RSMI_MEM_TYPE_VRAM, &m_mem_usage);

#undef ROCPROFSYS_RSMI_GET
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
std::vector<unique_ptr_t<bundle_t>*> _bundle_data{};
}

void
config()
{
    _bundle_data.resize(data::device_count, nullptr);
    for(size_t i = 0; i < data::device_count; ++i)
    {
        if(data::device_list.count(i) > 0)
        {
            _bundle_data.at(i) = &sampler_instances::get()->at(i);
            if(!*_bundle_data.at(i))
                *_bundle_data.at(i) = unique_ptr_t<bundle_t>{ new bundle_t{} };
        }
    }

    data::get_initial().resize(data::device_count);
    for(auto itr : data::device_list)
        data::get_initial().at(itr).sample(itr);
}

void
sample()
{
    for(auto itr : data::device_list)
    {
        if(rocm_smi::get_state() != State::Active) continue;
        ROCPROFSYS_DEBUG_F("Polling rocm-smi for device %u...\n", itr);
        auto& _data = *_bundle_data.at(itr);
        if(!_data) continue;
        _data->emplace_back(data{ itr });
        ROCPROFSYS_DEBUG_F("    %s\n", TIMEMORY_JOIN("", _data->back()).c_str());
    }
}

void
set_state(State _v)
{
    rocm_smi::get_state().store(_v);
}

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
    rocm_smi::set_state(State::PreInit);
    return true;
}

bool
data::shutdown()
{
    ROCPROFSYS_DEBUG("Shutting down rocm-smi...\n");
    rocm_smi::set_state(State::Finalized);
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
    using component::sampling_gpu_busy;
    using component::sampling_gpu_memory;
    using component::sampling_gpu_power;
    using component::sampling_gpu_temp;

    if(device_count < _dev_id) return;

    auto&       _rocm_smi_v = sampler_instances::get()->at(_dev_id);
    auto        _rocm_smi   = (_rocm_smi_v) ? *_rocm_smi_v : std::deque<rocm_smi::data>{};
    const auto& _thread_info = thread_info::get(0, InternalTID);

    ROCPROFSYS_VERBOSE(1, "Post-processing %zu rocm-smi samples from device %u\n",
                       _rocm_smi.size(), _dev_id);

    ROCPROFSYS_CI_THROW(!_thread_info, "Missing thread info for thread 0");
    if(!_thread_info) return;

    auto _settings = get_settings(_dev_id);

    auto _process_perfetto = [&]() {
        auto _idx = std::array<uint64_t, 4>{};
        {
            _idx.fill(_idx.size());
            uint64_t nidx = 0;
            if(_settings.busy) _idx.at(0) = nidx++;
            if(_settings.temp) _idx.at(1) = nidx++;
            if(_settings.power) _idx.at(2) = nidx++;
            if(_settings.mem_usage) _idx.at(3) = nidx++;
        }

        for(auto& itr : _rocm_smi)
        {
            using counter_track = perfetto_counter_track<data>;
            if(itr.m_dev_id != _dev_id) continue;
            if(!counter_track::exists(_dev_id))
            {
                auto addendum = [&](const char* _v) {
                    return JOIN(" ", "GPU", _v, JOIN("", '[', _dev_id, ']'), "(S)");
                };

                if(_settings.busy) counter_track::emplace(_dev_id, addendum("Busy"), "%");
                if(_settings.temp)
                    counter_track::emplace(_dev_id, addendum("Temperature"), "deg C");
                if(_settings.power)
                    counter_track::emplace(_dev_id, addendum("Power"), "watts");
                if(_settings.mem_usage)
                    counter_track::emplace(_dev_id, addendum("Memory Usage"),
                                           "megabytes");
            }
            uint64_t _ts = itr.m_ts;
            if(!_thread_info->is_valid_time(_ts)) continue;

            double _busy  = itr.m_busy_perc;
            double _temp  = itr.m_temp / 1.0e3;
            double _power = itr.m_power / 1.0e6;
            double _usage = itr.m_mem_usage / static_cast<double>(units::megabyte);

            if(_settings.busy)
                TRACE_COUNTER("device_busy", counter_track::at(_dev_id, _idx.at(0)), _ts,
                              _busy);
            if(_settings.temp)
                TRACE_COUNTER("device_temp", counter_track::at(_dev_id, _idx.at(1)), _ts,
                              _temp);
            if(_settings.power)
                TRACE_COUNTER("device_power", counter_track::at(_dev_id, _idx.at(2)), _ts,
                              _power);
            if(_settings.mem_usage)
                TRACE_COUNTER("device_memory_usage",
                              counter_track::at(_dev_id, _idx.at(3)), _ts, _usage);
        }
    };

    if(get_use_perfetto()) _process_perfetto();
}

//--------------------------------------------------------------------------------------//

void
setup()
{
    auto_lock_t _lk{ type_mutex<category::rocm_smi>() };

    if(is_initialized() || !get_use_rocm_smi()) return;

    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    // assign the data value to determined by rocm-smi
    data::device_count = device_count();

    auto _devices_v = get_sampling_gpus();
    for(auto& itr : _devices_v)
        itr = tolower(itr);
    if(_devices_v == "off")
        _devices_v = "none";
    else if(_devices_v == "on")
        _devices_v = "all";
    bool _all_devices = _devices_v.find("all") != std::string::npos || _devices_v.empty();
    bool _no_devices  = _devices_v.find("none") != std::string::npos;

    std::set<uint32_t> _devices = {};
    auto               _emplace = [&_devices](auto idx) {
        if(idx < data::device_count) _devices.emplace(idx);
    };

    if(_all_devices)
    {
        for(uint32_t i = 0; i < data::device_count; ++i)
            _emplace(i);
    }
    else if(!_no_devices)
    {
        auto _enabled = tim::delimit(_devices_v, ",; \t");
        for(auto&& itr : _enabled)
        {
            if(itr.find_first_not_of("0123456789-") != std::string::npos)
            {
                ROCPROFSYS_THROW("Invalid GPU specification: '%s'. Only numerical values "
                                 "(e.g., 0) or ranges (e.g., 0-7) are permitted.",
                                 itr.c_str());
            }

            if(itr.find('-') != std::string::npos)
            {
                auto _v = tim::delimit(itr, "-");
                ROCPROFSYS_CONDITIONAL_THROW(_v.size() != 2,
                                             "Invalid GPU range specification: '%s'. "
                                             "Required format N-M, e.g. 0-4",
                                             itr.c_str());
                for(auto i = std::stoul(_v.at(0)); i < std::stoul(_v.at(1)); ++i)
                    _emplace(i);
            }
            else
            {
                _emplace(std::stoul(itr));
            }
        }
    }

    data::device_list = _devices;

    auto _metrics = get_setting_value<std::string>("ROCPROFSYS_ROCM_SMI_METRICS");

    try
    {
        for(auto itr : _devices)
        {
            uint16_t dev_id = 0;
            ROCPROFSYS_ROCM_SMI_CALL(rsmi_dev_id_get(itr, &dev_id));
            // dev_id holds the device ID of device i, upon a successful call

            if(_metrics && !_metrics->empty())
            {
                using key_pair_t     = std::pair<std::string_view, bool&>;
                const auto supported = std::unordered_map<std::string_view, bool&>{
                    key_pair_t{ "busy", get_settings(dev_id).busy },
                    key_pair_t{ "temp", get_settings(dev_id).temp },
                    key_pair_t{ "power", get_settings(dev_id).power },
                    key_pair_t{ "mem_usage", get_settings(dev_id).mem_usage },
                };

                get_settings(dev_id) = { false, false, false, false };
                for(const auto& metric : tim::delimit(*_metrics, ",;:\t\n "))
                {
                    auto iitr = supported.find(metric);
                    if(iitr == supported.end())
                        ROCPROFSYS_FAIL_F("unsupported rocm-smi metric: %s\n",
                                          metric.c_str());

                    ROCPROFSYS_VERBOSE_F(1, "Enabling rocm-smi metric '%s'\n",
                                         metric.c_str());
                    iitr->second = true;
                }
            }
        }

        is_initialized() = true;

        data::setup();
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE(0, "Exception thrown when initializing rocm-smi: %s\n",
                           _e.what());
        data::device_list = {};
    }
}

void
shutdown()
{
    auto_lock_t _lk{ type_mutex<category::rocm_smi>() };

    if(!is_initialized()) return;

    try
    {
        if(data::shutdown())
        {
            ROCPROFSYS_ROCM_SMI_CALL(rsmi_shut_down());
        }
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE(0, "Exception thrown when shutting down rocm-smi: %s\n",
                           _e.what());
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
    return gpu::rsmi_device_count();
}
}  // namespace rocm_smi
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_temp>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_power>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_memory>), true,
    double)
