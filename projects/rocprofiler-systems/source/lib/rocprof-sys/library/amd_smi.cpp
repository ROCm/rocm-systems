// Copyright (c) 2018-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "library/amd_smi.hpp"
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

#include <cassert>
#include <chrono>
#include <ios>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <thread>

#define ROCPROFSYS_AMD_SMI_CALL(...)                                                     \
    ::rocprofsys::amd_smi::check_error(__FILE__, __LINE__, __VA_ARGS__)

namespace rocprofsys
{
namespace amd_smi
{
using bundle_t          = std::deque<data>;
using sampler_instances = thread_data<bundle_t, category::amd_smi>;

namespace
{
auto&
get_settings(uint32_t _dev_id)
{
    static auto _v = std::unordered_map<uint32_t, amd_smi::settings>{};
    return _v[_dev_id];
}

bool&
is_initialized()
{
    static bool _v = false;
    return _v;
}

amdsmi_version_t&
get_version()
{
    static amdsmi_version_t _v = {};

    if(_v.major == 0 && _v.minor == 0)
    {
        auto _err = amdsmi_get_lib_version(&_v);
        if(_err != AMDSMI_STATUS_SUCCESS)
            ROCPROFSYS_THROW(
                "amdsmi_get_version failed. No version information available.");
    }

    return _v;
}

void
check_error(const char* _file, int _line, amdsmi_status_t _code, bool* _option = nullptr)
{
    if(_code == AMDSMI_STATUS_SUCCESS)
        return;
    else if(_code == AMDSMI_STATUS_NOT_SUPPORTED && _option)
    {
        *_option = false;
        return;
    }

    const char* _msg = nullptr;
    auto        _err = amdsmi_status_code_to_string(_code, &_msg);
    if(_err != AMDSMI_STATUS_SUCCESS)
        ROCPROFSYS_THROW(
            "amdsmi_status_code_to_string failed. No error message available. "
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
    amdsmi_gpu_metrics_t _gpu_metrics;

    auto _state = get_state().load();

    if(_state != State::Active) return;

    m_dev_id = _dev_id;
    m_ts     = _ts;

#define ROCPROFSYS_AMDSMI_GET(OPTION, FUNCTION, ...)                                     \
    if(OPTION)                                                                           \
    {                                                                                    \
        try                                                                              \
        {                                                                                \
            ROCPROFSYS_AMD_SMI_CALL(FUNCTION(__VA_ARGS__), &OPTION);                     \
        } catch(std::runtime_error & _e)                                                 \
        {                                                                                \
            ROCPROFSYS_VERBOSE_F(                                                        \
                0, "[%s] Exception: %s. Disabling future samples from amd-smi...\n",     \
                #FUNCTION, _e.what());                                                   \
            get_state().store(State::Disabled);                                          \
        }                                                                                \
    }

    amdsmi_processor_handle sample_handle = gpu::get_handle_from_id(_dev_id);

    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).busy, amdsmi_get_gpu_activity,
                          sample_handle, &m_busy_perc);
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).temp, amdsmi_get_temp_metric,
                          sample_handle, AMDSMI_TEMPERATURE_TYPE_JUNCTION,
                          AMDSMI_TEMP_CURRENT, &m_temp);
#if(AMDSMI_LIB_VERSION_MAJOR == 2 && AMDSMI_LIB_VERSION_MINOR == 0) ||                   \
    (AMDSMI_LIB_VERSION_MAJOR == 25 && AMDSMI_LIB_VERSION_MINOR == 2)
    // This was a transient change in the AMD SMI API. It was never officially released.
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).power, amdsmi_get_power_info,
                          sample_handle, 0, &m_power)
#else
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).power, amdsmi_get_power_info,
                          sample_handle, &m_power)
#endif
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).mem_usage, amdsmi_get_gpu_memory_usage,
                          sample_handle, AMDSMI_MEM_TYPE_VRAM, &m_mem_usage);
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).vcn_activity,
                          amdsmi_get_gpu_metrics_info, sample_handle, &_gpu_metrics);
    ROCPROFSYS_AMDSMI_GET(get_settings(m_dev_id).jpeg_activity,
                          amdsmi_get_gpu_metrics_info, sample_handle, &_gpu_metrics);

    for(const auto& v_activity : _gpu_metrics.vcn_activity)
    {
        if(v_activity != UINT16_MAX) m_vcn_metrics.push_back(v_activity);
    }
    for(const auto& j_activity : _gpu_metrics.jpeg_activity)
    {
        if(j_activity != UINT16_MAX) m_jpeg_metrics.push_back(j_activity);
    }

#undef ROCPROFSYS_AMDSMI_GET
}

void
data::print(std::ostream& _os) const
{
    std::stringstream _ss{};

#if ROCPROFSYS_USE_ROCM > 0
    _ss << "device: " << m_dev_id << ", gpu busy: = " << m_busy_perc.gfx_activity
        << "%, mm busy: = " << m_busy_perc.mm_activity
        << "%, umc busy: = " << m_busy_perc.umc_activity << "%, temp = " << m_temp
        << ", current power = " << m_power.current_socket_power
        << ", memory usage = " << m_mem_usage;
#endif
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
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    for(auto itr : data::device_list)
    {
        if(amd_smi::get_state() != State::Active) continue;
        ROCPROFSYS_DEBUG_F("Polling amd-smi for device %u...\n", itr);
        auto& _data = *_bundle_data.at(itr);
        if(!_data) continue;
        _data->emplace_back(data{ itr });
        ROCPROFSYS_DEBUG_F("    %s\n", TIMEMORY_JOIN("", _data->back()).c_str());
    }
}

void
set_state(State _v)
{
    amd_smi::get_state().store(_v);
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
    amd_smi::set_state(State::PreInit);
    return true;
}

bool
data::shutdown()
{
    amd_smi::set_state(State::Finalized);
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
    using component::sampling_gpu_busy_gfx;
    using component::sampling_gpu_busy_mm;
    using component::sampling_gpu_busy_umc;
    using component::sampling_gpu_jpeg;
    using component::sampling_gpu_memory;
    using component::sampling_gpu_power;
    using component::sampling_gpu_temp;
    using component::sampling_gpu_vcn;

    if(device_count < _dev_id) return;

    auto&       _amd_smi_v   = sampler_instances::get()->at(_dev_id);
    auto        _amd_smi     = (_amd_smi_v) ? *_amd_smi_v : std::deque<amd_smi::data>{};
    const auto& _thread_info = thread_info::get(0, InternalTID);

    ROCPROFSYS_VERBOSE(1, "Post-processing %zu amd-smi samples from device %u\n",
                       _amd_smi.size(), _dev_id);

    ROCPROFSYS_CI_THROW(!_thread_info, "Missing thread info for thread 0");
    if(!_thread_info) return;

    auto _settings = get_settings(_dev_id);

    auto _process_perfetto = [&]() {
        constexpr uint8_t AMD_SMI_METRICS_COUNT = 8;
        auto              _idx = std::array<uint64_t, AMD_SMI_METRICS_COUNT>{};

        {
            _idx.fill(_idx.size());
            uint64_t nidx = 0;
            if(_settings.busy)
            {
                _idx.at(0) = nidx++;  // GFX Busy
                _idx.at(1) = nidx++;  // UMC Busy
                _idx.at(2) = nidx++;  // MM Busy
            }
            if(_settings.temp) _idx.at(3) = nidx++;
            if(_settings.power) _idx.at(4) = nidx++;
            if(_settings.mem_usage) _idx.at(5) = nidx++;
            if(_settings.vcn_activity) _idx.at(6) = nidx++;
            if(_settings.jpeg_activity) _idx.at(7) = nidx++;
        }

        for(auto& itr : _amd_smi)
        {
            using counter_track = perfetto_counter_track<data>;
            if(itr.m_dev_id != _dev_id) continue;
            if(!counter_track::exists(_dev_id))
            {
                auto addendum = [&](const char* _v) {
                    return JOIN(" ", "GPU", _v, JOIN("", '[', _dev_id, ']'), "(S)");
                };
                auto addendum_blk = [&](std::size_t _i, const char* _metric) {
                    if(_i < 10)
                    {
                        return JOIN(" ", "GPU", JOIN("", '[', _dev_id, ']'), _metric,
                                    JOIN("", "[0", _i, ']'), "(S)");
                    }
                    else
                    {
                        return JOIN(" ", "GPU", JOIN("", '[', _dev_id, ']'), _metric,
                                    JOIN("", '[', _i, ']'), "(S)");
                    }
                };

                if(_settings.busy)
                {
                    counter_track::emplace(_dev_id, addendum("GFX Busy"), "%");
                    counter_track::emplace(_dev_id, addendum("UMC Busy"), "%");
                    counter_track::emplace(_dev_id, addendum("MM Busy"), "%");
                }
                if(_settings.temp)
                    counter_track::emplace(_dev_id, addendum("Temperature"), "deg C");
                if(_settings.power)
                    counter_track::emplace(_dev_id, addendum("Current Power"), "watts");
                if(_settings.mem_usage)
                    counter_track::emplace(_dev_id, addendum("Memory Usage"),
                                           "megabytes");
                if(_settings.vcn_activity)
                {
                    for(std::size_t i = 0; i < std::size(itr.m_vcn_metrics); ++i)
                        counter_track::emplace(_dev_id, addendum_blk(i, "  VCN Activity"),
                                               "%");
                }
                if(_settings.jpeg_activity)
                {
                    for(std::size_t i = 0; i < std::size(itr.m_jpeg_metrics); ++i)
                        counter_track::emplace(_dev_id, addendum_blk(i, "JPEG Activity"),
                                               "%");
                }
            }
            uint64_t _ts = itr.m_ts;
            if(!_thread_info->is_valid_time(_ts)) continue;

            double _gfxbusy = itr.m_busy_perc.gfx_activity;
            double _umcbusy = itr.m_busy_perc.umc_activity;
            double _mmbusy  = itr.m_busy_perc.mm_activity;
            double _temp    = itr.m_temp;
            double _power   = itr.m_power.current_socket_power;
            double _usage   = itr.m_mem_usage / static_cast<double>(units::megabyte);

            if(_settings.busy)
            {
                TRACE_COUNTER("device_busy_gfx", counter_track::at(_dev_id, _idx.at(0)),
                              _ts, _gfxbusy);
                TRACE_COUNTER("device_busy_umc", counter_track::at(_dev_id, _idx.at(1)),
                              _ts, _umcbusy);
                TRACE_COUNTER("device_busy_mm", counter_track::at(_dev_id, _idx.at(2)),
                              _ts, _mmbusy);
            }
            if(_settings.temp)
                TRACE_COUNTER("device_temp", counter_track::at(_dev_id, _idx.at(3)), _ts,
                              _temp);
            if(_settings.power)
                TRACE_COUNTER("device_power", counter_track::at(_dev_id, _idx.at(4)), _ts,
                              _power);
            if(_settings.mem_usage)
                TRACE_COUNTER("device_memory_usage",
                              counter_track::at(_dev_id, _idx.at(5)), _ts, _usage);
            if(_settings.vcn_activity)
            {
                uint64_t idx = _idx.at(6);
                for(const auto& temp : itr.m_vcn_metrics)
                {
                    TRACE_COUNTER("device_vcn_activity", counter_track::at(_dev_id, idx),
                                  _ts, temp);
                    ++idx;
                }
            }
            if(_settings.jpeg_activity)
            {
                uint64_t idx = _idx.at(7);
                if(_settings.vcn_activity) idx += (itr.m_vcn_metrics.size() - 1);
                for(const auto& temp : itr.m_jpeg_metrics)
                {
                    TRACE_COUNTER("device_jpeg_activity", counter_track::at(_dev_id, idx),
                                  _ts, temp);
                    ++idx;
                }
            }
        }
    };

    if(get_use_perfetto()) _process_perfetto();
}

//--------------------------------------------------------------------------------------//

void
setup()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(is_initialized() || !get_use_amd_smi()) return;

    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    if(!gpu::initialize_amdsmi())
    {
        ROCPROFSYS_WARNING_F(0,
                             "AMD SMI is not available. Disabling AMD SMI sampling...");
        return;
    }

    amdsmi_version_t _version = get_version();
    ROCPROFSYS_VERBOSE_F(0, "AMD SMI version: %u.%u.%u - str: %s.\n", _version.major,
                         _version.minor, _version.release, _version.build);

    data::device_count = gpu::get_processor_count();

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

    auto _metrics = get_setting_value<std::string>("ROCPROFSYS_AMD_SMI_METRICS");

    try
    {
        for(auto itr : _devices)
        {
            // Enable selected metrics only
            if((_metrics && !_metrics->empty()) && (*_metrics != "all"))
            {
                using key_pair_t     = std::pair<std::string_view, bool&>;
                const auto supported = std::unordered_map<std::string_view, bool&>{
                    key_pair_t{ "busy", get_settings(itr).busy },
                    key_pair_t{ "temp", get_settings(itr).temp },
                    key_pair_t{ "power", get_settings(itr).power },
                    key_pair_t{ "mem_usage", get_settings(itr).mem_usage },
                    key_pair_t{ "vcn_activity", get_settings(itr).vcn_activity },
                    key_pair_t{ "jpeg_activity", get_settings(itr).jpeg_activity },
                };

                // Initialize all metrics to false
                for(auto& it : supported)
                    it.second = false;

                // Parse list of metrics enabled by the user
                if(*_metrics != "none")
                {
                    for(const auto& metric : tim::delimit(*_metrics, ",;:\t\n "))
                    {
                        auto iitr = supported.find(metric);
                        if(iitr == supported.end())
                            ROCPROFSYS_FAIL_F("unsupported amd-smi metric: %s\n",
                                              metric.c_str());
                        ROCPROFSYS_VERBOSE_F(
                            1, "Enabling amd-smi metric '%s' on device [%u]\n",
                            metric.c_str(), itr);
                        iitr->second = true;
                    }
                }
            }
        }

        is_initialized() = true;

        data::setup();
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE(0, "Exception thrown when initializing amd-smi: %s\n",
                           _e.what());
        data::device_list = {};
    }
}

void
shutdown()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(!is_initialized()) return;
    ROCPROFSYS_VERBOSE_F(1, "Shutting down amd-smi...\n");

    try
    {
        if(data::shutdown())
        {
            ROCPROFSYS_AMD_SMI_CALL(amdsmi_shut_down());
        }
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE(0, "Exception thrown when shutting down amd-smi: %s\n",
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
    return gpu::device_count();
}
}  // namespace amd_smi
}  // namespace rocprofsys

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_gfx>),
    true, double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_umc>),
    true, double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_busy_mm>),
    true, double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_temp>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_power>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_memory>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_vcn>), true,
    double)

ROCPROFSYS_INSTANTIATE_EXTERN_COMPONENT(
    TIMEMORY_ESC(data_tracker<double, rocprofsys::component::backtrace_gpu_jpeg>), true,
    double)
