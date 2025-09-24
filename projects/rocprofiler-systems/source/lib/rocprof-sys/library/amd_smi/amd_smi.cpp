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

#include "core/agent.hpp"
#include "core/benchmark/benchmark.hpp"
#include "core/benchmark/category.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cache_utility.hpp"
#include "core/trace_cache/sample_type.hpp"
#include "core/utility.hpp"
#include "library/amd_smi/processor.hpp"
#include <algorithm>
#include <amd_smi/amdsmi.h>
#include <cstdint>
#include <exception>
#include <ios>
#include <memory>
#include <ostream>
#include <regex>
#include <vector>
#if defined(NDEBUG)
#    undef NDEBUG
#endif

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/gpu.hpp"
#include "core/node_info.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/amd_smi/amd_smi.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/locking.hpp>

#include <cassert>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>

#include "amd_smi_driver.hpp"
#include "service.hpp"

namespace rocprofsys
{
namespace amd_smi
{
namespace
{

std::shared_ptr<service<amd_smi_driver_factory>> m_smi_service;
using processor_vector_t = service<amd_smi_driver_factory>::processor_vector_t;
processor_vector_t m_gpu_processors;

union enabled_metrics
{
    struct
    {
        uint16_t current_socket_power : 1;
        uint16_t average_socket_power : 1;
        uint16_t memory_usage         : 1;
        uint16_t hotspot_temperature  : 1;
        uint16_t edge_temperature     : 1;
        uint16_t gfx_activity         : 1;
        uint16_t umc_activity         : 1;
        uint16_t mm_activity          : 1;
        uint16_t vcn_activity         : 1;
        uint16_t jpeg_activity        : 1;
    } fields;
    uint16_t value;
};

constexpr uint16_t enable_all_metrics  = 0xffff;
constexpr uint16_t disable_all_metrics = 0x0000;

std::set<size_t>
parse_numeric_range(const std::string& input_range)
{
    std::set<size_t> result{};

    auto get_range_values = [](auto& token, const auto& range_delimiter_position) {
        size_t begin = std::stoi(std::string{ token.begin(), range_delimiter_position });
        size_t end = std::stoi(std::string{ range_delimiter_position + 1, token.end() });

        if(begin > end)
        {
            std::swap(begin, end);
        }

        return std::pair<size_t, size_t>{ begin, end };
    };

    // validate input string
    const std::regex validator{ R"(^\d+(?:-\d+)?(?:[;,]\d+(?:[-:]\d+)?)*$)" };

    if(!std::regex_match(input_range, validator))
    {
        ROCPROFSYS_VERBOSE(0, "Failed to parse gpu input list: %s\n",
                           input_range.c_str());
        return result;
    }

    std::regex           tokenizer{ R"(\d+(?:[-:]\d+)*)" };
    std::sregex_iterator it(input_range.begin(), input_range.end(), tokenizer);
    std::sregex_iterator end;

    for(; it != end; ++it)
    {
        auto token = it->str();
        auto delimiter_position =
            std::find_if(token.begin(), token.end(),
                         [](const auto& c) { return c == ':' || c == '-'; });

        if(delimiter_position != token.end())
        {
            auto [begining, end] = get_range_values(token, delimiter_position);
            for(auto i = begining; i <= end; ++i)
            {
                result.insert(i);
            }
        }
        else
        {
            size_t value = std::stoi(token);
            result.insert(value);
        }
    }

    return result;
}

processor_vector_t
filter_devices(const processor_vector_t& processors, const std::string& filter)
{
    if(filter == "all" || filter == "on")
    {
        return processors;
    }

    if(filter == "none" || filter == "off")
    {
        return {};
    }

    auto enabled_devices = parse_numeric_range(filter);

    processor_vector_t filtered_processors;
    for(const auto& processor : processors)
    {
        if(enabled_devices.count(processor->get_index()) > 0)
        {
            filtered_processors.emplace_back(processor);
        }
    }

    return filtered_processors;
}

enabled_metrics
get_enabled_metrics()
{
    auto _settings = get_setting_value<std::string>("ROCPROFSYS_AMD_SMI_METRICS");
    if(*_settings == "none")
    {
        enabled_metrics _metrics;
        _metrics.value = disable_all_metrics;
        return _metrics;
    }

    if(*_settings == "all")
    {
        enabled_metrics _metrics;
        _metrics.value = enable_all_metrics;
        return _metrics;
    }

    enabled_metrics _metrics;
    return _metrics;
};

void
metadata_initialize_category()
{
    trace_cache::get_metadata_registry().add_string(
        trait::name<category::amd_smi>::value);
}

void
metadata_initialize_smi_tracks(size_t gpu_id)
{
    const auto thread_id = std::nullopt;

    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_gfx_busy>(gpu_id),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_umc_busy>(gpu_id),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_mm_busy>(gpu_id),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_power>(gpu_id),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_temp>(gpu_id),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_memory_usage>(
              gpu_id),
          thread_id, "{}" });

    auto add_vcn_track = [&](std::optional<int> xcp_idx) {
        for(auto clk = 0; clk < AMDSMI_MAX_NUM_VCN; ++clk)
        {
            auto name = trace_cache::info::annotate_with_device_id<
                category::amd_smi_vcn_activity>(gpu_id, xcp_idx, clk);
            trace_cache::get_metadata_registry().add_track(
                { name.c_str(), thread_id, "{}" });
        }
    };

    auto add_jpeg_track = [&](std::optional<int> xcp_idx) {
        for(auto clk = 0; clk < AMDSMI_MAX_NUM_JPEG; ++clk)
        {
            auto name = trace_cache::info::annotate_with_device_id<
                category::amd_smi_jpeg_activity>(gpu_id, xcp_idx, clk);
            trace_cache::get_metadata_registry().add_track(
                { name.c_str(), thread_id, "{}" });
        }
    };

    if(gpu::is_vcn_activity_supported(gpu_id))
    {
        add_vcn_track(std::nullopt);
    }
    else
    {
        for(int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            add_vcn_track(xcp);
        }
    }

    if(gpu::is_jpeg_activity_supported(gpu_id))
    {
        add_jpeg_track(std::nullopt);
    }
    else
    {
        for(auto xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            add_jpeg_track(xcp);
        }
    }
}

void
metadata_initialize_smi_pmc(size_t gpu_id)
{
    // TODO: Find the proper values for a following definitions
    size_t      EVENT_CODE       = 0;
    size_t      INSTANCE_ID      = 0;
    const char* LONG_DESCRIPTION = "";
    const char* COMPONENT        = "";
    const char* BLOCK            = "";
    const char* EXPRESSION       = "";
    const char* CELSIUS_DEGREES  = "\u00B0C";
    auto        ni               = node_info::get_instance();
    const char* TARGET_ARCH      = "GPU";

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_gfx_busy>::value, "GFX Busy",
          trait::name<category::amd_smi_gfx_busy>::description, LONG_DESCRIPTION,
          COMPONENT, trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0, "{}" });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_umc_busy>::value, "UMC Busy",
          trait::name<category::amd_smi_umc_busy>::description, LONG_DESCRIPTION,
          COMPONENT, trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0, "{}" });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_mm_busy>::value, "MM Busy",
          trait::name<category::amd_smi_mm_busy>::description, LONG_DESCRIPTION,
          COMPONENT, trace_cache::PERCENTAGE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0, "{}" });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_temp>::value, "Temp",
          trait::name<category::amd_smi_temp>::description, LONG_DESCRIPTION, COMPONENT,
          CELSIUS_DEGREES, rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_power>::value, "Pow",
          trait::name<category::amd_smi_power>::description, LONG_DESCRIPTION, COMPONENT,
          "W", rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_memory_usage>::value, "MemUsg",
          trait::name<category::amd_smi_memory_usage>::description, LONG_DESCRIPTION,
          COMPONENT, tim::units::mem_repr(tim::units::megabyte),
          rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });

    auto add_vcn_pmc = [&](std::optional<int> xcp_idx) {
        for(int clk = 0; clk < AMDSMI_MAX_NUM_VCN; ++clk)
        {
            std::stringstream name_ss;
            name_ss << trait::name<category::amd_smi_vcn_activity>::value;
            if(xcp_idx) name_ss << "_" << *xcp_idx;
            name_ss << "_" << clk;

            std::stringstream symbol_ss;
            symbol_ss << "VcnAct";
            if(xcp_idx) symbol_ss << "_" << *xcp_idx;
            symbol_ss << "_" << clk;

            trace_cache::get_metadata_registry().add_pmc_info(
                { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                  name_ss.str(), symbol_ss.str(),
                  trait::name<category::amd_smi_vcn_activity>::description,
                  LONG_DESCRIPTION, COMPONENT, trace_cache::PERCENTAGE,
                  rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
        }
    };

    auto add_jpeg_pmc = [&](std::optional<int> xcp_idx) {
        for(auto clk = 0; clk < AMDSMI_MAX_NUM_JPEG; ++clk)
        {
            std::stringstream name_ss;
            name_ss << trait::name<category::amd_smi_jpeg_activity>::value;
            if(xcp_idx) name_ss << "_" << *xcp_idx;
            name_ss << "_" << std::to_string(clk);

            std::stringstream symbol_ss;
            symbol_ss << "JpegAct";
            if(xcp_idx) symbol_ss << "_" << *xcp_idx;
            symbol_ss << "_" << std::to_string(clk);

            trace_cache::get_metadata_registry().add_pmc_info(
                { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
                  name_ss.str(), symbol_ss.str(),
                  trait::name<category::amd_smi_jpeg_activity>::description,
                  LONG_DESCRIPTION, COMPONENT, trace_cache::PERCENTAGE,
                  rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
        }
    };

    if(gpu::is_vcn_activity_supported(gpu_id))
    {
        add_vcn_pmc(std::nullopt);
    }
    else
    {
        for(int xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            add_vcn_pmc(xcp);
        }
    }

    if(gpu::is_jpeg_activity_supported(gpu_id))
    {
        add_jpeg_pmc(std::nullopt);
    }
    else
    {
        for(auto xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
        {
            add_jpeg_pmc(xcp);
        }
    }
}

bool&
is_initialized()
{
    static bool _v = false;
    return _v;
}

std::atomic<State>&
get_state()
{
    static std::atomic<State> _v{ State::PreInit };
    return _v;
}

std::vector<uint8_t>
serialize_xcp_metrics(const bool& use_vcn_activity, const bool& use_jpeg_activity,
                      const amdsmi_gpu_metrics_t& gpu_metrics)
{
    // Chunk:
    // <vcn_data_0>..<vcn_data_[vcn_count]> // lower and higher byte
    // <jpeg_data_0>..<jpeg_data_[jpeg_count]> // lower and higher byte

    // Serialized:
    // <is_vcn_supported>
    // <is_jpeg_supported>
    // <xcp_count>
    // <vcn_count>
    // <jpeg_count>
    // Chunk_0
    // ...
    // Chunk_[xcp_count]

    constexpr uint8_t vcn_count          = AMDSMI_MAX_NUM_VCN;
    constexpr uint8_t jpeg_count         = AMDSMI_MAX_NUM_JPEG;
    constexpr uint8_t xcp_count          = AMDSMI_MAX_NUM_XCP;
    constexpr size_t  elem_size          = sizeof(uint16_t) / sizeof(uint8_t);
    constexpr uint8_t vector_size_header = sizeof(uint8_t);
    constexpr uint8_t serialized_data_headers =
        5 * vector_size_header;  // is_vcn_supported + is_jpeg_supported + xcp_count +
                                 // vcn_count + jpeg_count
    constexpr size_t chunk_size = ((vcn_count + jpeg_count) * elem_size);

    auto serialize_uint16_array = [](std::vector<uint8_t>& data, const uint16_t* arr,
                                     int array_size) {
        for(int i = 0; i < array_size; ++i)
        {
            data.push_back(static_cast<uint8_t>(arr[i] & 0xFF));
            data.push_back(static_cast<uint8_t>((arr[i] >> 8) & 0xFF));
        }
    };

    std::vector<uint8_t> result;

    const bool   is_vcn_jpeg_supported = (use_vcn_activity || use_jpeg_activity);
    const size_t chunk_count           = is_vcn_jpeg_supported ? 1 : xcp_count;
    const size_t total_size = serialized_data_headers + (chunk_count * chunk_size);

    result.reserve(total_size);

    result.push_back((uint8_t) use_vcn_activity);
    result.push_back((uint8_t) use_jpeg_activity);
    result.push_back(chunk_count);
    result.push_back(vcn_count);
    result.push_back(jpeg_count);

    for(size_t count = 0; count < chunk_count; ++count)
    {
        const auto* vcn_data =
            (is_vcn_jpeg_supported ? gpu_metrics.vcn_activity
                                   : gpu_metrics.xcp_stats[count].vcn_busy);
        const auto* jpeg_data =
            (is_vcn_jpeg_supported ? gpu_metrics.jpeg_activity
                                   : gpu_metrics.xcp_stats[count].jpeg_busy);

        serialize_uint16_array(result, vcn_data, vcn_count);
        serialize_uint16_array(result, jpeg_data, jpeg_count);
    }

    return result;
}

size_t
serialize_settings()
{
    return static_cast<size_t>(get_enabled_metrics().value);
}

}  // namespace

//--------------------------------------------------------------------------------------//

void
config()
{
    metadata_initialize_category();

    std::for_each(m_gpu_processors.begin(), m_gpu_processors.end(),
                  [](const auto& device) {
                      auto device_index = device->get_index();
                      metadata_initialize_smi_tracks(device_index);
                      metadata_initialize_smi_pmc(device_index);
                  });
}

void
sample()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };
    benchmark::start(benchmark::category::amd_smi_sample);
    if(amd_smi::get_state() != State::Active)
    {
        return;
    }

    for(auto& processor : m_gpu_processors)
    {
        auto _timestamp = tim::get_clock_real_now<size_t, std::nano>();
        assert(_timestamp < std::numeric_limits<int64_t>::max());

        try
        {
            auto _smi_metrics       = processor->get_smi_metrics();
            auto _supported_metrics = processor->get_supported_metrics();

            auto _power = _supported_metrics.current_socket_power
                              ? _smi_metrics.current_socket_power
                              : _smi_metrics.average_socket_power;
            auto _temp  = _supported_metrics.hotspot_temperature
                              ? static_cast<int32_t>(_smi_metrics.hotspot_temperature)
                              : static_cast<int32_t>(_smi_metrics.edge_temperature);

            trace_cache::get_buffer_storage().store(
                trace_cache::entry_type::amd_smi_sample,
                static_cast<size_t>(get_enabled_metrics().value), 0, _timestamp,
                _smi_metrics.gfx_activity, _smi_metrics.umc_activity,
                _smi_metrics.mm_activity, _power, _temp, _smi_metrics.memory_usage,
                std::vector<uint8_t>(40));
        } catch(const std::runtime_error& e)
        {
            ROCPROFSYS_WARNING(
                0,
                "Reading metrics failed for device with ID %zu. Error: %s. "
                "Disabling device!\n",
                processor->get_index(), e.what());
            auto device_to_remove =
                std::find(m_gpu_processors.begin(), m_gpu_processors.end(), processor);
            m_gpu_processors.erase(device_to_remove);
        }
    }
    benchmark::end(benchmark::category::amd_smi_sample);
}

void
set_state(State _v)
{
    amd_smi::get_state().store(_v);
}
/*

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

    auto use_perfetto = get_use_perfetto();

    for(auto& itr : _amd_smi)
    {
        using counter_track = perfetto_counter_track<data>;
        if(itr.m_dev_id != _dev_id) continue;

        uint64_t _ts = itr.m_ts;
        if(!_thread_info->is_valid_time(_ts)) continue;

        double _gfxbusy = itr.m_busy_perc.gfx_activity;
        double _umcbusy = itr.m_busy_perc.umc_activity;
        double _mmbusy  = itr.m_busy_perc.mm_activity;
        double _temp    = itr.m_temp;
        double _power   = itr.m_power;
        double _usage   = itr.m_mem_usage / static_cast<double>(units::megabyte);

        auto setup_perfetto_counter_tracks = [&]() {
            if(counter_track::exists(_dev_id)) return;

            auto addendum = [&](const char* _v) {
                return JOIN(" ", "GPU", _v, JOIN("", '[', _dev_id, ']'), "(S)");
            };

            auto addendum_blk = [&](std::size_t _i, const char* _metric,
                                    std::size_t xcp_idx = SIZE_MAX) {
                if(xcp_idx != SIZE_MAX)
                {
                    return JOIN(
                        " ", "GPU", JOIN("", '[', _dev_id, ']'), _metric,
                        JOIN("", "XCP_", xcp_idx, ": [", (_i < 10 ? "0" : ""), _i, ']'),
                        "(S)");
                }
                else
                {
                    return JOIN(" ", "GPU", JOIN("", '[', _dev_id, ']'), _metric,
                                JOIN("", "[", (_i < 10 ? "0" : ""), _i, ']'), "(S)");
                }
            };

            if(_settings.busy)
            {
                counter_track::emplace(_dev_id, addendum("GFX Busy"), "%");
                counter_track::emplace(_dev_id, addendum("UMC Busy"), "%");
                counter_track::emplace(_dev_id, addendum("MM Busy"), "%");
            }
            if(_settings.temp)
            {
                counter_track::emplace(_dev_id, addendum("Temperature"), "deg C");
            }
            if(_settings.power)
            {
                counter_track::emplace(_dev_id, addendum("Current Power"), "watts");
            }
            if(_settings.mem_usage)
            {
                counter_track::emplace(_dev_id, addendum("Memory Usage"), "megabytes");
            }
            if(_settings.vcn_activity)
            {
                if(itr.m_xcp_metrics.empty())
                {
                    ROCPROFSYS_VERBOSE(
                        1, "No VCN activity data collected from device %u\n", _dev_id);
                }
                else if(gpu::is_vcn_activity_supported(_dev_id))
                {
                    // For VCN activity, use simple indexing
                    for(std::size_t i = 0; i < std::size(itr.m_xcp_metrics[0].vcn_busy);
                        ++i)
                        counter_track::emplace(_dev_id, addendum_blk(i, "VCN Activity"),
                                               "%");
                }
                else
                {
                    for(std::size_t xcp = 0; xcp < std::size(itr.m_xcp_metrics); ++xcp)
                    {
                        for(std::size_t i = 0;
                            i < std::size(itr.m_xcp_metrics[xcp].vcn_busy); ++i)
                        {
                            counter_track::emplace(
                                _dev_id, addendum_blk(i, "VCN Activity", xcp), "%");
                        }
                    }
                }
            }
            if(_settings.jpeg_activity)
            {
                if(itr.m_xcp_metrics.empty())
                {
                    ROCPROFSYS_VERBOSE(
                        1, "No JPEG activity data collected from device %u\n", _dev_id);
                }
                else if(gpu::is_jpeg_activity_supported(_dev_id))
                {
                    for(std::size_t i = 0; i < std::size(itr.m_xcp_metrics[0].jpeg_busy);
                        ++i)
                        counter_track::emplace(_dev_id, addendum_blk(i, "JPEG Activity"),
                                               "%");
                }
                else
                {
                    for(std::size_t xcp = 0; xcp < std::size(itr.m_xcp_metrics); ++xcp)
                    {
                        for(std::size_t i = 0;
                            i < std::size(itr.m_xcp_metrics[xcp].jpeg_busy); ++i)
                            counter_track::emplace(
                                _dev_id, addendum_blk(i, "JPEG Activity", xcp), "%");
                    }
                }
            }
        };

        auto write_perfetto_metrics = [&]() {
            size_t track_index = 0;

            if(_settings.busy)
            {
                TRACE_COUNTER("device_busy_gfx",
                              counter_track::at(_dev_id, track_index++), _ts, _gfxbusy);
                TRACE_COUNTER("device_busy_umc",
                              counter_track::at(_dev_id, track_index++), _ts, _umcbusy);
                TRACE_COUNTER("device_busy_mm", counter_track::at(_dev_id, track_index++),
                              _ts, _mmbusy);
            }
            if(_settings.temp)
            {
                TRACE_COUNTER("device_temp", counter_track::at(_dev_id, track_index++),
                              _ts, _temp);
            }
            if(_settings.power)
            {
                TRACE_COUNTER("device_power", counter_track::at(_dev_id, track_index++),
                              _ts, _power);
            }
            if(_settings.mem_usage)
            {
                TRACE_COUNTER("device_memory_usage",
                              counter_track::at(_dev_id, track_index++), _ts, _usage);
            }

            if(_settings.vcn_activity && !itr.m_xcp_metrics.empty())
            {
                // Iterate over all XCPs and their VCN busy/activity values
                for(const auto& metrics : itr.m_xcp_metrics)
                {
                    for(const auto& vcn_val : metrics.vcn_busy)
                    {
                        TRACE_COUNTER("device_vcn_activity",
                                      counter_track::at(_dev_id, track_index++), _ts,
                                      vcn_val);
                    }
                }
            }

            if(_settings.jpeg_activity && !itr.m_xcp_metrics.empty())
            {
                // Iterate over all XCPs and their JPEG busy/activity values
                for(const auto& metrics : itr.m_xcp_metrics)
                {
                    for(const auto& jpeg_val : metrics.jpeg_busy)
                    {
                        TRACE_COUNTER("device_jpeg_activity",
                                      counter_track::at(_dev_id, track_index++), _ts,
                                      jpeg_val);
                    }
                }
            }
        };

        if(use_perfetto)
        {
            setup_perfetto_counter_tracks();
            write_perfetto_metrics();
        }
    }
}
*/
//--------------------------------------------------------------------------------------//

void
setup()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(is_initialized() || !get_use_amd_smi()) return;

    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    try
    {
        m_smi_service = std::make_shared<service<amd_smi_driver_factory>>();

        auto _version = m_smi_service->get_version();
        ROCPROFSYS_VERBOSE_F(0, "AMD SMI version: %u.%u.%u - str: %s.\n",
                             _version.numeric_representation.major,
                             _version.numeric_representation.minor,
                             _version.numeric_representation.release,
                             _version.string_representation.c_str());

        m_gpu_processors =
            m_smi_service->get_processors([](const processor_vector_t& processors) {
                auto devices_in_use = get_sampling_gpus();
                return filter_devices(processors, devices_in_use);
            });
        is_initialized() = true;
    } catch(std::runtime_error& _e)
    {
        ROCPROFSYS_VERBOSE(0, "Exception thrown when initializing amd-smi: %s\n",
                           _e.what());
    }
}

void
shutdown()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    if(!is_initialized()) return;
    ROCPROFSYS_VERBOSE_F(1, "Shutting down amd-smi...\n");

    // TODO shutdown smi
    benchmark::show_results();
    is_initialized() = false;
}

void
post_process()
{}

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
