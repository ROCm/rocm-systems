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
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include <amd_smi/amdsmi.h>
#include <cstdint>
#if defined(NDEBUG)
#    undef NDEBUG
#endif

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/debug.hpp"
#include "core/gpu.hpp"
#include "core/gpu_metrics.hpp"
#include "core/node_info.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/amd_smi.hpp"
#include "library/amd_smi/metrics.hpp"
#include "library/amd_smi/sampler_impl.hpp"
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

    if(gpu::vcn_is_device_level_only(gpu_id))
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

    if(gpu::jpeg_is_device_level_only(gpu_id))
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

    // Add XGMI tracks using specific categories for each metric type
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_xgmi_link_width>(
              gpu_id),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_xgmi_link_speed>(
              gpu_id),
          thread_id, "{}" });

    for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
    {
        auto read_name =
            trace_cache::info::annotate_with_device_id<category::amd_smi_xgmi_read_data>(
                gpu_id, std::nullopt, i);
        trace_cache::get_metadata_registry().add_track(
            { read_name.c_str(), thread_id, "{}" });

        auto write_name =
            trace_cache::info::annotate_with_device_id<category::amd_smi_xgmi_write_data>(
                gpu_id, std::nullopt, i);
        trace_cache::get_metadata_registry().add_track(
            { write_name.c_str(), thread_id, "{}" });
    }

    // Add PCIe tracks using specific categories for each metric
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_pcie_link_width>(
              gpu_id),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<category::amd_smi_pcie_link_speed>(
              gpu_id),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<
              category::amd_smi_pcie_bandwidth_acc>(gpu_id),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_device_id<
              category::amd_smi_pcie_bandwidth_inst>(gpu_id),
          thread_id, "{}" });
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

    if(gpu::vcn_is_device_level_only(gpu_id))
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

    if(gpu::jpeg_is_device_level_only(gpu_id))
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

    // Add XGMI PMC info using specific categories for each metric type
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_xgmi_link_width>::value, "XgmiLinkWidth",
          trait::name<category::amd_smi_xgmi_link_width>::description, LONG_DESCRIPTION,
          COMPONENT, "bits", rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0,
          0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_xgmi_link_speed>::value, "XgmiLinkSpeed",
          trait::name<category::amd_smi_xgmi_link_speed>::description, LONG_DESCRIPTION,
          COMPONENT, "GT/s", rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0,
          0 });

    for(size_t i = 0; i < AMDSMI_MAX_NUM_XGMI_LINKS; ++i)
    {
        std::stringstream read_name_ss, read_symbol_ss;
        read_name_ss << trait::name<category::amd_smi_xgmi_read_data>::value << "_" << i;
        read_symbol_ss << "XgmiRead_" << i;

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              read_name_ss.str(), read_symbol_ss.str(),
              trait::name<category::amd_smi_xgmi_read_data>::description,
              LONG_DESCRIPTION, COMPONENT, "KB", rocprofsys::trace_cache::ABSOLUTE, BLOCK,
              EXPRESSION, 0, 0 });

        std::stringstream write_name_ss, write_symbol_ss;
        write_name_ss << trait::name<category::amd_smi_xgmi_write_data>::value << "_"
                      << i;
        write_symbol_ss << "XgmiWrite_" << i;

        trace_cache::get_metadata_registry().add_pmc_info(
            { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
              write_name_ss.str(), write_symbol_ss.str(),
              trait::name<category::amd_smi_xgmi_write_data>::description,
              LONG_DESCRIPTION, COMPONENT, "KB", rocprofsys::trace_cache::ABSOLUTE, BLOCK,
              EXPRESSION, 0, 0 });
    }

    // Add PCIe PMC info using specific categories for each metric
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_pcie_link_width>::value, "PcieLinkWidth",
          trait::name<category::amd_smi_pcie_link_width>::description, LONG_DESCRIPTION,
          COMPONENT, "", rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_pcie_link_speed>::value, "PcieLinkSpeed",
          trait::name<category::amd_smi_pcie_link_speed>::description, LONG_DESCRIPTION,
          COMPONENT, "GT/s", rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0,
          0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_pcie_bandwidth_acc>::value, "PcieBwAcc",
          trait::name<category::amd_smi_pcie_bandwidth_acc>::description,
          LONG_DESCRIPTION, COMPONENT, "MB", rocprofsys::trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, gpu_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_pcie_bandwidth_inst>::value, "PcieBwInst",
          trait::name<category::amd_smi_pcie_bandwidth_inst>::description,
          LONG_DESCRIPTION, COMPONENT, "MB/s", rocprofsys::trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0 });
}

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

// Legacy wrapper functions - delegate to metrics module for consistency
std::vector<uint8_t>
serialize_gpu_metrics(uint32_t device_id, const data::gpu_metrics_t& gpu_metrics,
                      const gpu::gpu_metrics_capabilities_t& capabilities)
{
    return metrics::serialize_gpu_metrics(gpu_metrics, capabilities,
                                          get_settings(device_id));
}

size_t
serialize_settings(uint32_t _device_id)
{
    return metrics::serialize_settings(get_settings(_device_id));
}

}  // namespace

//--------------------------------------------------------------------------------------//

size_t                           data::device_count     = 0;
std::set<uint32_t>               data::device_list      = {};
std::unique_ptr<data::promise_t> data::polling_finished = {};

data::data(uint32_t _dev_id) { sample(_dev_id); }

namespace
{
// Get the shared Sampler instance for this module
// Uses template-based policies for zero virtual function overhead
Sampler&
get_sampler()
{
    // Create sampler with proper state reference to the global state
    // All policy calls are resolved at compile time - no virtual dispatch
    static Sampler instance(get_state());
    return instance;
}

}  // namespace

void
data::sample(uint32_t _device_id)
{
    // Use the refactored Sampler class which is fully unit testable
    // The Sampler handles all the complexity internally with dependency injection
    auto result = get_sampler().sample(_device_id, get_settings(_device_id));

    if(result.has_value())
    {
        // Copy results to this data object for backward compatibility
        m_dev_id      = result->m_dev_id;
        m_ts          = result->m_ts;
        m_busy_perc   = result->m_busy_perc;
        m_temp        = result->m_temp;
        m_power       = result->m_power;
        m_mem_usage   = result->m_mem_usage;
        m_gpu_metrics = result->m_gpu_metrics;
    }
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

    metadata_initialize_category();

    for(const auto& _dev_id : data::device_list)
    {
        metadata_initialize_smi_tracks(_dev_id);
        metadata_initialize_smi_pmc(_dev_id);
    }
}

void
sample()
{
    auto_lock_t _lk{ type_mutex<category::amd_smi>() };

    // TODO: Reorganize amd_smi::data and sampling mechanism not to store same data in
    // bundle_data and in trace_cache

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
        double _power   = itr.m_power.current_socket_power;
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
                if(itr.m_gpu_metrics.empty())
                {
                    ROCPROFSYS_VERBOSE(
                        1, "No VCN activity data collected from device %u\n", _dev_id);
                }
                else if(gpu::vcn_is_device_level_only(_dev_id))
                {
                    // For VCN activity supported: use vcn_activity vector
                    for(std::size_t i = 0;
                        i < std::size(itr.m_gpu_metrics[0].vcn_activity); ++i)
                        counter_track::emplace(_dev_id, addendum_blk(i, "VCN Activity"),
                                               "%");
                }
                else
                {
                    // For VCN activity NOT supported: use vcn_busy vector with per-XCP
                    // organization
                    for(size_t xcp = 0; xcp < itr.m_gpu_metrics[0].vcn_busy.size(); ++xcp)
                    {
                        // Loop through each XCP's VCN busy values
                        for(size_t i = 0; i < itr.m_gpu_metrics[0].vcn_busy[xcp].size();
                            ++i)
                        {
                            counter_track::emplace(
                                _dev_id, addendum_blk(i, "VCN Activity", xcp), "%");
                        }
                    }
                }
            }
            if(_settings.jpeg_activity)
            {
                if(itr.m_gpu_metrics.empty())
                {
                    ROCPROFSYS_VERBOSE(
                        1, "No JPEG activity data collected from device %u\n", _dev_id);
                }
                else if(gpu::jpeg_is_device_level_only(_dev_id))
                {
                    // For JPEG activity supported: use jpeg_activity vector
                    for(std::size_t i = 0;
                        i < std::size(itr.m_gpu_metrics[0].jpeg_activity); ++i)
                        counter_track::emplace(_dev_id, addendum_blk(i, "JPEG Activity"),
                                               "%");
                }
                else
                {
                    // For JPEG activity NOT supported: use jpeg_busy vector with per-XCP
                    // organization
                    for(size_t xcp = 0; xcp < itr.m_gpu_metrics[0].jpeg_busy.size();
                        ++xcp)
                    {
                        // Loop through each XCP's JPEG busy values
                        for(size_t i = 0; i < itr.m_gpu_metrics[0].jpeg_busy[xcp].size();
                            ++i)
                        {
                            counter_track::emplace(
                                _dev_id, addendum_blk(i, "JPEG Activity", xcp), "%");
                        }
                    }
                }
            }
            if(_settings.xgmi)
            {
                if(itr.m_gpu_metrics.empty())
                {
                    ROCPROFSYS_VERBOSE(
                        1, "No XGMI activity data collected from device %u\n", _dev_id);
                }
                else
                {
                    counter_track::emplace(_dev_id, addendum("XGMI Link Width"), "bits");
                    counter_track::emplace(_dev_id, addendum("XGMI Link Speed"), "GT/s");
                    for(std::size_t i = 0;
                        i < std::size(itr.m_gpu_metrics[0].xgmi_read_data_acc); ++i)
                        counter_track::emplace(_dev_id, addendum_blk(i, "XGMI Read Data"),
                                               "KB");
                    for(std::size_t i = 0;
                        i < std::size(itr.m_gpu_metrics[0].xgmi_write_data_acc); ++i)
                        counter_track::emplace(_dev_id,
                                               addendum_blk(i, "XGMI Write Data"), "KB");
                }
            }
            if(_settings.pcie)
            {
                if(itr.m_gpu_metrics.empty())
                {
                    ROCPROFSYS_VERBOSE(
                        1, "No PCIe activity data collected from device %u\n", _dev_id);
                }
                else
                {
                    counter_track::emplace(_dev_id, addendum("PCIe Link Width"), "");
                    counter_track::emplace(_dev_id, addendum("PCIe Link Speed"), "GT/s");
                    counter_track::emplace(_dev_id, addendum("PCIe Bandwidth Acc"), "MB");
                    counter_track::emplace(_dev_id, addendum("PCIe Bandwidth Inst"),
                                           "MB/s");
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

            if(_settings.vcn_activity && !itr.m_gpu_metrics.empty())
            {
                if(gpu::vcn_is_device_level_only(_dev_id))
                {
                    // Device-level VCN activity
                    for(const auto& vcn_val : itr.m_gpu_metrics[0].vcn_activity)
                    {
                        TRACE_COUNTER("device_vcn_activity",
                                      counter_track::at(_dev_id, track_index++), _ts,
                                      vcn_val);
                    }
                }
                else
                {
                    // XCP-level VCN busy (per-XCP organization)
                    for(const auto& xcp_data : itr.m_gpu_metrics[0].vcn_busy)
                    {
                        for(const auto& vcn_val : xcp_data)
                        {
                            TRACE_COUNTER("device_vcn_activity",
                                          counter_track::at(_dev_id, track_index++), _ts,
                                          vcn_val);
                        }
                    }
                }
            }

            if(_settings.jpeg_activity && !itr.m_gpu_metrics.empty())
            {
                if(gpu::jpeg_is_device_level_only(_dev_id))
                {
                    // Device-level JPEG activity
                    for(const auto& jpeg_val : itr.m_gpu_metrics[0].jpeg_activity)
                    {
                        TRACE_COUNTER("device_jpeg_activity",
                                      counter_track::at(_dev_id, track_index++), _ts,
                                      jpeg_val);
                    }
                }
                else
                {
                    // XCP-level JPEG busy (per-XCP organization)
                    for(const auto& xcp_data : itr.m_gpu_metrics[0].jpeg_busy)
                    {
                        for(const auto& jpeg_val : xcp_data)
                        {
                            TRACE_COUNTER("device_jpeg_activity",
                                          counter_track::at(_dev_id, track_index++), _ts,
                                          jpeg_val);
                        }
                    }
                }
            }

            if(_settings.xgmi && !itr.m_gpu_metrics.empty())
            {
                TRACE_COUNTER("device_xgmi_link_width",
                              counter_track::at(_dev_id, track_index++), _ts,
                              itr.m_gpu_metrics[0].xgmi_link_width);
                TRACE_COUNTER("device_xgmi_link_speed",
                              counter_track::at(_dev_id, track_index++), _ts,
                              itr.m_gpu_metrics[0].xgmi_link_speed);
                for(const auto& read_val : itr.m_gpu_metrics[0].xgmi_read_data_acc)
                {
                    TRACE_COUNTER("device_xgmi_read_data",
                                  counter_track::at(_dev_id, track_index++), _ts,
                                  read_val);
                }

                for(const auto& write_val : itr.m_gpu_metrics[0].xgmi_write_data_acc)
                {
                    TRACE_COUNTER("device_xgmi_write_data",
                                  counter_track::at(_dev_id, track_index++), _ts,
                                  write_val);
                }
            }

            if(_settings.pcie && !itr.m_gpu_metrics.empty())
            {
                TRACE_COUNTER("device_pcie_link_width",
                              counter_track::at(_dev_id, track_index++), _ts,
                              itr.m_gpu_metrics[0].pcie_link_width);
                TRACE_COUNTER("device_pcie_link_speed",
                              counter_track::at(_dev_id, track_index++), _ts,
                              itr.m_gpu_metrics[0].pcie_link_speed);
                TRACE_COUNTER("device_pcie_bandwidth_acc",
                              counter_track::at(_dev_id, track_index++), _ts,
                              itr.m_gpu_metrics[0].pcie_bandwidth_acc);
                TRACE_COUNTER("device_pcie_bandwidth_inst",
                              counter_track::at(_dev_id, track_index++), _ts,
                              itr.m_gpu_metrics[0].pcie_bandwidth_inst);
            }
        };

        if(use_perfetto)
        {
            setup_perfetto_counter_tracks();
            write_perfetto_metrics();
        }
    }
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

    data::device_count = gpu::device_count();

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
                    key_pair_t{ "xgmi", get_settings(itr).xgmi },
                    key_pair_t{ "pcie", get_settings(itr).pcie },
                };

                // Initialize all metrics to false
                for(const auto& it : supported)
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
    {
        ROCPROFSYS_VERBOSE(2, "Post-processing amd-smi data for device: %d", itr);
        data::post_process(itr);
    }
}

uint32_t
device_count()
{
    return gpu::device_count();
}

void
postfork_child_cleanup()
{
    // In child process, disable AMD SMI to prevent shutdown errors
    ROCPROFSYS_VERBOSE_F(2, "Disabling AMD SMI in child process after fork...\n");

    // Set to Finalized to prevent any sampling attempts (though is_child_process() check
    // in sample() already handles this)
    get_state().store(State::Finalized);

    // Mark as not initialized so shutdown won't try to cleanup AMD SMI library
    is_initialized() = false;

    // Clear device list to prevent any GPU operations
    data::device_list.clear();
}

void
postfork_parent_reinit()
{
    // In parent process, AMD SMI device handles may be corrupted after fork
    // Reinitialize AMD SMI to get fresh handles
    ROCPROFSYS_VERBOSE_F(2, "Reinitializing AMD SMI in parent process after fork...\n");

    // Shutdown and reinitialize to get fresh device handles
    shutdown();
    setup();
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
