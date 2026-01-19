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

#include "library/gpu_process_stats.hpp"
#include "core/categories.hpp"
#include "core/common.hpp"
#include "core/config.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "core/timemory.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "library/amd_smi.hpp"
#include "library/process_sampler.hpp"
#include "library/runtime.hpp"
#include "library/thread_data.hpp"
#include "library/thread_info.hpp"

#include <timemory/utility/types.hpp>

#include <cassert>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
#include <vector>

#if ROCPROFSYS_USE_ROCM > 0
#    include <amd_smi/amdsmi.h>
#endif

namespace rocprofsys
{
namespace gpu_process_stats
{
namespace
{
// Process-level data storage
std::vector<process_info_t> process_samples       = {};
std::mutex                  process_samples_mutex = {};
pid_t                       target_process_id     = 0;  // Cached target PID

// Singleton accessor for process settings
process_settings&
get_process_settings_impl()
{
    static process_settings _v{};
    return _v;
}
}  // namespace

process_settings&
get_process_settings()
{
    return get_process_settings_impl();
}

void
initialize(pid_t target_pid)
{
    target_process_id = target_pid;
    LOG_DEBUG("GPU process stats initialized for target PID: {}", target_pid);
}

void
sample(pid_t _pid)
{
    if(is_child_process()) return;

    auto _state = get_state();
    if(_state != State::Active) return;

    // Check if any process metrics are enabled
    auto _proc_settings           = get_process_settings();
    bool _process_metrics_enabled = _proc_settings.vram_usage ||
                                    _proc_settings.sdma_usage ||
                                    _proc_settings.cu_occupancy;

    if(!_process_metrics_enabled) return;

    [[maybe_unused]] auto _timestamp = tim::get_clock_real_now<size_t, std::nano>();
    assert(_timestamp < std::numeric_limits<int64_t>::max());

#if ROCPROFSYS_USE_ROCM > 0
    amdsmi_process_info_t _proc_info;

    try
    {
        // Call the AMD SMI API to get process info for the target PID
        auto _status = amdsmi_get_gpu_compute_process_info_by_pid(_pid, &_proc_info);

        if(_status == AMDSMI_STATUS_SUCCESS)
        {
            // Store process info
            process_info_t _info;
            _info.process_id   = _proc_info.process_id;
            _info.vram_usage   = _proc_info.vram_usage;
            _info.sdma_usage   = _proc_info.sdma_usage;
            _info.cu_occupancy = _proc_info.cu_occupancy;

            // Lock and store the sample
            {
                std::lock_guard<std::mutex> _lk(process_samples_mutex);
                process_samples.push_back(_info);
            }

            LOG_DEBUG("Collected process GPU metrics for PID {}: VRAM={} MB, "
                      "SDMA={} us, CU={}",
                      _pid, _info.vram_usage, _info.sdma_usage, _info.cu_occupancy);
        }
        else if(_status == AMDSMI_STATUS_NOT_SUPPORTED)
        {
            LOG_DEBUG("Process GPU metrics not supported by AMD SMI. Disabling...\n");
            // Disable all process metrics
            get_process_settings().vram_usage   = false;
            get_process_settings().sdma_usage   = false;
            get_process_settings().cu_occupancy = false;
        }
        else
        {
            LOG_DEBUG("Failed to get process GPU info for PID {}: status={}", _pid,
                      static_cast<int>(_status));
        }
    } catch(std::runtime_error& _e)
    {
        LOG_DEBUG(
            "[gpu_process_stats::sample] Exception: {}. Disabling process metrics...",
            _e.what());
        // Disable all process metrics on error
        get_process_settings().vram_usage   = false;
        get_process_settings().sdma_usage   = false;
        get_process_settings().cu_occupancy = false;
    }
#else
    (void) _pid;  // Suppress unused parameter warning
#endif
}

void
post_process()
{
    std::lock_guard<std::mutex> _lk(process_samples_mutex);

    if(process_samples.empty())
    {
        LOG_DEBUG("No process GPU metrics collected.");
        return;
    }

    LOG_DEBUG("Post-processing {} process GPU metric samples", process_samples.size());

    // Get process settings (global, not device-specific)
    auto _proc_settings = get_process_settings();

    // Check if any process metrics are enabled
    bool _any_enabled = _proc_settings.vram_usage || _proc_settings.sdma_usage ||
                        _proc_settings.cu_occupancy;

    if(!_any_enabled) return;

    const auto& _thread_info = thread_info::get(0, InternalTID);
    if(!_thread_info) return;

    auto use_perfetto = get_use_perfetto();

    using counter_track = perfetto_counter_track<amd_smi::data>;

    // Setup perfetto counter tracks for process metrics
    auto setup_process_perfetto_tracks = [&]() {
        // Use a special "device id" for process metrics (e.g., max value)
        uint32_t _process_track_id = std::numeric_limits<uint32_t>::max();

        if(counter_track::exists(_process_track_id)) return;

        auto make_track_name = [](const char* metric) {
            return JOIN(" ", "Process GPU", metric, "(S)");
        };

        if(_proc_settings.vram_usage)
        {
            counter_track::emplace(_process_track_id, make_track_name("VRAM Usage"),
                                   "MB");
        }
        if(_proc_settings.sdma_usage)
        {
            counter_track::emplace(_process_track_id, make_track_name("SDMA Usage"),
                                   "us");
        }
        if(_proc_settings.cu_occupancy)
        {
            counter_track::emplace(_process_track_id, make_track_name("CU Occupancy"),
                                   "%");
        }
    };

    if(use_perfetto)
    {
        setup_process_perfetto_tracks();

        uint32_t _process_track_id = std::numeric_limits<uint32_t>::max();
        size_t   track_index       = 0;

        // Write perfetto metrics for each sample
        // For now, just report the final values
        if(!process_samples.empty())
        {
            const auto& last_sample = process_samples.back();
            auto        _ts         = tim::get_clock_real_now<size_t, std::nano>();

            if(_proc_settings.vram_usage)
            {
                TRACE_COUNTER("amd_smi",
                              counter_track::at(_process_track_id, track_index++), _ts,
                              last_sample.vram_usage);
            }
            if(_proc_settings.sdma_usage)
            {
                TRACE_COUNTER("amd_smi",
                              counter_track::at(_process_track_id, track_index++), _ts,
                              last_sample.sdma_usage);
            }
            if(_proc_settings.cu_occupancy)
            {
                TRACE_COUNTER("amd_smi",
                              counter_track::at(_process_track_id, track_index++), _ts,
                              last_sample.cu_occupancy);
            }
        }
    }

    // Clear processed samples
    process_samples.clear();
}

void
initialize_perfetto_tracks()
{
    const auto thread_id = std::nullopt;

    // Add tracks for process-specific metrics (not tied to a specific device)
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::amd_smi_process_vram_usage>::value, thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::amd_smi_process_sdma_usage>::value, thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trait::name<category::amd_smi_process_cu_occupancy>::value, thread_id, "{}" });
}

void
initialize_pmc()
{
    // TODO: Find the proper values for following definitions
    size_t      EVENT_CODE       = 0;
    size_t      INSTANCE_ID      = 0;
    const char* LONG_DESCRIPTION = "";
    const char* COMPONENT        = "";
    const char* BLOCK            = "";
    const char* EXPRESSION       = "";
    const char* TARGET_ARCH      = "GPU";
    uint32_t    agent_id         = 0;  // Process metrics not tied to specific device

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, agent_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_process_vram_usage>::value, "ProcVRAM",
          trait::name<category::amd_smi_process_vram_usage>::description,
          LONG_DESCRIPTION, COMPONENT, "MB", trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, agent_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_process_sdma_usage>::value, "ProcSDMA",
          trait::name<category::amd_smi_process_sdma_usage>::description,
          LONG_DESCRIPTION, COMPONENT, "us", trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0 });

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::GPU, agent_id, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_process_cu_occupancy>::value, "ProcCUOcc",
          trait::name<category::amd_smi_process_cu_occupancy>::description,
          LONG_DESCRIPTION, COMPONENT, trace_cache::PERCENTAGE,
          trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0 });
}

void
configure(const std::string& env_value)
{
    using key_pair_t     = std::pair<std::string_view, bool&>;
    const auto supported = std::unordered_map<std::string_view, bool&>{
        key_pair_t{ "process_vram_usage", get_process_settings().vram_usage },
        key_pair_t{ "process_sdma_usage", get_process_settings().sdma_usage },
        key_pair_t{ "process_cu_occupancy", get_process_settings().cu_occupancy },
    };

    // Initialize all process metrics to false
    for(auto& it : supported)
        it.second = false;

    auto _keys = tim::delimit(env_value, ", ;:");
    for(auto itr : _keys)
    {
        if(supported.find(itr) != supported.end())
        {
            LOG_DEBUG("Enabling AMD SMI process metric: %s\n", itr.data());
            supported.at(itr) = true;
        }
    }
}

}  // namespace gpu_process_stats
}  // namespace rocprofsys
