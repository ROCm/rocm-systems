#pragma once

#include "core/config.hpp"
#include "core/perfetto.hpp"
#include "library/amd_smi/common.hpp"
#include "library/thread_info.hpp"

#include <vector>

namespace rocprofsys
{
namespace amd_smi
{

namespace
{
struct amd_smi_sample
{
    size_t      timestamp;
    smi_metrics metrics;
};

// TODO: we are keeping this just to support the old perfetto way
std::map<size_t, std::unique_ptr<std::vector<amd_smi_sample>>> g_perfetto_bundle;

}  // namespace

struct perfetto
{
    static void setup_counter_tracks(const size_t              device_index,
                                     const smi_metric_options& metrics)
    {
        if(!get_use_perfetto())
        {
            return;
        }

        using counter_track = perfetto_counter_track<smi_metrics>;

        auto addendum = [&](const char* _v) {
            return JOIN(" ", "GPU", _v, JOIN("", '[', device_index, ']'), "(S)");
        };

        auto addendum_blk = [&](std::size_t _i, const char* _metric,
                                std::size_t xcp_idx = SIZE_MAX) {
            if(xcp_idx != SIZE_MAX)
            {
                auto result =
                    JOIN(" ", "GPU", JOIN("", '[', device_index, ']'), _metric,
                         JOIN("", "XCP_", xcp_idx, ": [", (_i < 10 ? "0" : ""), _i, ']'),
                         "(S)");
                // std::cout << result << std::endl;
                return result;
            }
            else
            {
                return JOIN(" ", "GPU", JOIN("", '[', device_index, ']'), _metric,
                            JOIN("", "[", (_i < 10 ? "0" : ""), _i, ']'), "(S)");
            }
        };
        struct track_description
        {
            const char* const track_name;
            const char* const units;
        };
        std::map<uint32_t, track_description> perfetto_track_names{
            { smi_metric_options{ .bits{ .gfx_activity = 1 } }.value,
              { "GFX Busy", "%" } },
            { smi_metric_options{ .bits{ .umc_activity = 1 } }.value,
              { "UMC Busy", "%" } },
            { smi_metric_options{ .bits{ .mm_activity = 1 } }.value, { "MM Busy", "%" } },
            { smi_metric_options{
                  .bits{ .hotspot_temperature = 1, .edge_temperature = 1 } }
                  .value,
              { "Temperature ", "deg C" } },
            { smi_metric_options{
                  .bits{ .current_socket_power = 1, .average_socket_power = 1 } }
                  .value,
              { "Current Power", "watts" } },
            { smi_metric_options{ .bits{ .memory_usage = 1 } }.value,
              { "Memory Usage", "megabytes" } },
            { smi_metric_options{ .bits{ .vcn_activity = 1 } }.value,
              { "VCN Activity", "%" } },
            { smi_metric_options{ .bits{ .jpeg_activity = 1 } }.value,
              { "JPEG Activity", "%" } },
        };

        for(auto& [num, description] : perfetto_track_names)
        {
            auto metric_map_id = num & metrics.value;
            if(metric_map_id > 0)
            {
                if(metric_map_id ==
                   smi_metric_options{ .bits{ .vcn_activity = 1 } }.value)
                {
                    for(std::size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
                    {
                        for(std::size_t i = 0; i < AMDSMI_MAX_NUM_VCN; ++i)
                        {
                            counter_track::emplace(
                                device_index, addendum_blk(i, "VCN Activity", xcp), "%");
                        }
                    }
                }
                else if(metric_map_id ==
                        smi_metric_options{ .bits{ .jpeg_activity = 1 } }.value)
                {
                    for(std::size_t xcp = 0; xcp < AMDSMI_MAX_NUM_XCP; ++xcp)
                    {
                        for(std::size_t i = 0; i < AMDSMI_MAX_NUM_JPEG_ENGINES; ++i)
                        {
                            counter_track::emplace(
                                device_index, addendum_blk(i, "JPEG Activity", xcp), "%");
                        }
                    }
                }
                else
                {
                    counter_track::emplace(device_index, addendum(description.track_name),
                                           description.units);
                }
            }
        }
    };

    static void init_storage(size_t device_index)
    {
        g_perfetto_bundle.insert(
            { device_index, std::make_unique<std::vector<amd_smi_sample>>() });
    };

    static void store_sample(size_t device_index, const smi_metrics& _smi_metrics,
                             unsigned long _timestamp)
    {
        if(get_use_perfetto())
        {
            g_perfetto_bundle[device_index]->emplace_back(
                amd_smi_sample{ _timestamp, _smi_metrics });
        }
    };

    static void post_process(size_t device_index, smi_metric_options enabled_metrics,
                             smi_metric_options supported_metrics)
    {
        if(!get_use_perfetto())
        {
            return;
        }

        using counter_track = perfetto_counter_track<smi_metrics>;

        auto        _amd_smi     = *g_perfetto_bundle[device_index];
        const auto& _thread_info = thread_info::get(0, InternalTID);

        ROCPROFSYS_VERBOSE(1, "Post-processing %zu amd-smi samples from device %zu\n",
                           _amd_smi.size(), device_index);

        ROCPROFSYS_CI_THROW(!_thread_info, "Missing thread info for thread 0");
        if(!_thread_info) return;

        smi_metric_options _enabled_metrics = { .value = enabled_metrics.value &
                                                         supported_metrics.value };

        for(auto& itr : _amd_smi)
        {
            uint64_t _ts = itr.timestamp;
            if(!_thread_info->is_valid_time(_ts)) continue;

            double _gfxbusy = itr.metrics.gfx_activity;
            double _umcbusy = itr.metrics.umc_activity;
            double _mmbusy  = itr.metrics.mm_activity;
            double _temp    = itr.metrics.hotspot_temperature;
            double _power   = _enabled_metrics.bits.average_socket_power
                                  ? itr.metrics.average_socket_power
                                  : itr.metrics.current_socket_power;
            double _usage =
                itr.metrics.memory_usage / static_cast<double>(units::megabyte);

            size_t track_index = 0;

            if(_enabled_metrics.bits.gfx_activity)
            {
                TRACE_COUNTER("device_busy_gfx",
                              counter_track::at(device_index, track_index++), _ts,
                              _gfxbusy);
            }
            if(_enabled_metrics.bits.umc_activity)
            {
                TRACE_COUNTER("device_busy_umc",
                              counter_track::at(device_index, track_index++), _ts,
                              _umcbusy);
            }
            if(_enabled_metrics.bits.mm_activity)
            {
                TRACE_COUNTER("device_busy_mm",
                              counter_track::at(device_index, track_index++), _ts,
                              _mmbusy);
            }
            if(_enabled_metrics.bits.edge_temperature ||
               _enabled_metrics.bits.hotspot_temperature)
            {
                TRACE_COUNTER("device_temp",
                              counter_track::at(device_index, track_index++), _ts, _temp);
            }
            if(_enabled_metrics.bits.average_socket_power ||
               _enabled_metrics.bits.current_socket_power)
            {
                TRACE_COUNTER("device_power",
                              counter_track::at(device_index, track_index++), _ts,
                              _power);
            }
            if(_enabled_metrics.bits.memory_usage)
            {
                TRACE_COUNTER("device_memory_usage",
                              counter_track::at(device_index, track_index++), _ts,
                              _usage);
            }
            if(_enabled_metrics.bits.vcn_activity)
            {
                std::for_each(
                    std::begin(itr.metrics.xcp_stats), std::end(itr.metrics.xcp_stats),
                    [&](const auto& xcp_stats) {
                        std::for_each(
                            std::begin(xcp_stats.vcn_busy), std::end(xcp_stats.vcn_busy),
                            [&](const auto& vcn_val) {
                                if(vcn_val != std::numeric_limits<uint16_t>::max())
                                {
                                    TRACE_COUNTER(
                                        "device_vcn_activity",
                                        counter_track::at(device_index, track_index++),
                                        _ts, vcn_val);
                                }
                            });
                    });
            }

            if(_enabled_metrics.bits.jpeg_activity)
            {
                std::for_each(
                    std::begin(itr.metrics.xcp_stats), std::end(itr.metrics.xcp_stats),
                    [&](const auto& xcp_stats) {
                        std::for_each(
                            std::begin(xcp_stats.jpeg_busy),
                            std::end(xcp_stats.jpeg_busy), [&](const auto& jpeg_val) {
                                if(jpeg_val != std::numeric_limits<uint16_t>::max())
                                {
                                    TRACE_COUNTER(
                                        "device_jpeg_activity",
                                        counter_track::at(device_index, track_index++),
                                        _ts, jpeg_val);
                                }
                            });
                    });
            }
        }
    }
};

}  // namespace amd_smi
}  // namespace rocprofsys
