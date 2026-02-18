// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/amd_smi_ainic.hpp"

#include "core/agent_manager.hpp"
#include "core/trace_cache/cache_manager.hpp"
#include "core/trace_cache/cacheable.hpp"
#include "core/trace_cache/sample_type.hpp"
#include <cstdint>
#if defined(NDEBUG)
#    undef NDEBUG
#endif

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/config.hpp"
#include "core/gpu.hpp"
#include "core/gpu_metrics.hpp"
#include "core/node_info.hpp"
#include "core/perfetto.hpp"
#include "core/state.hpp"
#include "core/trace_cache/metadata_registry.hpp"
#include "library/amd_smi.hpp"
#include "library/runtime.hpp"
#include "library/thread_info.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/components/timing/backends.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/units.hpp>
#include <timemory/utility/delimit.hpp>
#include <timemory/utility/locking.hpp>

#include "logger/debug.hpp"

#include <cassert>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <unordered_set>

namespace rocprofsys
{
namespace amd_smi
{

using nic_bundle_t          = std::deque<nic_data>;
using nic_sampler_instances = thread_data<nic_bundle_t, category::amd_smi_nic>;
static std::vector<nic_bundle_t> nic_sampler_vec   = {};
std::vector<std::string>         nic_data::nic_vec = {};
ai_nic_stats_collector           nic_data::nic_stats_collector;

namespace
{
std::vector<unique_ptr_t<nic_bundle_t>*> _nic_bundle_data{};
}  // namespace

nic_data::nic_data(uint32_t nic_index, const std::string& nic)
: _nic(nic)
, _nic_index(nic_index)
{
    sample();
}

void
nic_data::sample()
{
    nic_stats stats;
    nic_data::nic_stats_collector.get_data(_nic, stats);
    _rx_rdma_cnp_pkts = stats._rx_rdma_cnp_pkts;
    _tx_rdma_cnp_pkts = stats._tx_rdma_cnp_pkts;
    _rx_ucast_bytes   = stats._rx_rdma_ucast_bytes;
    _tx_ucast_bytes   = stats._tx_rdma_ucast_bytes;
    _rx_ucast_pkts    = stats._rx_rdma_ucast_pkts;
    _tx_ucast_pkts    = stats._tx_rdma_ucast_pkts;

    auto _timestamp = tim::get_clock_real_now<size_t, std::nano>();
    assert(_timestamp < std::numeric_limits<int64_t>::max());
    m_ts = _timestamp;

    trace_cache::get_buffer_storage().store(trace_cache::ainic_sample{
        _timestamp, _nic_index, _rx_rdma_cnp_pkts, _tx_rdma_cnp_pkts, _rx_ucast_bytes,
        _tx_ucast_bytes, _rx_ucast_pkts, _tx_ucast_pkts });
}

bool
nic_data::setup()
{
    perfetto_counter_track<nic_data>::init();
    return true;
}

void
metadata_initialize_ainic_smi_tracks(uint32_t nic_index)
{
    const auto   thread_id = std::nullopt;
    std::string& nic       = nic_data::nic_vec[nic_index];

    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_nic<category::amd_smi_nic_rx_cnp_pkts>(
              nic, nic_index),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_nic<category::amd_smi_nic_tx_cnp_pkts>(
              nic, nic_index),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_nic<category::amd_smi_nic_rx_ucast_bytes>(
              nic, nic_index),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_nic<category::amd_smi_nic_tx_ucast_bytes>(
              nic, nic_index),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_nic<category::amd_smi_nic_rx_ucast_pkts>(
              nic, nic_index),
          thread_id, "{}" });
    trace_cache::get_metadata_registry().add_track(
        { trace_cache::info::annotate_with_nic<category::amd_smi_nic_tx_ucast_pkts>(
              nic, nic_index),
          thread_id, "{}" });
}

void
metadata_initialize_ainic_smi_pmc(uint32_t nic_index)
{
    size_t      EVENT_CODE       = 0;
    size_t      INSTANCE_ID      = 0;
    const char* LONG_DESCRIPTION = "";
    const char* COMPONENT        = "";
    const char* BLOCK            = "";
    const char* EXPRESSION       = "";
    auto        ni               = node_info::get_instance();
    const char* TARGET_ARCH      = "";

    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::NIC, nic_index, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_nic_rx_cnp_pkts>::value, "NIC RX CNP PKTS",
          trait::name<category::amd_smi_nic_rx_cnp_pkts>::description, LONG_DESCRIPTION,
          COMPONENT, trace_cache::ABSOLUTE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0, "{}" });
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::NIC, nic_index, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_nic_tx_cnp_pkts>::value, "NIC TX CNP PKTS",
          trait::name<category::amd_smi_nic_tx_cnp_pkts>::description, LONG_DESCRIPTION,
          COMPONENT, trace_cache::ABSOLUTE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0, "{}" });
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::NIC, nic_index, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_nic_rx_ucast_bytes>::value,
          "AI NIC RX UCAST BYTES",
          trait::name<category::amd_smi_nic_rx_ucast_bytes>::description,
          LONG_DESCRIPTION, COMPONENT, trace_cache::ABSOLUTE,
          rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0, "{}" });
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::NIC, nic_index, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_nic_tx_ucast_bytes>::value,
          "AI NIC TX UCAST BYTES",
          trait::name<category::amd_smi_nic_tx_ucast_bytes>::description,
          LONG_DESCRIPTION, COMPONENT, trace_cache::ABSOLUTE,
          rocprofsys::trace_cache::ABSOLUTE, BLOCK, EXPRESSION, 0, 0, "{}" });
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::NIC, nic_index, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_nic_rx_ucast_pkts>::value, "AI NIC RX UCAST PKTS",
          trait::name<category::amd_smi_nic_rx_ucast_pkts>::description, LONG_DESCRIPTION,
          COMPONENT, trace_cache::ABSOLUTE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0, "{}" });
    trace_cache::get_metadata_registry().add_pmc_info(
        { agent_type::NIC, nic_index, TARGET_ARCH, EVENT_CODE, INSTANCE_ID,
          trait::name<category::amd_smi_nic_tx_ucast_pkts>::value, "AI NIC TX UCAST PKTS",
          trait::name<category::amd_smi_nic_tx_ucast_pkts>::description, LONG_DESCRIPTION,
          COMPONENT, trace_cache::ABSOLUTE, rocprofsys::trace_cache::ABSOLUTE, BLOCK,
          EXPRESSION, 0, 0, "{}" });
}

void
nic_config()
{
    for(uint32_t nic_index = 0; nic_index < nic_data::nic_vec.size(); ++nic_index)
    {
        auto nic_bundle = std::deque<nic_data>{};
        nic_sampler_vec.push_back(nic_bundle);
        metadata_initialize_ainic_smi_tracks(nic_index);
        metadata_initialize_ainic_smi_pmc(nic_index);
    }
}
void
nic_sample()
{
    if(amd_smi::get_state() != State::Active) return;

    // Get AI NIC data for all NICs at once.
    nic_data::nic_stats_collector.update_stats();

    for(uint32_t nic_index = 0; nic_index < nic_data::nic_vec.size(); ++nic_index)
    {
        std::string& nic  = nic_data::nic_vec[nic_index];
        auto         data = nic_data{ nic_index, nic };
        nic_sampler_vec[nic_index].push_back(data);
    }
}

void
nic_data::post_process(size_t nic_index)
{
    using counter_track = perfetto_counter_track<nic_data>;
    std::string& nic    = nic_data::nic_vec[nic_index];

    const auto& _thread_info = thread_info::get(0, InternalTID);
    if(!_thread_info)
    {
        if(get_is_continuous_integration())
        {
            throw std::runtime_error("Missing thread info for thread 0");
        }
        LOG_ERROR("Missing thread info for thread 0");
        return;
    }

    auto addendum = [&](const char* _v) {
        return fmt::format("{} {} [ {} ] (S)", nic, _v, nic_index);
    };

    for(auto& itr : nic_sampler_vec[nic_index])
    {
        uint64_t _ts = itr.m_ts;
        if(!_thread_info->is_valid_time(_ts)) continue;

        uint64_t _rx_rdma_cnp_pkts = itr._rx_rdma_cnp_pkts;
        uint64_t _tx_rdma_cnp_pkts = itr._tx_rdma_cnp_pkts;
        uint64_t _rx_ucast_bytes   = itr._rx_ucast_bytes;
        uint64_t _tx_ucast_bytes   = itr._tx_ucast_bytes;
        uint64_t _rx_ucast_pkts    = itr._rx_ucast_pkts;
        uint64_t _tx_ucast_pkts    = itr._tx_ucast_pkts;

        counter_track::emplace(nic_index, addendum("RX CNP PKTS"), "packets");
        counter_track::emplace(nic_index, addendum("TX CNP PKTS"), "packets");
        counter_track::emplace(nic_index, addendum("RX UCAST BYTES"), "bytes");
        counter_track::emplace(nic_index, addendum("TX UCAST BYTES"), "bytes");
        counter_track::emplace(nic_index, addendum("RX UCAST PKTS"), "packets");
        counter_track::emplace(nic_index, addendum("TX UCAST PKTS"), "packets");

        size_t track_index = 0;

        TRACE_COUNTER("nic_rx_cnp_pkts", counter_track::at(nic_index, track_index++), _ts,
                      _rx_rdma_cnp_pkts);
        TRACE_COUNTER("nic_tx_cnp_pkts", counter_track::at(nic_index, track_index++), _ts,
                      _tx_rdma_cnp_pkts);
        TRACE_COUNTER("nic_rx_ucast_bytes", counter_track::at(nic_index, track_index++),
                      _ts, _rx_ucast_bytes);
        TRACE_COUNTER("nic_tx_ucast_bytes", counter_track::at(nic_index, track_index++),
                      _ts, _tx_ucast_bytes);
        TRACE_COUNTER("nic_rx_ucast_pkts", counter_track::at(nic_index, track_index++),
                      _ts, _rx_ucast_pkts);
        TRACE_COUNTER("nic_tx_ucast_pkts", counter_track::at(nic_index, track_index++),
                      _ts, _tx_ucast_pkts);
    }
}

void
nic_setup()
{
    // Run update_stats() the first time, to get the names of all existing NICs.
    nic_data::nic_stats_collector.update_stats();

    auto ainic_devices = get_sampling_ainics();

    std::string devices_lowercase = ainic_devices;
    for(auto& itr : devices_lowercase)
        itr = std::tolower(itr);

    if(devices_lowercase == "all")
    {
        // Set nic_vec to all devices.
        nic_data::nic_vec = nic_data::nic_stats_collector.get_nic_list();
    }
    else if(devices_lowercase == "none")
    {
        // Set nic_vec to an empty vector.
        nic_data::nic_vec = {};
    }
    else
    {
        // Get list of devices from the command line and add those that are
        // valid to nic_vec.
        nic_data::nic_vec                        = {};
        auto                            nic_list = tim::delimit(ainic_devices, ",");
        std::unordered_set<std::string> nic_set{};  // For detecting duplicates
        for(const auto& nic : nic_list)
        {
            if(!nic_data::nic_stats_collector.is_nic_valid(nic))
            {
                LOG_WARNING("Invalid NIC: {}", nic);
            }
            else if(nic_set.find(nic) != nic_set.end())
            {
                LOG_WARNING("Repeated NIC: {}", nic);
            }
            else
            {
                nic_data::nic_vec.push_back(nic);
                nic_set.insert(nic);
            }
        }
    }

    for(auto nic_index{ 0u }; nic_index < nic_data::nic_vec.size(); ++nic_index)
    {
        std::string& nic       = nic_data::nic_vec[nic_index];
        auto         cur_agent = agent{ agent_type::NIC,
                                0,
                                nic_index,
                                nic_index,
                                static_cast<int32_t>(nic_index),
                                static_cast<int32_t>(nic_index),
                                nic,
                                nic,
                                "AI NIC",
                                "AI NIC" };
        get_agent_manager_instance().insert_agent(cur_agent);
    }

    nic_data::setup();
}

}  // namespace amd_smi

}  // namespace rocprofsys
