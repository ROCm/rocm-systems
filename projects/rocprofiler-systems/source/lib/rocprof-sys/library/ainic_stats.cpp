// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ainic_stats.hpp"

#include <memory>

#include "logger/debug.hpp"

#include <spdlog/fmt/fmt.h>

std::string
nic_stats::to_string() const
{
    return fmt::format("[_name={}, _netdev={}, _rx_rdma_ucast_bytes={}, "
                       "_rx_rdma_ucast_pkts={}, _tx_rdma_ucast_bytes={}, "
                       "_tx_rdma_ucast_pkts={}, _rx_rdma_cnp_pkts={}, "
                       "_tx_rdma_cnp_pkts={}]",
                       _name, _netdev, _rx_rdma_ucast_bytes, _rx_rdma_ucast_pkts,
                       _tx_rdma_ucast_bytes, _tx_rdma_ucast_pkts, _rx_rdma_cnp_pkts,
                       _tx_rdma_cnp_pkts);
}

ai_nic_stats_collector::ai_nic_stats_collector() = default;

bool
ai_nic_stats_collector::find_nic(const std::string& nic, nic_stats& data) const
{
    auto pair = _nic_params.find(nic);
    if(pair == _nic_params.end())
    {
        return false;
    }
    data = pair->second;
    return true;
}

bool
ai_nic_stats_collector::is_nic_valid(const std::string& nic) const
{
    return (_nic_params.find(nic) != _nic_params.end());
}

void
ai_nic_stats_collector::update_stats()
{
#ifdef AINIC_SUPPORTED
    uint32_t                                soc_count{};
    std::unique_ptr<amdsmi_socket_handle[]> sockets;
    // Call amdsmi_get_socket_handles with second parameter (socket_handles)
    // nullptr to get the number of socket handles.
    amdsmi_status_t status = amdsmi_get_socket_handles(&soc_count, nullptr);
    if(status != AMDSMI_STATUS_SUCCESS)
    {
        LOG_ERROR("amdsmi_get_socket_handles failed with status {}", (int) status);
        return;
    }

    if(soc_count == 0)  // Nothing to do.
        return;

    // Allocate a buffer for soc_count socket handles.
    sockets = std::make_unique<amdsmi_socket_handle[]>(soc_count);
    // Get the socket handles.
    status = amdsmi_get_socket_handles(&soc_count, sockets.get());
    if(status != AMDSMI_STATUS_SUCCESS)
    {
        LOG_ERROR("amdsmi_get_socket_handles failed with status {}", (int) status);
        return;
    }

    // Iterate through all socket handles to find all AI NIC processor
    // handles and update the statistics for each of them.
    for(uint32_t index = 0; index < soc_count; index++)
    {
        uint32_t processor_count = 0;
        status                   = amdsmi_get_processor_handles_by_type(
            sockets[index], AMDSMI_PROCESSOR_TYPE_AMD_NIC, nullptr, &processor_count);
        if(status != AMDSMI_STATUS_SUCCESS)
        {
            LOG_ERROR("amdsmi_get_processor_handles_by_type failed with status {}",
                      (int) status);
            return;
        }
        std::vector<amdsmi_processor_handle> processor_handles(processor_count);
        status = amdsmi_get_processor_handles_by_type(
            sockets[index], AMDSMI_PROCESSOR_TYPE_AMD_NIC, processor_handles.data(),
            &processor_count);
        if(status != AMDSMI_STATUS_SUCCESS)
        {
            LOG_ERROR("amdsmi_get_processor_handles_by_type failed with status {}",
                      (int) status);
            return;
        }
        for(uint32_t idx = 0; idx < processor_count; ++idx)
        {
            amdsmi_status_t                status;
            amdsmi_nic_rdma_devices_info_t info;
            status = amdsmi_get_nic_rdma_dev_info(processor_handles[idx], &info);
            if(status != AMDSMI_STATUS_SUCCESS) continue;

            // Update info and stats.
            update_data_for_one_handle(processor_handles[idx], info);
        }
    }
#endif  // AINIC_SUPPORTED
}

size_t
ai_nic_stats_collector::get_nic_count()
{
#ifdef AINIC_SUPPORTED
    uint32_t                                soc_count{};
    std::unique_ptr<amdsmi_socket_handle[]> sockets;
    // Call amdsmi_get_socket_handles with second parameter (socket_handles)
    // nullptr to get the number of socket handles.
    amdsmi_status_t status = amdsmi_get_socket_handles(&soc_count, nullptr);
    if(status != AMDSMI_STATUS_SUCCESS)
    {
        LOG_ERROR("amdsmi_get_socket_handles failed with status {}", (int) status);
        return 0;
    }

    if(soc_count == 0)  // Nothing to do.
        return 0;

    // Allocate a buffer for soc_count socket handles.
    sockets = std::make_unique<amdsmi_socket_handle[]>(soc_count);
    // Get the socket handles.
    status = amdsmi_get_socket_handles(&soc_count, sockets.get());
    if(status != AMDSMI_STATUS_SUCCESS)
    {
        LOG_ERROR("amdsmi_get_socket_handles failed with status {}", (int) status);
        return 0;
    }

    // For all sockets, find all NIC processor handles.
    size_t nic_count{};
    for(uint32_t index = 0; index < soc_count; index++)
    {
        uint32_t processor_count = 0;
        status                   = amdsmi_get_processor_handles_by_type(
            sockets[index], AMDSMI_PROCESSOR_TYPE_AMD_NIC, nullptr, &processor_count);
        if(status != AMDSMI_STATUS_SUCCESS)
        {
            continue;
        }
        nic_count += processor_count;
    }
    return nic_count;
#else
    return 0;
#endif  // AINIC_SUPPORTED
}

#ifdef AINIC_SUPPORTED
void
ai_nic_stats_collector::update_data_for_one_handle(
    amdsmi_processor_handle processor_handle, amdsmi_nic_rdma_devices_info_t& info)
{
    for(uint8_t rdma_dev_idx = 0; rdma_dev_idx < info.num_rdma_dev; ++rdma_dev_idx)
    {
        amdsmi_nic_rdma_dev_info_t dev_info = info.rdma_dev_info[rdma_dev_idx];
        for(uint8_t rdma_port_idx = 0; rdma_port_idx < dev_info.num_rdma_ports;
            ++rdma_port_idx)
        {
            amdsmi_nic_rdma_port_info_t port_info =
                dev_info.rdma_port_info[rdma_port_idx];
            nic_stats data;
            data._name   = dev_info.rdma_dev;
            data._netdev = port_info.netdev;

            std::unique_ptr<amdsmi_nic_stat_t[]> stats;

            // Call *_statistics the first time to get the number of statistics.
            uint32_t        num_stats{};
            amdsmi_status_t status;

            status = amdsmi_get_nic_rdma_port_statistics(processor_handle, rdma_port_idx,
                                                         &num_stats, nullptr);
            if(status != AMDSMI_STATUS_SUCCESS) continue;

            // Allocate stats.
            stats = std::make_unique<amdsmi_nic_stat_t[]>(num_stats);

            // Call *_statistics the second time to get the statistics.
            status = amdsmi_get_nic_rdma_port_statistics(processor_handle, rdma_port_idx,
                                                         &num_stats, stats.get());
            if(status != AMDSMI_STATUS_SUCCESS) continue;

            const std::unordered_map<std::string_view,
                                     std::function<void(nic_stats&, uint64_t)>>
                stat_handlers = {
                    { nic_stats::RX_RDMA_UCAST_BYTES,
                      [](nic_stats& d, uint64_t v) { d._rx_rdma_ucast_bytes = v; } },
                    { nic_stats::RX_RDMA_UCAST_PKTS,
                      [](nic_stats& d, uint64_t v) { d._rx_rdma_ucast_pkts = v; } },
                    { nic_stats::TX_RDMA_UCAST_BYTES,
                      [](nic_stats& d, uint64_t v) { d._tx_rdma_ucast_bytes = v; } },
                    { nic_stats::TX_RDMA_UCAST_PKTS,
                      [](nic_stats& d, uint64_t v) { d._tx_rdma_ucast_pkts = v; } },
                    { nic_stats::RX_RDMA_CNP_PKTS,
                      [](nic_stats& d, uint64_t v) { d._rx_rdma_cnp_pkts = v; } },
                    { nic_stats::TX_RDMA_CNP_PKTS,
                      [](nic_stats& d, uint64_t v) { d._tx_rdma_cnp_pkts = v; } },
                };

            // Retrieve relevant stats.
            for(uint32_t stat_idx{}; stat_idx < num_stats; ++stat_idx)
            {
                if(auto it = stat_handlers.find(stats[stat_idx].name);
                   it != stat_handlers.end())
                {
                    it->second(data, stats[stat_idx].value);
                }
            }

            // We have filled in the fields of data. Now update _nic_params and
            // _nic_delta_params.
            auto it = _nic_params.find(data._netdev);
            if(it == _nic_params.end())  // not found
            {
                nic_stats new_delta;
                new_delta._name   = data._name;
                new_delta._netdev = data._netdev;

                new_delta._rx_rdma_ucast_bytes = 0;
                new_delta._tx_rdma_ucast_bytes = 0;
                new_delta._rx_rdma_ucast_pkts  = 0;
                new_delta._tx_rdma_ucast_pkts  = 0;

                new_delta._rx_rdma_cnp_pkts     = 0;
                new_delta._tx_rdma_cnp_pkts     = 0;
                _nic_params[data._netdev]       = data;
                _nic_delta_params[data._netdev] = new_delta;
            }
            else
            {
                nic_stats  new_delta;
                nic_stats& old_data = it->second;

                new_delta._name   = data._name;
                new_delta._netdev = data._netdev;

                new_delta._rx_rdma_ucast_bytes =
                    data._rx_rdma_ucast_bytes - old_data._rx_rdma_ucast_bytes;
                new_delta._tx_rdma_ucast_bytes =
                    data._tx_rdma_ucast_bytes - old_data._tx_rdma_ucast_bytes;
                new_delta._rx_rdma_ucast_pkts =
                    data._rx_rdma_ucast_pkts - old_data._rx_rdma_ucast_pkts;
                new_delta._tx_rdma_ucast_pkts =
                    data._tx_rdma_ucast_pkts - old_data._tx_rdma_ucast_pkts;

                new_delta._rx_rdma_cnp_pkts =
                    data._rx_rdma_cnp_pkts - old_data._rx_rdma_cnp_pkts;
                new_delta._tx_rdma_cnp_pkts =
                    data._tx_rdma_cnp_pkts - old_data._tx_rdma_cnp_pkts;

                _nic_params[data._netdev]       = data;
                _nic_delta_params[data._netdev] = new_delta;
            }
        }
    }
}
#endif  //  AINIC_SUPPORTED

void
ai_nic_stats_collector::get_data(const std::string& nic, nic_stats& data) const
{
    auto it = _nic_delta_params.find(nic);
    if(it == _nic_delta_params.end())  // not found
    {
        data._netdev              = nic;
        data._name                = "";
        data._rx_rdma_ucast_bytes = 0;
        data._tx_rdma_ucast_bytes = 0;
        data._rx_rdma_ucast_pkts  = 0;
        data._tx_rdma_ucast_pkts  = 0;

        data._rx_rdma_cnp_pkts = 0;
        data._tx_rdma_cnp_pkts = 0;
    }
    else
    {
        data = it->second;
    }
}

std::vector<std::string>
ai_nic_stats_collector::get_nic_list() const
{
    std::vector<std::string> nic_list = {};
    for(auto& it : _nic_params)
    {
        nic_list.push_back(it.first);
    }
    return nic_list;
}
