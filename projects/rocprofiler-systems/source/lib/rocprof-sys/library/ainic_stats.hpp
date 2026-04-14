// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/amd_smi.hpp"

#ifdef AINIC_SUPPORTED
#    include <amd_smi/amdsmi.h>
#endif

struct nic_stats
{
    std::string _name;    // RDMA device name
    std::string _netdev;  // NIC name

    uint64_t _rx_rdma_ucast_bytes{};  // unicast received bytes
    uint64_t _rx_rdma_ucast_pkts{};   // unicast received packets
    uint64_t _tx_rdma_ucast_bytes{};  // unicast transmitted bytes
    uint64_t _tx_rdma_ucast_pkts{};   // unicast transmitted packets

    uint64_t _rx_rdma_cnp_pkts{};  // received CNP packets
    uint64_t _tx_rdma_cnp_pkts{};  // transmitted CNP packets

    std::string to_string() const;

    static constexpr const char* RX_RDMA_UCAST_BYTES = "rx_rdma_ucast_bytes";
    static constexpr const char* RX_RDMA_UCAST_PKTS  = "rx_rdma_ucast_pkts";
    static constexpr const char* TX_RDMA_UCAST_BYTES = "tx_rdma_ucast_bytes";
    static constexpr const char* TX_RDMA_UCAST_PKTS  = "tx_rdma_ucast_pkts";
    static constexpr const char* RX_RDMA_CNP_PKTS    = "rx_rdma_cnp_pkts";
    static constexpr const char* TX_RDMA_CNP_PKTS    = "tx_rdma_cnp_pkts";
};

class ai_nic_stats_collector
{
public:
    using nic_params_t = std::unordered_map<std::string, nic_stats>;

private:
    // _nic_params and _nic_delta_params both hold network stats. _nic_params holds the
    // total values as read on sysfs via amd-smi. _nic_delta_params hold the differences
    // between the latest read and the read before that.
    // e.g. field rx_rdma_cnp_pkts in one instance of nic_stats contains 1100000 and the
    // previous one was 1000000. That means the total number of CNP packets received in
    // the time interval between the two reads was 100000, so the equivalent field
    // rx_rdma_cnp_pkts in the instance of nic_stats pointed to in _nic_delta_params will
    // get the value 100000. The total value are read from amd-smi, but the sampling code
    // in rocprof-sys needs to get the differences between two reads.
    nic_params_t _nic_params;  // Mapping NIC name -> NIC statistics
    nic_params_t _nic_delta_params;

public:
    // Get data associated with the specified NIC in _nic_delta_params.
    // If the data for nic don't exist, set all measure values to 0 (as a protection
    // in case the caller is requesting stats for a nonexistent NIC).
    void get_data(const std::string& nic, nic_stats& data) const;

    // get_nic_list returns the list of NICs on the system.
    [[nodiscard]] std::vector<std::string> get_nic_list() const;

    ai_nic_stats_collector();

    // Update the statistics for all NICs.
    void update_stats();

    // Find nic and fill in the data.
    // If the nic is not found, return false.
    [[nodiscard]] bool find_nic(const std::string& nic, nic_stats& data) const;

    [[nodiscard]] bool is_nic_valid(const std::string& nic) const;

private:
    size_t get_nic_count();

#ifdef AINIC_SUPPORTED
    void update_data_for_one_handle(amdsmi_processor_handle         processor_handle,
                                    amdsmi_nic_rdma_devices_info_t& info);
#endif
};
