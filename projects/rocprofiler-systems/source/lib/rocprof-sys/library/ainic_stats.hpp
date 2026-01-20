#pragma once

#include <cstdint>
#include <map>
#include <string>

#include "core/amd_smi.hpp"
#include <amd_smi/amd_smi_system.h>

struct NICData
{
    std::string _name;          // RDMA device name
    std::string _netdev;        // NIC name
    uint32_t    _num_stats {};  // Number of stats collected for this NIC

    std::uint32_t _rx_rdma_ucast_bytes{};  // unicast received bytes
    std::uint32_t _rx_rdma_ucast_pkts{};   // unicast received packets
    std::uint32_t _tx_rdma_ucast_bytes{};  // unicast transmitted bytes
    std::uint32_t _tx_rdma_ucast_pkts{};   // unicast transmitted packets

    std::uint32_t _rx_rdma_cnp_pkts{};  // received CNP packets
    std::uint32_t _tx_rdma_cnp_pkts{};  // transmitted CNP packets

    std::string to_string() const;

    static const char* RX_RDMA_UCAST_BYTES;
    static const char* RX_RDMA_UCAST_PKTS;
    static const char* TX_RDMA_UCAST_BYTES;
    static const char* TX_RDMA_UCAST_PKTS;

    static const char* RX_RDMA_CNP_PKTS;
    static const char* TX_RDMA_CNP_PKTS;
};

class AINICStatsCollector
{
    using nic_params_t = std::map<std::string, NICData>;

#ifdef USE_AINIC
    amd::smi::AMDSmiSystem&
        _amdsmi;  // Reference to the singleton instance of AMDSmiSystem
#endif

    // _nic_params and _nic_delta_params both hold network stats. _nic_params holds the
    // total values as read on sysfs via amd-smi. _nic_delta_params hold the differences
    // between the latest read and the read before that.
    // e.g. field rx_rdma_cnp_pkts in one instance of NICData contains 1100000 and the
    // previous one was 1000000. That means the total number of CNP packets received in
    // the time interval between the two reads was 100000, so the equivalent field
    // rx_rdma_cnp_pkts in the instance of NICData pointed to in _nic_delta_params will
    // get the value 100000. The total value are read from amd-smi, but the sampling code
    // in rocprof-sys needs to get the differences between two reads.
    nic_params_t _nic_params;  // Mapping NIC name -> NIC statistics
    nic_params_t _nic_delta_params;

public:
    // Get data associated with the specified NIC in _nic_delta_params.
    // If the data for nic don't exist, set all measure values to 0 (as a protection
    // in case the caller is requesting stats for a nonexistent NIC).
    void get_data(const std::string& nic, NICData& data) const;

    AINICStatsCollector();

    // Update the statistics for all NICs.
    void update_stats();

    const nic_params_t& params();

    // Find nic and fill in the data.
    // If the nic is not found, return false.
    bool find_nic(const std::string& nic, NICData& data);

private:
    void update_data_for_one_nic(NICData&);
};
