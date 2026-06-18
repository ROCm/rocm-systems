// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/nic_types.hpp"

#include <cstdint>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace nic
{

// Data types are owned by the backend layer (the producer); re-exported here so
// pmc consumers keep their existing pmc::collectors::nic::* spellings.
namespace backend = ::rocprofsys::backends::amd_smi::nic;

using backend::asic_info;
using backend::port_info;
using backend::rdma_info;
using backend::stat_entry;

/**
 * @brief Bitfield union for selecting which NIC RDMA metrics to collect.
 *
 * Bit positions (for value access):
 *   - rx_rdma_ucast_bytes = 0      Received unicast bytes
 *   - tx_rdma_ucast_bytes = 1      Transmitted unicast bytes
 *   - rx_rdma_ucast_pkts = 2       Received unicast packets
 *   - tx_rdma_ucast_pkts = 3       Transmitted unicast packets
 *   - rx_rdma_cnp_pkts = 4         Received CNP (congestion) packets
 *   - tx_rdma_cnp_pkts = 5         Transmitted CNP packets
 *   - tx_rdma_ack_timeout = 6      Local ACK timeout errors
 *   - resp_tx_pkt_seq_err = 7      Responder has detected a packet sequence error and
 * issued a NAK
 *   - req_rx_pkt_seq_err = 8       Requester has detected a sequence error via a NAK
 *   - req_rx_impl_nak_seq_err = 9  Requester has received an ACK out of sequence
 */
union enabled_metrics
{
    struct
    {
        std::uint32_t rx_rdma_ucast_bytes     : 1;
        std::uint32_t tx_rdma_ucast_bytes     : 1;
        std::uint32_t rx_rdma_ucast_pkts      : 1;
        std::uint32_t tx_rdma_ucast_pkts      : 1;
        std::uint32_t rx_rdma_cnp_pkts        : 1;
        std::uint32_t tx_rdma_cnp_pkts        : 1;
        std::uint32_t tx_rdma_ack_timeout     : 1;
        std::uint32_t resp_tx_pkt_seq_err     : 1;
        std::uint32_t req_rx_pkt_seq_err      : 1;
        std::uint32_t req_rx_impl_nak_seq_err : 1;
    } bits;
    std::uint32_t value = 0;
};

// Bit-mask for all NIC RDMA metrics enabled
inline constexpr std::uint32_t NIC_METRICS_COUNT = 10;
inline constexpr std::uint32_t ALL_NIC_METRICS   = (1 << NIC_METRICS_COUNT) - 1;
/**
 * @brief Container for NIC RDMA metrics.
 *
 * These metrics are collected per-port from the NIC and represent
 * cumulative counters for RDMA traffic statistics.
 */
struct metrics
{
    std::uint64_t rx_rdma_ucast_bytes     = 0;
    std::uint64_t tx_rdma_ucast_bytes     = 0;
    std::uint64_t rx_rdma_ucast_pkts      = 0;
    std::uint64_t tx_rdma_ucast_pkts      = 0;
    std::uint64_t rx_rdma_cnp_pkts        = 0;
    std::uint64_t tx_rdma_cnp_pkts        = 0;
    std::uint64_t tx_rdma_ack_timeout     = 0;
    std::uint64_t resp_tx_pkt_seq_err     = 0;
    std::uint64_t req_rx_pkt_seq_err      = 0;
    std::uint64_t req_rx_impl_nak_seq_err = 0;
};

}  // namespace nic
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
