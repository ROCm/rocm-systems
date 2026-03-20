// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include <cstdint>

namespace rocprofsys
{
namespace pmc
{
namespace collectors
{
namespace nic
{

/**
 * @brief Bitfield union for selecting which NIC RDMA metrics to collect.
 *
 * Bit positions (for value access):
 *   - rx_rdma_ucast_bytes = 0   Received unicast bytes
 *   - tx_rdma_ucast_bytes = 1   Transmitted unicast bytes
 *   - rx_rdma_ucast_pkts = 2    Received unicast packets
 *   - tx_rdma_ucast_pkts = 3    Transmitted unicast packets
 *   - rx_rdma_cnp_pkts = 4      Received CNP (congestion) packets
 *   - tx_rdma_cnp_pkts = 5      Transmitted CNP packets
 */
union enabled_metrics
{
    struct
    {
        uint32_t rx_rdma_ucast_bytes : 1;
        uint32_t tx_rdma_ucast_bytes : 1;
        uint32_t rx_rdma_ucast_pkts  : 1;
        uint32_t tx_rdma_ucast_pkts  : 1;
        uint32_t rx_rdma_cnp_pkts    : 1;
        uint32_t tx_rdma_cnp_pkts    : 1;
    } bits;
    uint32_t value = 0;
};

/// All 6 NIC RDMA metrics enabled (bits 0-5)
static constexpr uint32_t ALL_NIC_METRICS = 0x3F;

/**
 * @brief Container for NIC RDMA metrics collected from AMD SMI.
 *
 * These metrics are collected per-port from the NIC and represent
 * cumulative counters for RDMA traffic statistics.
 */
struct metrics
{
    uint64_t rx_rdma_ucast_bytes = 0;  // Received unicast bytes
    uint64_t tx_rdma_ucast_bytes = 0;  // Transmitted unicast bytes
    uint64_t rx_rdma_ucast_pkts  = 0;  // Received unicast packets
    uint64_t tx_rdma_ucast_pkts  = 0;  // Transmitted unicast packets
    uint64_t rx_rdma_cnp_pkts    = 0;  // Received CNP (congestion) packets
    uint64_t tx_rdma_cnp_pkts    = 0;  // Transmitted CNP packets
};

}  // namespace nic
}  // namespace collectors
}  // namespace pmc
}  // namespace rocprofsys
