// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/categories.hpp"
#include "core/collection/metric_row.hpp"
#include "core/collection/schema.hpp"
#include "library/pmc/collectors/nic/types.hpp"
#include "rocprofiler-systems/categories.h"

#include <fmt/format.h>

#include <array>
#include <cstddef>
#include <string>

namespace rocprofsys::pmc::pollers::catalog::nic_smi
{
using enabled_t = pmc::collectors::nic::enabled_metrics;
using metrics_t = pmc::collectors::nic::metrics;
using row_t     = collection::metric_row<enabled_t, metrics_t>;

/** The NIC RDMA counters; the rocpd tracks keep their ainic_* names. */
inline constexpr auto k_metrics = std::to_array<row_t>({
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.rx_rdma_ucast_bytes != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.rx_rdma_ucast_bytes);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_RX_UCAST_BYTES,
      .label       = "RX RDMA BYTES",
      .unit        = "bytes",
      .symbol      = "NIC RX UCast Bytes",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_rx_rdma_ucast_bytes" },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.tx_rdma_ucast_bytes != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.tx_rdma_ucast_bytes);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_TX_UCAST_BYTES,
      .label       = "TX RDMA BYTES",
      .unit        = "bytes",
      .symbol      = "NIC TX UCast Bytes",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_tx_rdma_ucast_bytes" },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.rx_rdma_ucast_pkts != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.rx_rdma_ucast_pkts); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_RX_UCAST_PKTS,
      .label       = "RX RDMA PACKETS",
      .unit        = "packets",
      .symbol      = "NIC RX UCast PKTS",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_rx_rdma_ucast_pkts" },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.tx_rdma_ucast_pkts != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.tx_rdma_ucast_pkts); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_TX_UCAST_PKTS,
      .label       = "TX RDMA PACKETS",
      .unit        = "packets",
      .symbol      = "NIC TX UCast PKTS",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_tx_rdma_ucast_pkts" },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.rx_rdma_cnp_pkts != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.rx_rdma_cnp_pkts); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_RX_CNP_PKTS,
      .label       = "RX CNP PACKETS",
      .unit        = "packets",
      .symbol      = "NIC RX CNP PKTS",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_rx_rdma_cnp_pkts" },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.tx_rdma_cnp_pkts != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.tx_rdma_cnp_pkts); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_TX_CNP_PKTS,
      .label       = "TX CNP PACKETS",
      .unit        = "packets",
      .symbol      = "NIC TX CNP PKTS",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_tx_rdma_cnp_pkts" },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.tx_rdma_ack_timeout != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.tx_rdma_ack_timeout);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_TX_ACK_TIMEOUT,
      .label       = "TX ACK TIMEOUT",
      .unit        = "timeouts",
      .symbol      = "NIC TX RDMA ACK Timeout",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_tx_rdma_ack_timeout" },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.resp_tx_pkt_seq_err != 0; },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.resp_tx_pkt_seq_err);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_RESP_TX_PKT_SEQ_ERR,
      .label       = "RESP TX PKT SEQ ERR",
      .unit        = "errors",
      .symbol      = "NIC RESP TX PKT SEQ Error",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_resp_tx_pkt_seq_err" },
    { .enabled =
          [](const enabled_t& enabled) { return enabled.bits.req_rx_pkt_seq_err != 0; },
      .read        = [](const metrics_t& reading, std::size_t,
                 std::size_t) { return static_cast<double>(reading.req_rx_pkt_seq_err); },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_REQ_RX_PKT_SEQ_ERR,
      .label       = "REQ RX PKT SEQ ERR",
      .unit        = "errors",
      .symbol      = "NIC REQ RX PKT SEQ Error",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_req_rx_pkt_seq_err" },
    { .enabled =
          [](const enabled_t& enabled) {
              return enabled.bits.req_rx_impl_nak_seq_err != 0;
          },
      .read =
          [](const metrics_t& reading, std::size_t, std::size_t) {
              return static_cast<double>(reading.req_rx_impl_nak_seq_err);
          },
      .category_id = ROCPROFSYS_CATEGORY_AMD_SMI_AINIC_REQ_RX_IMPL_NAK_SEQ_ERR,
      .label       = "REQ RX IMPL NAK SEQ ERR",
      .unit        = "errors",
      .symbol      = "NIC REQ RX Impl NAK SEQ Error",
      .pmc_name    = {},
      .description = {},
      .rocpd_track = "ainic_req_rx_impl_nak_seq_err" },
});

/** The column of `row` on the NIC with AMD SMI index `smi_index`. */
[[nodiscard]] inline collection::column_spec
make_column(const row_t& row, std::size_t smi_index)
{
    const auto        info = category_info(row.category_id);
    const std::string pmc{ row.pmc_name.empty() ? info.name : row.pmc_name };
    return { .symbol      = std::string{ row.symbol },
             .description = std::string{ row.description.empty() ? info.description
                                                                 : row.description },
             .unit        = std::string{ row.unit },
             .pmc_name    = pmc,
             .perfetto_track_name = fmt::format("NIC [{}] {} (S)", smi_index, row.label),
             .rocpd_track_name =
                 row.rocpd_track.empty() ? pmc : std::string{ row.rocpd_track },
             .category_id = row.category_id,
             .block       = {},
             .expression  = {} };
}
}  // namespace rocprofsys::pmc::pollers::catalog::nic_smi
