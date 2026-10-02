// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "broadcom_stats.h"

// clang-format off
const StatTable_t kBroadcomStatTable = {
    // Default (23): PFC frames, ring discards, continuous RoCE pause, FEC
    // blocks, per-CoS RX/TX bytes. Verified present on all 4 ports, b20-31.
    {"rx_pfc_frames", StatTier_t::Default},
    {"tx_pfc_frames", StatTier_t::Default},
    {"rx_total_ring_discards", StatTier_t::Default},
    {"tx_total_ring_discards", StatTier_t::Default},
    {"continuous_roce_pause_events", StatTier_t::Default},
    {"rx_fec_corrected_blocks", StatTier_t::Default},
    {"rx_fec_uncorrectable_blocks", StatTier_t::Default},
    {"rx_bytes_cos0", StatTier_t::Default},
    {"rx_bytes_cos1", StatTier_t::Default},
    {"rx_bytes_cos2", StatTier_t::Default},
    {"rx_bytes_cos3", StatTier_t::Default},
    {"rx_bytes_cos4", StatTier_t::Default},
    {"rx_bytes_cos5", StatTier_t::Default},
    {"rx_bytes_cos6", StatTier_t::Default},
    {"rx_bytes_cos7", StatTier_t::Default},
    {"tx_bytes_cos0", StatTier_t::Default},
    {"tx_bytes_cos1", StatTier_t::Default},
    {"tx_bytes_cos2", StatTier_t::Default},
    {"tx_bytes_cos3", StatTier_t::Default},
    {"tx_bytes_cos4", StatTier_t::Default},
    {"tx_bytes_cos5", StatTier_t::Default},
    {"tx_bytes_cos6", StatTier_t::Default},
    {"tx_bytes_cos7", StatTier_t::Default},

    // Extended: per-priority PFC RX/TX transitions (16)
    {"pfc_pri0_rx_transitions", StatTier_t::Extended},
    {"pfc_pri1_rx_transitions", StatTier_t::Extended},
    {"pfc_pri2_rx_transitions", StatTier_t::Extended},
    {"pfc_pri3_rx_transitions", StatTier_t::Extended},
    {"pfc_pri4_rx_transitions", StatTier_t::Extended},
    {"pfc_pri5_rx_transitions", StatTier_t::Extended},
    {"pfc_pri6_rx_transitions", StatTier_t::Extended},
    {"pfc_pri7_rx_transitions", StatTier_t::Extended},
    {"pfc_pri0_tx_transitions", StatTier_t::Extended},
    {"pfc_pri1_tx_transitions", StatTier_t::Extended},
    {"pfc_pri2_tx_transitions", StatTier_t::Extended},
    {"pfc_pri3_tx_transitions", StatTier_t::Extended},
    {"pfc_pri4_tx_transitions", StatTier_t::Extended},
    {"pfc_pri5_tx_transitions", StatTier_t::Extended},
    {"pfc_pri6_tx_transitions", StatTier_t::Extended},
    {"pfc_pri7_tx_transitions", StatTier_t::Extended},

    // Extended: per-priority TX PFC enabled frames (8). The RX counterpart
    // (rx_pfc_ena_frames_pri0..7, also present on this driver) is deliberately
    // not in this table; the accepted acceptance criteria (55 total) only
    // ever counted the TX side.
    {"tx_pfc_ena_frames_pri0", StatTier_t::Extended},
    {"tx_pfc_ena_frames_pri1", StatTier_t::Extended},
    {"tx_pfc_ena_frames_pri2", StatTier_t::Extended},
    {"tx_pfc_ena_frames_pri3", StatTier_t::Extended},
    {"tx_pfc_ena_frames_pri4", StatTier_t::Extended},
    {"tx_pfc_ena_frames_pri5", StatTier_t::Extended},
    {"tx_pfc_ena_frames_pri6", StatTier_t::Extended},
    {"tx_pfc_ena_frames_pri7", StatTier_t::Extended},

    // Extended: per-priority PFC watchdog storms detected (8). Absent on the
    // in-tree 6.8 bnxt_en driver (verified: not in S_plain.txt on any of the
    // 4 captured ports); present in the table so it surfaces on a driver that
    // exposes it.
    {"pfc_wd_storms_detected_p0", StatTier_t::Extended},
    {"pfc_wd_storms_detected_p1", StatTier_t::Extended},
    {"pfc_wd_storms_detected_p2", StatTier_t::Extended},
    {"pfc_wd_storms_detected_p3", StatTier_t::Extended},
    {"pfc_wd_storms_detected_p4", StatTier_t::Extended},
    {"pfc_wd_storms_detected_p5", StatTier_t::Extended},
    {"pfc_wd_storms_detected_p6", StatTier_t::Extended},
    {"pfc_wd_storms_detected_p7", StatTier_t::Extended},
};
// clang-format on
