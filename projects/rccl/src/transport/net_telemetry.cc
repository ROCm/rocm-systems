/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net_telemetry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <pthread.h>

/* Global telemetry state */
int rcclTelemetryEnabled = 0;
RcclTelemetryConfig rcclTelemetryCfg;
RcclDeviceStats rcclTelemetryDevs[RCCL_TELEMETRY_MAX_DEVS];
int rcclTelemetryNumDevs = 0;

/* Internal state */
static int rcclTelemetryInitialized = 0;
static char rcclTelemetryStartTime[64];
static char rcclTelemetryProcessName[256];

/* ---- Periodic HW-counter sampler (time series for congestion) ------ */
/* When RCCL_TELEMETRY_SAMPLE_MS > 0, a background thread samples a small
 * set of congestion-relevant IB-sysfs counters plus the atomic SW byte
 * counters at a fixed interval. Only cheap file reads + atomic loads are
 * done (no ethtool/popen), so the hot path is undisturbed. The absolute
 * per-sample values are emitted as a "hw_samples" time series; rates are
 * computed offline by the trace merger. */
static const char* const rcclTelSampledNames[] = {
  "np_ecn_marked_roce_packets", "np_cnp_sent", "rp_cnp_handled",
  "out_of_buffer", "packet_seq_err", "out_of_sequence",
  "local_ack_timeout_err", "rnr_nak_retry_err"
};
#define RCCL_TEL_NUM_SAMPLED ((int)(sizeof(rcclTelSampledNames) / sizeof(rcclTelSampledNames[0])))
#define RCCL_TEL_MAX_SAMPLES 100000

typedef struct {
  int64_t  ts_us;                       /* absolute CLOCK_MONOTONIC microseconds */
  int      dev_idx;
  uint64_t tx_bytes;                    /* SW cumulative */
  uint64_t rx_bytes;
  int64_t  cong[RCCL_TEL_NUM_SAMPLED];  /* absolute HW counter values, -1 = N/A */
} RcclHwSample;

static RcclHwSample* rcclTelemetrySamples = NULL;
static int           rcclTelemetryNumSamples = 0;
static int           rcclTelemetrySampleIntervalMs = 0;
static pthread_t     rcclTelemetrySamplerThread;
static int           rcclTelemetrySamplerRunning = 0;
static volatile int  rcclTelemetrySamplerStopFlag = 0;

/* ================================================================== */
/* Hardware-agnostic counter model                                     */
/*                                                                     */
/* Each supported HW type owns an independent config block containing: */
/*   - a list of scalar counter descriptors (json_name + source + key) */
/*   - PFC per-priority key format strings                             */
/*   - ethtool byte/packet delta key names                             */
/*                                                                     */
/* Collection and JSON emission iterate over the active HW config and  */
/* are otherwise hardware-agnostic.                                    */
/* ================================================================== */

enum RcclHwcSource {
  HWC_NONE = 0,
  HWC_IB_SYSFS,
  HWC_ETHTOOL,
  HWC_DEBUGFS
};

typedef struct {
  const char*        json_name;    /* key written to the JSON output */
  enum RcclHwcSource source;       /* where to read the counter from */
  const char*        key;          /* primary source-specific identifier */
  const char*        key_fallback; /* optional fallback if primary read is N/A */
} RcclHwCounterDesc;

typedef struct {
  const char* rx_frames_fmt;
  const char* tx_frames_fmt;
  const char* rx_pause_us_fmt;
  const char* tx_pause_us_fmt;
} RcclPfcPatterns;

typedef struct {
  /* Source for the four delta counters. ETHTOOL reads port-wide L2 stats;
   * IB_SYSFS reads the per-port RoCE counters under
   *   /sys/class/infiniband/<dev>/ports/1/hw_counters/.
   * Some drivers only refresh ETHTOOL tx_bytes/rx_bytes every ~1 s,
   * so short brackets see delta=0; IB sysfs values update per WQE. */
  enum RcclHwcSource source;
  const char* tx_bytes;
  const char* rx_bytes;
  const char* tx_packets;
  const char* rx_packets;
} RcclDeltaPatterns;

typedef struct {
  const char*              name;           /* "ainic" */
  const RcclHwCounterDesc* counters;
  int                      num_counters;
  RcclPfcPatterns          pfc;
  RcclDeltaPatterns        delta;
} RcclHwConfig;

/* Helpers for building counter tables without per-row boilerplate. */
#define HWC(json, src, key)            { (json), (src), (key), NULL }
#define HWC_FB(json, src, key, fb)     { (json), (src), (key), (fb) }

/* ------------------------------------------------------------------ */
/* AINIC (AMD / Pensando ionic driver)                                 */
/* ------------------------------------------------------------------ */

static const RcclHwCounterDesc rcclHwcAinic[] = {
  /* Shared / cross-driver counters (canonical json_name, ainic sysfs key) */
  HWC("cnp_rcvd",                    HWC_IB_SYSFS, "rx_rdma_cnp_pkts"),
  HWC("cnp_sent",                    HWC_IB_SYSFS, "tx_rdma_cnp_pkts"),
  HWC("rx_roce_discards",            HWC_IB_SYSFS, "rx_rdma_mtu_discard_pkts"),
  HWC("pfc_rx_pause_frames",         HWC_ETHTOOL,  "frames_rx_pripause"),
  HWC("pfc_tx_pause_frames",         HWC_ETHTOOL,  "frames_tx_pripause"),
  HWC("hw_rx_dropped",               HWC_ETHTOOL,  "hw_rx_dropped"),
  HWC("hw_tx_dropped",               HWC_ETHTOOL,  "hw_tx_dropped"),
  HWC("rx_errors",                   HWC_ETHTOOL,  "hw_rx_over_errors"),
  HWC("to_retransmits",              HWC_IB_SYSFS, "tx_rdma_ack_timeout"),
  HWC("max_retry_exceeded",          HWC_IB_SYSFS, "req_tx_retry_excd_err"),
  HWC("oos_drop_count",              HWC_IB_SYSFS, "resp_rx_outouf_seq"),
  HWC("seq_err_naks_rcvd",           HWC_IB_SYSFS, "req_rx_pkt_seq_err"),

  /* RDMA traffic counters */
  HWC("tx_rdma_retx_pkts",           HWC_IB_SYSFS, "tx_rdma_retx_pkts"),
  HWC("tx_rdma_retx_bytes",          HWC_IB_SYSFS, "tx_rdma_retx_bytes"),
  HWC("tx_rdma_ack_timeout",         HWC_IB_SYSFS, "tx_rdma_ack_timeout"),
  HWC("ecn_marked_pkts",             HWC_IB_SYSFS, "rx_rdma_ecn_pkts"),
  HWC("rx_rdma_mtu_discard_pkts",    HWC_IB_SYSFS, "rx_rdma_mtu_discard_pkts"),

  /* Requester errors (RX path) */
  HWC("req_rx_pkt_seq_err",          HWC_IB_SYSFS, "req_rx_pkt_seq_err"),
  HWC("rnr_retry_err",               HWC_IB_SYSFS, "req_rx_rnr_retry_err"),
  HWC("req_rx_rmt_acc_err",          HWC_IB_SYSFS, "req_rx_rmt_acc_err"),
  HWC("req_rx_cqe_err",              HWC_IB_SYSFS, "req_rx_cqe_err"),
  HWC("req_rx_dup_response",         HWC_IB_SYSFS, "req_rx_dup_response"),

  /* Requester errors (TX path) */
  HWC("req_tx_retry_excd_err",       HWC_IB_SYSFS, "req_tx_retry_excd_err"),
  HWC("req_tx_loc_oper_err",         HWC_IB_SYSFS, "req_tx_loc_oper_err"),

  /* Responder errors (RX path) */
  HWC("resp_rx_dup_request",         HWC_IB_SYSFS, "resp_rx_dup_request"),
  HWC("out_of_buffer",               HWC_IB_SYSFS, "resp_rx_outof_buf"),
  HWC("resp_rx_outouf_seq",          HWC_IB_SYSFS, "resp_rx_outouf_seq"),
  HWC("resp_rx_cqe_err",             HWC_IB_SYSFS, "resp_rx_cqe_err"),

  /* Responder errors (TX path) */
  HWC("resp_tx_rnr_retry_err",       HWC_IB_SYSFS, "resp_tx_rnr_retry_err"),

  /* RDMA traffic — unicast/multicast */
  HWC("tx_rdma_ucast_bytes",         HWC_IB_SYSFS, "tx_rdma_ucast_bytes"),
  HWC("tx_rdma_ucast_pkts",          HWC_IB_SYSFS, "tx_rdma_ucast_pkts"),
  HWC("tx_rdma_mcast_bytes",         HWC_IB_SYSFS, "tx_rdma_mcast_bytes"),
  HWC("tx_rdma_mcast_pkts",          HWC_IB_SYSFS, "tx_rdma_mcast_pkts"),
  HWC("rx_rdma_ucast_bytes",         HWC_IB_SYSFS, "rx_rdma_ucast_bytes"),
  HWC("rx_rdma_ucast_pkts",          HWC_IB_SYSFS, "rx_rdma_ucast_pkts"),
  HWC("rx_rdma_mcast_bytes",         HWC_IB_SYSFS, "rx_rdma_mcast_bytes"),
  HWC("rx_rdma_mcast_pkts",          HWC_IB_SYSFS, "rx_rdma_mcast_pkts"),

  /* CCL/CTS traffic (FW-dependent) */
  HWC("tx_rdma_ccl_cts_bytes",       HWC_IB_SYSFS, "tx_rdma_ccl_cts_bytes"),
  HWC("tx_rdma_ccl_cts_pkts",        HWC_IB_SYSFS, "tx_rdma_ccl_cts_pkts"),
  HWC("tx_rdma_ccl_cts_retx_bytes",  HWC_IB_SYSFS, "tx_rdma_ccl_cts_retx_bytes"),
  HWC("tx_rdma_ccl_cts_retx_pkts",   HWC_IB_SYSFS, "tx_rdma_ccl_cts_retx_pkts"),
  HWC("tx_rdma_ccl_cts_ack_timeout", HWC_IB_SYSFS, "tx_rdma_ccl_cts_ack_timeout"),
  HWC("rx_rdma_ccl_cts_bytes",       HWC_IB_SYSFS, "rx_rdma_ccl_cts_bytes"),
  HWC("rx_rdma_ccl_cts_pkts",        HWC_IB_SYSFS, "rx_rdma_ccl_cts_pkts"),

  /* Requester errors — additional RX */
  HWC("req_rx_rmt_req_err",          HWC_IB_SYSFS, "req_rx_rmt_req_err"),
  HWC("req_rx_oper_err",             HWC_IB_SYSFS, "req_rx_oper_err"),
  HWC("req_rx_impl_nak_seq_err",     HWC_IB_SYSFS, "req_rx_impl_nak_seq_err"),
  HWC("req_rx_cqe_flush",            HWC_IB_SYSFS, "req_rx_cqe_flush"),
  HWC("req_rx_inval_pkts",           HWC_IB_SYSFS, "req_rx_inval_pkts"),

  /* Requester errors — additional TX */
  HWC("req_tx_loc_acc_err",          HWC_IB_SYSFS, "req_tx_loc_acc_err"),
  HWC("req_tx_mem_mgmt_err",         HWC_IB_SYSFS, "req_tx_mem_mgmt_err"),
  HWC("req_tx_loc_sgl_inv_err",      HWC_IB_SYSFS, "req_tx_loc_sgl_inv_err"),

  /* Responder errors — additional RX */
  HWC("resp_rx_cqe_flush",           HWC_IB_SYSFS, "resp_rx_cqe_flush"),
  HWC("resp_rx_loc_len_err",         HWC_IB_SYSFS, "resp_rx_loc_len_err"),
  HWC("resp_rx_inval_request",       HWC_IB_SYSFS, "resp_rx_inval_request"),
  HWC("resp_rx_loc_oper_err",        HWC_IB_SYSFS, "resp_rx_loc_oper_err"),
  HWC("resp_rx_outof_atomic",        HWC_IB_SYSFS, "resp_rx_outof_atomic"),
  HWC("resp_rx_ccl_cts_outouf_seq",  HWC_IB_SYSFS, "resp_rx_ccl_cts_outouf_seq"),
  HWC("resp_rx_s0_table_err",        HWC_IB_SYSFS, "resp_rx_s0_table_err"),

  /* Responder errors — additional TX */
  HWC("resp_tx_pkt_seq_err",         HWC_IB_SYSFS, "resp_tx_pkt_seq_err"),
  HWC("resp_tx_rmt_inval_req_err",   HWC_IB_SYSFS, "resp_tx_rmt_inval_req_err"),
  HWC("resp_tx_rmt_acc_err",         HWC_IB_SYSFS, "resp_tx_rmt_acc_err"),
  HWC("resp_tx_rmt_oper_err",        HWC_IB_SYSFS, "resp_tx_rmt_oper_err"),
  HWC("resp_tx_loc_sgl_inv_err",     HWC_IB_SYSFS, "resp_tx_loc_sgl_inv_err"),
};

static const RcclHwConfig rcclHwConfigAinic = {
  "ainic",
  rcclHwcAinic,
  (int)(sizeof(rcclHwcAinic) / sizeof(rcclHwcAinic[0])),
  { "frames_rx_pri_%d",        "frames_tx_pri_%d",
    "rx_pripause_%d_1us_count", "tx_pripause_%d_1us_count" },
  { HWC_ETHTOOL, "octets_tx_ok", "octets_rx_ok", "frames_tx_ok", "frames_rx_ok" },
};

/* Compile-time check: per-HW counter arrays must fit in RcclDeviceStats::hw_counters */
#ifndef __cplusplus
_Static_assert(sizeof(rcclHwcAinic) / sizeof(rcclHwcAinic[0]) <= RCCL_TELEMETRY_MAX_HWC,
               "AINIC counter table exceeds RCCL_TELEMETRY_MAX_HWC");
#else
static_assert(sizeof(rcclHwcAinic) / sizeof(rcclHwcAinic[0]) <= RCCL_TELEMETRY_MAX_HWC,
              "AINIC counter table exceeds RCCL_TELEMETRY_MAX_HWC");
#endif

/* ------------------------------------------------------------------ */
/* MLX5 (NVIDIA/Mellanox ConnectX, mlx5_core driver)                   */
/* ------------------------------------------------------------------ */
/* RoCE counters exposed under                                         */
/*   /sys/class/infiniband/<dev>/ports/<p>/hw_counters/                */
/* PFC per-priority pause frames/duration come from `ethtool -S`.      */

static const RcclHwCounterDesc rcclHwcMlx5[] = {
  /* --- Canonical cross-driver counters (shared json_name, mlx5 sysfs key) --- */
  /* ECN / congestion notification (the primary congestion signals) */
  HWC("ecn_marked_pkts",            HWC_IB_SYSFS, "np_ecn_marked_roce_packets"),
  HWC("cnp_sent",                   HWC_IB_SYSFS, "np_cnp_sent"),
  HWC("cnp_handled",                HWC_IB_SYSFS, "rp_cnp_handled"),
  HWC("cnp_ignored",                HWC_IB_SYSFS, "rp_cnp_ignored"),
  /* Buffer exhaustion / drops / out-of-sequence / retransmits */
  HWC("out_of_buffer",              HWC_IB_SYSFS, "out_of_buffer"),
  HWC("oos_drop_count",             HWC_IB_SYSFS, "out_of_sequence"),
  HWC("seq_err_naks_rcvd",          HWC_IB_SYSFS, "packet_seq_err"),
  HWC("local_ack_timeout_err",      HWC_IB_SYSFS, "local_ack_timeout_err"),
  HWC("rnr_retry_err",              HWC_IB_SYSFS, "rnr_nak_retry_err"),
  HWC("max_retry_exceeded",         HWC_IB_SYSFS, "req_transport_retries_exceeded"),

  /* --- mlx5-specific counters (canonical mlx5 json_name == sysfs key) --- */
  HWC("roce_slow_restart_cnps",     HWC_IB_SYSFS, "roce_slow_restart_cnps"),
  HWC("implied_nak_seq_err",        HWC_IB_SYSFS, "implied_nak_seq_err"),
  HWC("duplicate_request",          HWC_IB_SYSFS, "duplicate_request"),
  HWC("roce_adp_retrans",           HWC_IB_SYSFS, "roce_adp_retrans"),
  HWC("roce_adp_retrans_to",        HWC_IB_SYSFS, "roce_adp_retrans_to"),
  HWC("roce_slow_restart",          HWC_IB_SYSFS, "roce_slow_restart"),
  HWC("roce_slow_restart_trans",    HWC_IB_SYSFS, "roce_slow_restart_trans"),

  /* Requester errors */
  HWC("req_cqe_error",              HWC_IB_SYSFS, "req_cqe_error"),
  HWC("req_cqe_flush_error",        HWC_IB_SYSFS, "req_cqe_flush_error"),
  HWC("req_remote_access_errors",   HWC_IB_SYSFS, "req_remote_access_errors"),
  HWC("req_remote_invalid_request", HWC_IB_SYSFS, "req_remote_invalid_request"),
  HWC("req_rnr_retries_exceeded",   HWC_IB_SYSFS, "req_rnr_retries_exceeded"),

  /* Responder errors */
  HWC("resp_cqe_error",             HWC_IB_SYSFS, "resp_cqe_error"),
  HWC("resp_cqe_flush_error",       HWC_IB_SYSFS, "resp_cqe_flush_error"),
  HWC("resp_local_length_error",    HWC_IB_SYSFS, "resp_local_length_error"),
  HWC("resp_remote_access_errors",  HWC_IB_SYSFS, "resp_remote_access_errors"),
  HWC("rx_icrc_encapsulated",       HWC_IB_SYSFS, "rx_icrc_encapsulated"),

  /* RDMA request traffic (context for the error rates) */
  HWC("rx_write_requests",          HWC_IB_SYSFS, "rx_write_requests"),
  HWC("rx_read_requests",           HWC_IB_SYSFS, "rx_read_requests"),
  HWC("rx_atomic_requests",         HWC_IB_SYSFS, "rx_atomic_requests"),

  /* Global PFC pause frames (PHY-level, ethtool). Always exposed, unlike the
   * per-priority "rx_prio%d_pause" names which are firmware-dependent and are
   * absent on some ConnectX FW (there only rx_pause_ctrl_phy exists). This is
   * the primary PFC-backpressure congestion signal; the per-priority
   * breakdown, when available, is in the pfc_* arrays. */
  HWC_FB("pfc_rx_pause_frames",     HWC_ETHTOOL, "rx_pause_ctrl_phy", "rx_pause"),
  HWC_FB("pfc_tx_pause_frames",     HWC_ETHTOOL, "tx_pause_ctrl_phy", "tx_pause"),
};

static const RcclHwConfig rcclHwConfigMlx5 = {
  "mlx5",
  rcclHwcMlx5,
  (int)(sizeof(rcclHwcMlx5) / sizeof(rcclHwcMlx5[0])),
  /* PFC per-priority pause frames + pause duration, from ethtool -S. */
  { "rx_prio%d_pause",          "tx_prio%d_pause",
    "rx_prio%d_pause_duration", "tx_prio%d_pause_duration" },
  { HWC_ETHTOOL, "tx_bytes", "rx_bytes", "tx_packets", "rx_packets" },
};

#ifndef __cplusplus
_Static_assert(sizeof(rcclHwcMlx5) / sizeof(rcclHwcMlx5[0]) <= RCCL_TELEMETRY_MAX_HWC,
               "MLX5 counter table exceeds RCCL_TELEMETRY_MAX_HWC");
#else
static_assert(sizeof(rcclHwcMlx5) / sizeof(rcclHwcMlx5[0]) <= RCCL_TELEMETRY_MAX_HWC,
              "MLX5 counter table exceeds RCCL_TELEMETRY_MAX_HWC");
#endif

/* ------------------------------------------------------------------ */
/* THOR2 (Broadcom ConnectX-class NIC, bnxt_re driver)                 */
/* ------------------------------------------------------------------ */
/* RoCE counters exposed under                                         */
/*   /sys/class/infiniband/<dev>/ports/<p>/hw_counters/                */
/* Counter names follow the upstream bnxt_re driver (hw_counters.c).   */
/* Best-effort: names absent on a given firmware read back as N/A (-1) */
/* rather than failing, so the table is safe across bnxt_re versions.  */
/* Delta bytes/packets come from the IB sysfs rx/tx_bytes counters,    */
/* which bnxt_re updates per WQE (ethtool L2 stats refresh too slowly).*/

static const RcclHwCounterDesc rcclHwcThor2[] = {
  /* --- Canonical cross-driver counters (shared json_name, bnxt_re sysfs key) --- */
  /* ECN / congestion notification (primary congestion signals) */
  HWC_FB("ecn_marked_pkts",         HWC_IB_SYSFS, "rx_ecn_marked_pkts", "np_ecn_marked_roce_packets"),
  HWC("cnp_sent",                   HWC_IB_SYSFS, "np_cnp_sent"),
  HWC("cnp_handled",                HWC_IB_SYSFS, "rp_cnp_handled"),
  HWC("cnp_ignored",                HWC_IB_SYSFS, "rp_cnp_ignored"),

  /* Retransmits / timeouts / out-of-sequence (congestion under load) */
  HWC("to_retransmits",             HWC_IB_SYSFS, "to_retransmits"),
  HWC("seq_err_naks_rcvd",          HWC_IB_SYSFS, "seq_err_naks_rcvd"),
  HWC("rnr_retry_err",              HWC_IB_SYSFS, "rnr_naks_rcvd"),
  HWC("max_retry_exceeded",         HWC_IB_SYSFS, "max_retry_exceeded"),
  HWC("local_ack_timeout_err",      HWC_IB_SYSFS, "local_ack_timeout_err"),
  HWC_FB("oos_drop_count",          HWC_IB_SYSFS, "res_oos_drop_count", "oos_drop_count"),
  HWC("dup_req",                    HWC_IB_SYSFS, "dup_req"),
  HWC("missing_resp",               HWC_IB_SYSFS, "missing_resp"),

  /* Error / discard counters */
  HWC("bad_resp_err",               HWC_IB_SYSFS, "bad_resp_err"),
  HWC("unrecoverable_err",          HWC_IB_SYSFS, "unrecoverable_err"),
  HWC("recoverable_errors",         HWC_IB_SYSFS, "recoverable_errors"),
  HWC("rx_errors",                  HWC_IB_SYSFS, "rx_errors"),
  HWC("rx_discards",                HWC_IB_SYSFS, "rx_discards"),
  HWC("tx_errors",                  HWC_IB_SYSFS, "tx_errors"),
  HWC("tx_discards",                HWC_IB_SYSFS, "tx_discards"),
  HWC("rx_roce_error_pkts",         HWC_IB_SYSFS, "rx_roce_error_pkts"),
  HWC("rx_roce_discard_pkts",       HWC_IB_SYSFS, "rx_roce_discard_pkts"),

  /* Responder resource errors (context for the error rates) */
  HWC("res_rx_pci_err",             HWC_IB_SYSFS, "res_rx_pci_err"),
  HWC("res_tx_pci_err",             HWC_IB_SYSFS, "res_tx_pci_err"),
  HWC("res_mem_error",              HWC_IB_SYSFS, "res_mem_error"),
  HWC("res_cq_load_err",            HWC_IB_SYSFS, "res_cq_load_err"),
  HWC("res_srq_load_err",           HWC_IB_SYSFS, "res_srq_load_err"),

  /* RDMA request traffic (context) */
  HWC("rx_write_req",               HWC_IB_SYSFS, "rx_write_req"),
  HWC("rx_read_req",                HWC_IB_SYSFS, "rx_read_req"),
  HWC("rx_atomic_req",              HWC_IB_SYSFS, "rx_atomic_req"),

  /* Global PFC pause frames (ethtool, best-effort bnxt_en names). */
  HWC_FB("pfc_rx_pause_frames",     HWC_ETHTOOL, "rx_pause_frames", "rx_pause_ctrl_phy"),
  HWC_FB("pfc_tx_pause_frames",     HWC_ETHTOOL, "tx_pause_frames", "tx_pause_ctrl_phy"),
};

static const RcclHwConfig rcclHwConfigThor2 = {
  "thor2",
  rcclHwcThor2,
  (int)(sizeof(rcclHwcThor2) / sizeof(rcclHwcThor2[0])),
  /* Broadcom bnxt_en per-priority pause frames via ethtool -S. */
  { "rx_prio%d_pause",  "tx_prio%d_pause",
    NULL,               NULL },
  /* bnxt_re exposes rx/tx_bytes + rx/tx_pkts in IB sysfs hw_counters. */
  { HWC_IB_SYSFS, "tx_bytes", "rx_bytes", "tx_pkts", "rx_pkts" },
};

#ifndef __cplusplus
_Static_assert(sizeof(rcclHwcThor2) / sizeof(rcclHwcThor2[0]) <= RCCL_TELEMETRY_MAX_HWC,
               "THOR2 counter table exceeds RCCL_TELEMETRY_MAX_HWC");
#else
static_assert(sizeof(rcclHwcThor2) / sizeof(rcclHwcThor2[0]) <= RCCL_TELEMETRY_MAX_HWC,
              "THOR2 counter table exceeds RCCL_TELEMETRY_MAX_HWC");
#endif

/* ------------------------------------------------------------------ */
/* Driver name → HW config resolution                                  */
/* ------------------------------------------------------------------ */

static const RcclHwConfig* rcclTelemetryResolveHw(const char* driver_name) {
  if (driver_name == NULL || driver_name[0] == '\0') return NULL;
  if (strcmp(driver_name, "ionic") == 0)               return &rcclHwConfigAinic;
  if (strcmp(driver_name, "mlx5_core") == 0)           return &rcclHwConfigMlx5;
  if (strcmp(driver_name, "bnxt_re") == 0)             return &rcclHwConfigThor2;
  return NULL;
}

/* ------------------------------------------------------------------ */
/* Forward declarations                                               */
/* ------------------------------------------------------------------ */

typedef struct {
  const char* key;
  int         counter_idx;
} RcclDebugfsWanted;

static void rcclTelemetryCollectHwCounters(RcclDeviceStats* dev);
static int64_t rcclTelemetryReadSysfsCounter(const char* path);
static int64_t rcclTelemetryReadHwCounter(const char* roce_device, const char* counter_name);
static void rcclTelemetryGetDriverName(const char* roce_device, char* driver_name, size_t size);
static int rcclTelemetryIsCounterEnabled(const char* counter_name);
static void rcclTelemetryGetTimestamp(char* buf, size_t size);
static void rcclTelemetryWriteJson(FILE* fp);
static void rcclTelemetrySnapshotInit(RcclDeviceStats* dev);
static void rcclTelemetrySamplerStart(void);
static void rcclTelemetrySamplerStop(void);
static void rcclTelemetryCollectDebugfs(int64_t* hwc,
                                         const char* roce_device,
                                         const char* driver_name,
                                         const RcclDebugfsWanted* wanted,
                                         int num_wanted);

/*
 * Read every configured HW counter for `dev` (IB sysfs + batched ethtool +
 * debugfs, plus the four tx/rx byte/packet delta sources) into caller-provided
 * buffers. All outputs are reset to -1 first, so entries that cannot be read
 * stay N/A. Used for both the baseline snapshot (writes snap_init_*) and the
 * current sample (writes the live arrays), which differ only by target buffer.
 */
static void rcclTelemetryReadCounters(RcclDeviceStats* dev, int64_t* hwc,
                                      int64_t* pfc_rx_frames, int64_t* pfc_tx_frames,
                                      int64_t* pfc_rx_pause_us, int64_t* pfc_tx_pause_us,
                                      int64_t* tx_bytes, int64_t* rx_bytes,
                                      int64_t* tx_packets, int64_t* rx_packets);

/* ------------------------------------------------------------------ */
/* Unified batched ethtool reader                                     */
/* ------------------------------------------------------------------ */

#define RCCL_ETHTOOL_MAX_WANTED 80

typedef struct {
  char     key[64];
  int64_t* target;
} RcclEthtoolWantedEx;

static void rcclTelemetryCollectEthtoolBatch(const char* eth_device,
                                              RcclEthtoolWantedEx* wanted,
                                              int num_wanted) {
  if (eth_device[0] == '\0' || num_wanted == 0) {
    return;
  }

  char cmd[256];
  snprintf(cmd, sizeof(cmd), "ethtool -S %s 2>/dev/null", eth_device);

  FILE* fp = popen(cmd, "r");
  if (fp == NULL) {
    return;
  }

  int found = 0;
  char line[256];
  while (fgets(line, sizeof(line), fp) != NULL && found < num_wanted) {
    for (int i = 0; i < num_wanted; i++) {
      if (*(wanted[i].target) >= 0) continue;

      char* p = strstr(line, wanted[i].key);
      if (p == NULL) continue;

      /* Verify word boundary: char before key must be whitespace or start of line */
      if (p > line) {
        char prev = *(p - 1);
        if (prev != ' ' && prev != '\t') continue;
      }
      /* Verify word boundary: char after key must not be alphanumeric or underscore */
      size_t klen = strlen(wanted[i].key);
      char after = *(p + klen);
      if (after != ':' && after != ' ' && after != '\t' &&
          after != '\0' && after != '\n') continue;

      p += klen;
      while (*p == ' ' || *p == ':') p++;
      if (*p != '\0' && *p != '\n') {
        *(wanted[i].target) = strtoll(p, NULL, 10);
        found++;
      }
    }
  }

  pclose(fp);
}

/* ------------------------------------------------------------------ */
/* Init / Flush / Register                                            */
/* ------------------------------------------------------------------ */

void rcclTelemetryInit(void) {
  if (__atomic_exchange_n(&rcclTelemetryInitialized, 1, __ATOMIC_SEQ_CST)) {
    return;
  }

  const char* enable_env = getenv("RCCL_TELEMETRY_ENABLE");
  if (enable_env == NULL || strcmp(enable_env, "1") != 0) {
    rcclTelemetryEnabled = 0;
    return;
  }

  rcclTelemetryEnabled = 1;

  strncpy(rcclTelemetryCfg.output_dir, "/tmp", sizeof(rcclTelemetryCfg.output_dir) - 1);
  rcclTelemetryCfg.output_dir[sizeof(rcclTelemetryCfg.output_dir) - 1] = '\0';
  rcclTelemetryCfg.histogram_max_buckets = 5;
  rcclTelemetryCfg.histogram_bucket_interval_ns = 30000;
  rcclTelemetryCfg.hw_counter_list[0] = '\0';

  const char* env_val;

  env_val = getenv("RCCL_TELEMETRY_OUTPUT_DIR");
  if (env_val != NULL && env_val[0] != '\0') {
    strncpy(rcclTelemetryCfg.output_dir, env_val, sizeof(rcclTelemetryCfg.output_dir) - 1);
    rcclTelemetryCfg.output_dir[sizeof(rcclTelemetryCfg.output_dir) - 1] = '\0';
  }

  env_val = getenv("RCCL_TELEMETRY_HISTOGRAM_BUCKETS");
  if (env_val != NULL && env_val[0] != '\0') {
    int val = atoi(env_val);
    if (val > 0 && val <= RCCL_TELEMETRY_HISTOGRAM_SIZE)
      rcclTelemetryCfg.histogram_max_buckets = val;
  }

  env_val = getenv("RCCL_TELEMETRY_HISTOGRAM_INTERVAL_NS");
  if (env_val != NULL && env_val[0] != '\0') {
    int64_t val = strtoll(env_val, NULL, 10);
    if (val > 0)
      rcclTelemetryCfg.histogram_bucket_interval_ns = val;
  }

  env_val = getenv("RCCL_TELEMETRY_HW_COUNTERS");
  if (env_val != NULL) {
    strncpy(rcclTelemetryCfg.hw_counter_list, env_val, sizeof(rcclTelemetryCfg.hw_counter_list) - 1);
    rcclTelemetryCfg.hw_counter_list[sizeof(rcclTelemetryCfg.hw_counter_list) - 1] = '\0';
  }

  env_val = getenv("RCCL_TELEMETRY_SAMPLE_MS");
  if (env_val != NULL && env_val[0] != '\0') {
    int val = atoi(env_val);
    if (val > 0) rcclTelemetrySampleIntervalMs = val;
  }

  memset(rcclTelemetryDevs, 0, sizeof(rcclTelemetryDevs));
  rcclTelemetryNumDevs = 0;

  for (int i = 0; i < RCCL_TELEMETRY_MAX_DEVS; i++) {
    for (int c = 0; c < RCCL_TELEMETRY_MAX_HWC; c++) {
      rcclTelemetryDevs[i].hw_counters[c] = -1;
      rcclTelemetryDevs[i].snap_init_hw_counters[c] = -1;
    }
    for (int p = 0; p < 8; p++) {
      rcclTelemetryDevs[i].pfc_rx_frames[p] = -1;
      rcclTelemetryDevs[i].pfc_tx_frames[p] = -1;
      rcclTelemetryDevs[i].pfc_rx_pause_us[p] = -1;
      rcclTelemetryDevs[i].pfc_tx_pause_us[p] = -1;
      rcclTelemetryDevs[i].snap_init_pfc_rx_frames[p]   = -1;
      rcclTelemetryDevs[i].snap_init_pfc_tx_frames[p]   = -1;
      rcclTelemetryDevs[i].snap_init_pfc_rx_pause_us[p] = -1;
      rcclTelemetryDevs[i].snap_init_pfc_tx_pause_us[p] = -1;
    }
    rcclTelemetryDevs[i].snap_init_tx_bytes = -1;
    rcclTelemetryDevs[i].snap_init_rx_bytes = -1;
    rcclTelemetryDevs[i].snap_init_tx_packets = -1;
    rcclTelemetryDevs[i].snap_init_rx_packets = -1;
    rcclTelemetryDevs[i].delta_tx_bytes = -1;
    rcclTelemetryDevs[i].delta_rx_bytes = -1;
    rcclTelemetryDevs[i].delta_tx_packets = -1;
    rcclTelemetryDevs[i].delta_rx_packets = -1;
  }

  rcclTelemetryGetTimestamp(rcclTelemetryStartTime, sizeof(rcclTelemetryStartTime));

  rcclTelemetryProcessName[0] = '\0';
  char proc_path[64];
  snprintf(proc_path, sizeof(proc_path), "/proc/%d/comm", (int)getpid());
  FILE* fp = fopen(proc_path, "r");
  if (fp != NULL) {
    if (fgets(rcclTelemetryProcessName, sizeof(rcclTelemetryProcessName), fp) != NULL) {
      size_t len = strlen(rcclTelemetryProcessName);
      if (len > 0 && rcclTelemetryProcessName[len - 1] == '\n')
        rcclTelemetryProcessName[len - 1] = '\0';
    }
    fclose(fp);
  }

  atexit(rcclTelemetryFlush);

  rcclTelemetrySamplerStart();
}

void rcclTelemetryFlush(void) {
  if (!rcclTelemetryEnabled) {
    return;
  }

  static int flushed = 0;
  if (__atomic_exchange_n(&flushed, 1, __ATOMIC_SEQ_CST)) {
    return;
  }

  /* Stop the sampler first so the sample buffer is stable while we write. */
  rcclTelemetrySamplerStop();

  for (int i = 0; i < rcclTelemetryNumDevs; i++) {
    rcclTelemetryCollectHwCounters(&rcclTelemetryDevs[i]);
  }

  if (getenv("RCCL_TELEMETRY_DEBUG") != NULL) {
    fprintf(stderr, "RCCL NET_TELEMETRY: flush pid=%d numDevs=%d\n",
            (int)getpid(), rcclTelemetryNumDevs);
    for (int i = 0; i < rcclTelemetryNumDevs; i++) {
      RcclDeviceStats* d = &rcclTelemetryDevs[i];
      uint64_t wqe_sent = 0, wqe_rcvd = 0, wqe_comp = 0;
      for (int c = 0; c < d->num_channels && c < RCCL_TELEMETRY_MAX_CHANNELS; c++) {
        wqe_sent += d->channels[c].num_wqe_sent;
        wqe_rcvd += d->channels[c].num_wqe_rcvd;
        wqe_comp += d->channels[c].num_wqe_completed;
      }
      fprintf(stderr, "RCCL NET_TELEMETRY:   dev[%d] roce=%s eth=%s chans=%d "
              "tx=%lu rx=%lu wqe_sent=%lu wqe_rcvd=%lu wqe_comp=%lu cq_err=%lu\n",
              i, d->roce_device, d->eth_device, d->num_channels,
              (unsigned long)d->tx_bytes, (unsigned long)d->rx_bytes,
              (unsigned long)wqe_sent, (unsigned long)wqe_rcvd,
              (unsigned long)wqe_comp, (unsigned long)d->num_cq_errors);
    }
  }

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strncpy(hostname, "unknown", sizeof(hostname) - 1);
    hostname[sizeof(hostname) - 1] = '\0';
  }

  char filepath[1024];
  snprintf(filepath, sizeof(filepath), "%s/rccl_telemetry_%s_%d.json",
           rcclTelemetryCfg.output_dir, hostname, (int)getpid());

  FILE* fp = fopen(filepath, "w");
  if (fp == NULL) {
    return;
  }

  rcclTelemetryWriteJson(fp);
  fclose(fp);
}

__attribute__((visibility("default")))
int rcclTelemetrySwCapture(RcclTelemetrySwSnapshot* out, int maxDevs) {
  if (!rcclTelemetryEnabled || out == NULL || maxDevs <= 0) return 0;

  int num_devs = __atomic_load_n(&rcclTelemetryNumDevs, __ATOMIC_ACQUIRE);
  if (num_devs > RCCL_TELEMETRY_MAX_DEVS) num_devs = RCCL_TELEMETRY_MAX_DEVS;
  if (num_devs > maxDevs) num_devs = maxDevs;

  for (int i = 0; i < num_devs; i++) {
    RcclDeviceStats* dev = &rcclTelemetryDevs[i];
    RcclTelemetrySwSnapshot* s = &out[i];

    s->device_id     = dev->device_id;
    s->tx_bytes      = __atomic_load_n(&dev->tx_bytes,      __ATOMIC_RELAXED);
    s->rx_bytes      = __atomic_load_n(&dev->rx_bytes,      __ATOMIC_RELAXED);
    s->num_cq_errors = __atomic_load_n(&dev->num_cq_errors, __ATOMIC_RELAXED);
    s->wqe_sent = s->wqe_rcvd = s->wqe_completed = 0;
    s->wqe_completion_ns_min = 0;
    s->wqe_completion_ns_max = 0;
    for (int b = 0; b < RCCL_TELEMETRY_HISTOGRAM_SIZE; b++)
      s->wqe_completion_histogram[b] = 0;

    int nch = __atomic_load_n(&dev->num_channels, __ATOMIC_RELAXED);
    if (nch > RCCL_TELEMETRY_MAX_CHANNELS) nch = RCCL_TELEMETRY_MAX_CHANNELS;
    for (int c = 0; c < nch; c++) {
      RcclChannelStats* ch = &dev->channels[c];
      int nqp = __atomic_load_n(&ch->num_qps, __ATOMIC_RELAXED);
      if (nqp > RCCL_TELEMETRY_MAX_QPS) nqp = RCCL_TELEMETRY_MAX_QPS;
      for (int q = 0; q < nqp; q++) {
        RcclQpStats* qp = &ch->qp[q];
        s->wqe_sent      += __atomic_load_n(&qp->num_wqe_sent,      __ATOMIC_RELAXED);
        s->wqe_rcvd      += __atomic_load_n(&qp->num_wqe_rcvd,      __ATOMIC_RELAXED);
        s->wqe_completed += __atomic_load_n(&qp->num_wqe_completed, __ATOMIC_RELAXED);

        int64_t qmin = __atomic_load_n(&qp->wqe_completion_ns_min, __ATOMIC_RELAXED);
        int64_t qmax = __atomic_load_n(&qp->wqe_completion_ns_max, __ATOMIC_RELAXED);
        if (qmin > 0 && (s->wqe_completion_ns_min == 0 || qmin < s->wqe_completion_ns_min))
          s->wqe_completion_ns_min = qmin;
        if (qmax > s->wqe_completion_ns_max)
          s->wqe_completion_ns_max = qmax;

        for (int b = 0; b < RCCL_TELEMETRY_HISTOGRAM_SIZE; b++)
          s->wqe_completion_histogram[b] +=
            __atomic_load_n(&qp->wqe_completion_histogram[b], __ATOMIC_RELAXED);
      }
    }
  }

  return num_devs;
}

int rcclTelemetryRegisterDevice(int device_id, const char* roce_device,
                                 const char* eth_device, const char* transport) {
  if (!rcclTelemetryEnabled) {
    return -1;
  }

  if (rcclTelemetryNumDevs >= RCCL_TELEMETRY_MAX_DEVS) {
    return -1;
  }

  int idx = __atomic_fetch_add(&rcclTelemetryNumDevs, 1, __ATOMIC_SEQ_CST);
  if (idx >= RCCL_TELEMETRY_MAX_DEVS) {
    __atomic_fetch_sub(&rcclTelemetryNumDevs, 1, __ATOMIC_SEQ_CST);
    return -1;
  }

  if (getenv("RCCL_TELEMETRY_DEBUG") != NULL) {
    fprintf(stderr, "RCCL NET_TELEMETRY: RegisterDevice idx=%d id=%d roce=%s eth=%s transport=%s\n",
            idx, device_id, roce_device ? roce_device : "(null)",
            eth_device ? eth_device : "(null)", transport ? transport : "(null)");
  }

  RcclDeviceStats* dev = &rcclTelemetryDevs[idx];
  dev->device_id = device_id;

  if (roce_device != NULL) {
    strncpy(dev->roce_device, roce_device, sizeof(dev->roce_device) - 1);
    dev->roce_device[sizeof(dev->roce_device) - 1] = '\0';
  }

  if (eth_device != NULL) {
    strncpy(dev->eth_device, eth_device, sizeof(dev->eth_device) - 1);
    dev->eth_device[sizeof(dev->eth_device) - 1] = '\0';
  }

  if (transport != NULL) {
    strncpy(dev->transport, transport, sizeof(dev->transport) - 1);
    dev->transport[sizeof(dev->transport) - 1] = '\0';
  }

  /* Resolve HW config once at registration time; stays valid for process lifetime. */
  char driver_name[64] = {0};
  rcclTelemetryGetDriverName(dev->roce_device, driver_name, sizeof(driver_name));
  dev->hw_config = rcclTelemetryResolveHw(driver_name);

  rcclTelemetrySnapshotInit(dev);

  return idx;
}

void rcclTelemetryGetEthDevice(const char* roce_device, char* eth_device, size_t eth_device_size) {
  eth_device[0] = '\0';

  if (roce_device == NULL || roce_device[0] == '\0') {
    return;
  }

  char path[512];
  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/device/net", roce_device);

  DIR* dir = opendir(path);
  if (dir == NULL) {
    return;
  }

  struct dirent* entry;
  while ((entry = readdir(dir)) != NULL) {
    if (entry->d_name[0] != '.') {
      strncpy(eth_device, entry->d_name, eth_device_size - 1);
      eth_device[eth_device_size - 1] = '\0';
      break;
    }
  }
  closedir(dir);
}

/* ------------------------------------------------------------------ */
/* Init snapshot for delta counters                                    */
/* ------------------------------------------------------------------ */

static void rcclTelemetrySnapshotInit(RcclDeviceStats* dev) {
  if (dev->eth_device[0] == '\0' || dev->hw_config == NULL) {
    return;
  }

  /* Capture the current absolute values as the baseline. ReadCounters resets
   * every target to -1 first, so counters that fail to read stay N/A and
   * produce a -1 delta at flush rather than a spurious value. */
  rcclTelemetryReadCounters(dev, dev->snap_init_hw_counters,
                            dev->snap_init_pfc_rx_frames, dev->snap_init_pfc_tx_frames,
                            dev->snap_init_pfc_rx_pause_us, dev->snap_init_pfc_tx_pause_us,
                            &dev->snap_init_tx_bytes, &dev->snap_init_rx_bytes,
                            &dev->snap_init_tx_packets, &dev->snap_init_rx_packets);
}

/*
 * Shared HW-counter reader used for both the baseline snapshot and the current
 * sample. Fills the caller's buffers from IB sysfs, a single batched ethtool
 * pass, and debugfs. Every output is reset to -1 up front so unread counters
 * stay N/A (and so the ethtool "skip if already found" dedup works on repeated
 * snapshots instead of seeing a stale delta from a previous flush).
 */
static void rcclTelemetryReadCounters(RcclDeviceStats* dev, int64_t* hwc,
                                      int64_t* pfc_rx_frames, int64_t* pfc_tx_frames,
                                      int64_t* pfc_rx_pause_us, int64_t* pfc_tx_pause_us,
                                      int64_t* tx_bytes, int64_t* rx_bytes,
                                      int64_t* tx_packets, int64_t* rx_packets) {
  if (dev->roce_device[0] == '\0' || dev->hw_config == NULL) return;
  const RcclHwConfig* hw = (const RcclHwConfig*)dev->hw_config;

  for (int c = 0; c < RCCL_TELEMETRY_MAX_HWC; c++) hwc[c] = -1;
  int64_t* pfc_out[4] = { pfc_rx_frames, pfc_tx_frames, pfc_rx_pause_us, pfc_tx_pause_us };
  for (int k = 0; k < 4; k++)
    for (int p = 0; p < 8; p++) pfc_out[k][p] = -1;
  *tx_bytes = *rx_bytes = *tx_packets = *rx_packets = -1;

  /* 1. IB sysfs hw_counters (individual reads, with fallback key). */
  for (int c = 0; c < hw->num_counters; c++) {
    const RcclHwCounterDesc* d = &hw->counters[c];
    if (d->source == HWC_IB_SYSFS && d->key != NULL &&
        rcclTelemetryIsCounterEnabled(d->json_name)) {
      int64_t v = rcclTelemetryReadHwCounter(dev->roce_device, d->key);
      if (v < 0 && d->key_fallback != NULL)
        v = rcclTelemetryReadHwCounter(dev->roce_device, d->key_fallback);
      hwc[c] = v;
    }
  }

  /* 2. Batched ethtool: scalar hw_counters + PFC per-priority + the 4-way
   *    tx/rx bytes/packets sources. Both primary and fallback keys are queued
   *    with the same target; the batch reader skips targets already >= 0. */
  RcclEthtoolWantedEx ew[RCCL_ETHTOOL_MAX_WANTED];
  int ew_n = 0;
  for (int c = 0; c < hw->num_counters; c++) {
    const RcclHwCounterDesc* d = &hw->counters[c];
    if (d->source != HWC_ETHTOOL || d->key == NULL) continue;
    if (!rcclTelemetryIsCounterEnabled(d->json_name)) continue;
    if (ew_n < RCCL_ETHTOOL_MAX_WANTED) {
      strncpy(ew[ew_n].key, d->key, 63); ew[ew_n].key[63] = '\0';
      ew[ew_n].target = &hwc[c]; ew_n++;
    }
    if (d->key_fallback != NULL && ew_n < RCCL_ETHTOOL_MAX_WANTED) {
      strncpy(ew[ew_n].key, d->key_fallback, 63); ew[ew_n].key[63] = '\0';
      ew[ew_n].target = &hwc[c]; ew_n++;
    }
  }

  const RcclPfcPatterns* pfc = &hw->pfc;
  const char* pfc_fmt[4] = { pfc->rx_frames_fmt, pfc->tx_frames_fmt,
                             pfc->rx_pause_us_fmt, pfc->tx_pause_us_fmt };
  for (int pri = 0; pri < 8; pri++) {
    for (int k = 0; k < 4; k++) {
      if (pfc_fmt[k] && ew_n < RCCL_ETHTOOL_MAX_WANTED - 4) {
        snprintf(ew[ew_n].key, 64, pfc_fmt[k], pri);
        ew[ew_n].target = &pfc_out[k][pri]; ew_n++;
      }
    }
  }

  const RcclDeltaPatterns* dp = &hw->delta;
  if (dp->source == HWC_ETHTOOL && ew_n + 4 <= RCCL_ETHTOOL_MAX_WANTED) {
    snprintf(ew[ew_n].key, 64, "%s", dp->tx_bytes);   ew[ew_n].target = tx_bytes;   ew_n++;
    snprintf(ew[ew_n].key, 64, "%s", dp->rx_bytes);   ew[ew_n].target = rx_bytes;   ew_n++;
    snprintf(ew[ew_n].key, 64, "%s", dp->tx_packets); ew[ew_n].target = tx_packets; ew_n++;
    snprintf(ew[ew_n].key, 64, "%s", dp->rx_packets); ew[ew_n].target = rx_packets; ew_n++;
  }

  rcclTelemetryCollectEthtoolBatch(dev->eth_device, ew, ew_n);

  if (dp->source == HWC_IB_SYSFS) {
    *tx_bytes   = rcclTelemetryReadHwCounter(dev->roce_device, dp->tx_bytes);
    *rx_bytes   = rcclTelemetryReadHwCounter(dev->roce_device, dp->rx_bytes);
    *tx_packets = rcclTelemetryReadHwCounter(dev->roce_device, dp->tx_packets);
    *rx_packets = rcclTelemetryReadHwCounter(dev->roce_device, dp->rx_packets);
  }

  /* 3. Debugfs counters (single file read, writes into hwc directly). */
  RcclDebugfsWanted debugfs_list[RCCL_TELEMETRY_MAX_HWC];
  int debugfs_count = 0;
  for (int c = 0; c < hw->num_counters; c++) {
    const RcclHwCounterDesc* d = &hw->counters[c];
    if (d->source == HWC_DEBUGFS && d->key != NULL &&
        rcclTelemetryIsCounterEnabled(d->json_name)) {
      debugfs_list[debugfs_count].key = d->key;
      debugfs_list[debugfs_count].counter_idx = c;
      debugfs_count++;
    }
  }
  if (debugfs_count > 0) {
    char driver_name[64];
    rcclTelemetryGetDriverName(dev->roce_device, driver_name, sizeof(driver_name));
    rcclTelemetryCollectDebugfs(hwc, dev->roce_device, driver_name, debugfs_list, debugfs_count);
  }
}

/* ------------------------------------------------------------------ */
/* Timestamp helper                                                   */
/* ------------------------------------------------------------------ */

static void rcclTelemetryGetTimestamp(char* buf, size_t size) {
  time_t now = time(NULL);
  struct tm* tm_info = localtime(&now);
  strftime(buf, size, "%Y-%m-%d %H:%M:%S", tm_info);
}

/* ------------------------------------------------------------------ */
/* Counter-reading primitives                                         */
/* ------------------------------------------------------------------ */

static int64_t rcclTelemetryReadSysfsCounter(const char* path) {
  int fd = open(path, O_RDONLY);
  if (fd < 0) return -1;

  char buf[64];
  ssize_t n = read(fd, buf, sizeof(buf) - 1);
  close(fd);

  if (n <= 0) return -1;

  buf[n] = '\0';
  return strtoll(buf, NULL, 10);
}

static int rcclTelemetryIsCounterEnabled(const char* counter_name) {
  if (rcclTelemetryCfg.hw_counter_list[0] == '\0') return 1;

  const char* list = rcclTelemetryCfg.hw_counter_list;
  size_t name_len = strlen(counter_name);

  while (*list != '\0') {
    while (*list == ' ' || *list == ',') list++;
    if (strncmp(list, counter_name, name_len) == 0) {
      char next = list[name_len];
      if (next == '\0' || next == ',' || next == ' ') return 1;
    }
    while (*list != '\0' && *list != ',') list++;
  }
  return 0;
}

static void rcclTelemetryGetDriverName(const char* roce_device, char* driver_name, size_t size) {
  driver_name[0] = '\0';

  char link_path[512];
  snprintf(link_path, sizeof(link_path), "/sys/class/infiniband/%s/device/driver", roce_device);

  char resolved[512];
  ssize_t len = readlink(link_path, resolved, sizeof(resolved) - 1);
  if (len < 0) return;
  resolved[len] = '\0';

  char* last_slash = strrchr(resolved, '/');
  if (last_slash != NULL) {
    strncpy(driver_name, last_slash + 1, size - 1);
    driver_name[size - 1] = '\0';
  }
}

static int64_t rcclTelemetryReadHwCounter(const char* roce_device, const char* counter_name) {
  char path[512];
  int64_t val;

  snprintf(path, sizeof(path), "/sys/class/infiniband/%s/hw_counters/%s",
           roce_device, counter_name);
  val = rcclTelemetryReadSysfsCounter(path);
  if (val >= 0) return val;

  for (int port = 1; port <= 2; port++) {
    snprintf(path, sizeof(path), "/sys/class/infiniband/%s/ports/%d/hw_counters/%s",
             roce_device, port, counter_name);
    val = rcclTelemetryReadSysfsCounter(path);
    if (val >= 0) return val;
  }

  return -1;
}

/* ------------------------------------------------------------------ */
/* Periodic HW-counter sampler                                        */
/* ------------------------------------------------------------------ */

/* Resolve a sampled counter name to its index in the device's HW table. */
static int rcclTelemetrySampledTableIdx(const RcclHwConfig* hw, int sampled) {
  for (int c = 0; c < hw->num_counters; c++) {
    if (strcmp(hw->counters[c].json_name, rcclTelSampledNames[sampled]) == 0)
      return c;
  }
  return -1;
}

static void* rcclTelemetrySamplerMain(void* arg) {
  (void)arg;
  /* Per-device cache of resolved table indices (-2 = not yet resolved). */
  int idx_cache[RCCL_TELEMETRY_MAX_DEVS][RCCL_TEL_NUM_SAMPLED];
  for (int i = 0; i < RCCL_TELEMETRY_MAX_DEVS; i++)
    for (int c = 0; c < RCCL_TEL_NUM_SAMPLED; c++) idx_cache[i][c] = -2;

  while (!__atomic_load_n(&rcclTelemetrySamplerStopFlag, __ATOMIC_ACQUIRE)) {
    int64_t ts_us = rcclTelemetryGetNs() / 1000;
    int nd = __atomic_load_n(&rcclTelemetryNumDevs, __ATOMIC_ACQUIRE);
    if (nd > RCCL_TELEMETRY_MAX_DEVS) nd = RCCL_TELEMETRY_MAX_DEVS;

    for (int i = 0; i < nd; i++) {
      RcclDeviceStats* dev = &rcclTelemetryDevs[i];
      if (dev->roce_device[0] == '\0' || dev->hw_config == NULL) continue;
      const RcclHwConfig* hw = (const RcclHwConfig*)dev->hw_config;

      int s = rcclTelemetryNumSamples;
      if (s >= RCCL_TEL_MAX_SAMPLES) return NULL;   /* buffer full: stop sampling */
      RcclHwSample* smp = &rcclTelemetrySamples[s];

      smp->ts_us    = ts_us;
      smp->dev_idx  = i;
      smp->tx_bytes = __atomic_load_n(&dev->tx_bytes, __ATOMIC_RELAXED);
      smp->rx_bytes = __atomic_load_n(&dev->rx_bytes, __ATOMIC_RELAXED);
      for (int c = 0; c < RCCL_TEL_NUM_SAMPLED; c++) {
        if (idx_cache[i][c] == -2) idx_cache[i][c] = rcclTelemetrySampledTableIdx(hw, c);
        int ti = idx_cache[i][c];
        smp->cong[c] = (ti >= 0)
          ? rcclTelemetryReadHwCounter(dev->roce_device, hw->counters[ti].key)
          : -1;
      }
      rcclTelemetryNumSamples = s + 1;
    }

    struct timespec req;
    req.tv_sec  = rcclTelemetrySampleIntervalMs / 1000;
    req.tv_nsec = (long)(rcclTelemetrySampleIntervalMs % 1000) * 1000000L;
    nanosleep(&req, NULL);
  }
  return NULL;
}

static void rcclTelemetrySamplerStart(void) {
  if (rcclTelemetrySampleIntervalMs <= 0) return;
  rcclTelemetrySamples =
    (RcclHwSample*)calloc(RCCL_TEL_MAX_SAMPLES, sizeof(RcclHwSample));
  if (rcclTelemetrySamples == NULL) return;
  rcclTelemetrySamplerStopFlag = 0;
  if (pthread_create(&rcclTelemetrySamplerThread, NULL,
                     rcclTelemetrySamplerMain, NULL) == 0)
    rcclTelemetrySamplerRunning = 1;
}

static void rcclTelemetrySamplerStop(void) {
  if (!rcclTelemetrySamplerRunning) return;
  __atomic_store_n(&rcclTelemetrySamplerStopFlag, 1, __ATOMIC_RELEASE);
  pthread_join(rcclTelemetrySamplerThread, NULL);
  rcclTelemetrySamplerRunning = 0;
}

/* ------------------------------------------------------------------ */
/* Batched debugfs reader                                             */
/* ------------------------------------------------------------------ */

static void rcclTelemetryCollectDebugfs(int64_t* hwc,
                                         const char* roce_device,
                                         const char* driver_name,
                                         const RcclDebugfsWanted* wanted,
                                         int num_wanted) {
  if (driver_name[0] == '\0' || num_wanted == 0) return;

  char path[512];
  snprintf(path, sizeof(path), "/sys/kernel/debug/%s/%s/info",
           driver_name, roce_device);

  FILE* fp = fopen(path, "r");
  if (fp == NULL) return;

  int found = 0;
  char line[256];
  while (fgets(line, sizeof(line), fp) != NULL && found < num_wanted) {
    for (int i = 0; i < num_wanted; i++) {
      if (hwc[wanted[i].counter_idx] >= 0) continue;

      const char* key = wanted[i].key;
      char* p = strstr(line, key);
      if (p != NULL) {
        p += strlen(key);
        while (*p == ' ' || *p == ':' || *p == '=') p++;
        if (*p != '\0') {
          hwc[wanted[i].counter_idx] = strtoll(p, NULL, 10);
          found++;
        }
      }
    }
  }

  fclose(fp);
}

/* ------------------------------------------------------------------ */
/* Main hw-counter collection (HW-agnostic; driven by dev->hw_config)  */
/* ------------------------------------------------------------------ */

static void rcclTelemetryCollectHwCounters(RcclDeviceStats* dev) {
  if (dev->roce_device[0] == '\0' || dev->hw_config == NULL) return;

  const RcclHwConfig* hw = (const RcclHwConfig*)dev->hw_config;

  /* 1-3. Read the current absolute values into the live arrays. */
  int64_t cur_tx_bytes, cur_rx_bytes, cur_tx_packets, cur_rx_packets;
  rcclTelemetryReadCounters(dev, dev->hw_counters,
                            dev->pfc_rx_frames, dev->pfc_tx_frames,
                            dev->pfc_rx_pause_us, dev->pfc_tx_pause_us,
                            &cur_tx_bytes, &cur_rx_bytes,
                            &cur_tx_packets, &cur_rx_packets);

  /* Compute deltas: snap_init < 0 means snapshot was never taken -> delta = -1 */
  dev->delta_tx_bytes   = (dev->snap_init_tx_bytes   >= 0 && cur_tx_bytes   >= 0)
                          ? cur_tx_bytes   - dev->snap_init_tx_bytes   : -1;
  dev->delta_rx_bytes   = (dev->snap_init_rx_bytes   >= 0 && cur_rx_bytes   >= 0)
                          ? cur_rx_bytes   - dev->snap_init_rx_bytes   : -1;
  dev->delta_tx_packets = (dev->snap_init_tx_packets >= 0 && cur_tx_packets >= 0)
                          ? cur_tx_packets - dev->snap_init_tx_packets : -1;
  dev->delta_rx_packets = (dev->snap_init_rx_packets >= 0 && cur_rx_packets >= 0)
                          ? cur_rx_packets - dev->snap_init_rx_packets : -1;

  /* 4. Transform absolute hw_counters/pfc_* values into deltas vs. the
   *    baseline captured by rcclTelemetrySnapshotInit. If either end of the
   *    pair is -1 (counter unavailable / baseline never taken), keep -1. */
  for (int c = 0; c < hw->num_counters; c++) {
    int64_t cur  = dev->hw_counters[c];
    int64_t init = dev->snap_init_hw_counters[c];
    dev->hw_counters[c] = (cur >= 0 && init >= 0) ? (cur - init) : -1;
  }
  for (int p = 0; p < 8; p++) {
#define RCCL_TEL_PFC_DELTA(field, init_field) do {                           \
      int64_t cur  = dev->field[p];                                          \
      int64_t init = dev->init_field[p];                                     \
      dev->field[p] = (cur >= 0 && init >= 0) ? (cur - init) : -1;           \
    } while (0)
    RCCL_TEL_PFC_DELTA(pfc_rx_frames,   snap_init_pfc_rx_frames);
    RCCL_TEL_PFC_DELTA(pfc_tx_frames,   snap_init_pfc_tx_frames);
    RCCL_TEL_PFC_DELTA(pfc_rx_pause_us, snap_init_pfc_rx_pause_us);
    RCCL_TEL_PFC_DELTA(pfc_tx_pause_us, snap_init_pfc_tx_pause_us);
#undef RCCL_TEL_PFC_DELTA
  }
}

/* ------------------------------------------------------------------ */
/* JSON writer                                                        */
/* ------------------------------------------------------------------ */

static void rcclTelemetryWriteJsonArray8(FILE* fp, const char* name, const int64_t arr[8], int trailing_comma) {
  fprintf(fp, "        \"%s\": [%ld, %ld, %ld, %ld, %ld, %ld, %ld, %ld]%s\n",
          name,
          (long)arr[0], (long)arr[1], (long)arr[2], (long)arr[3],
          (long)arr[4], (long)arr[5], (long)arr[6], (long)arr[7],
          trailing_comma ? "," : "");
}

static void rcclTelemetryWriteJson(FILE* fp) {
  char end_time[64];
  rcclTelemetryGetTimestamp(end_time, sizeof(end_time));

  char hostname[256];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strncpy(hostname, "unknown", sizeof(hostname) - 1);
    hostname[sizeof(hostname) - 1] = '\0';
  }

  fprintf(fp, "{\n");
  fprintf(fp, "  \"version\": \"1.0\",\n");
  fprintf(fp, "  \"host_name\": \"%s\",\n", hostname);
  fprintf(fp, "  \"process_name\": \"%s\",\n", rcclTelemetryProcessName);
  fprintf(fp, "  \"process_id\": \"%d\",\n", (int)getpid());
  fprintf(fp, "  \"start_time\": \"%s\",\n", rcclTelemetryStartTime);
  fprintf(fp, "  \"end_time\": \"%s\",\n", end_time);

  const char* transport = "IB-CAST";
  if (rcclTelemetryNumDevs > 0 && rcclTelemetryDevs[0].transport[0] != '\0')
    transport = rcclTelemetryDevs[0].transport;
  fprintf(fp, "  \"transport\": \"%s\",\n", transport);

  fprintf(fp, "  \"devices\": [\n");

  int devsPrinted = 0;
  for (int d = 0; d < rcclTelemetryNumDevs; d++) {
    RcclDeviceStats* dev = &rcclTelemetryDevs[d];

    int activeChannels = 0;
    for (int c = 0; c < dev->num_channels; c++) {
      RcclChannelStats* ch = &dev->channels[c];
      if (ch->num_qps > 0 || ch->num_wqe_sent || ch->num_wqe_rcvd || ch->num_wqe_completed)
        activeChannels++;
    }
    if (dev->tx_bytes == 0 && dev->rx_bytes == 0 && dev->num_cq_errors == 0 && activeChannels == 0)
      continue;

    if (devsPrinted > 0) fprintf(fp, ",\n");
    fprintf(fp, "    {\n");
    fprintf(fp, "      \"device_id\": %d,\n", dev->device_id);
    fprintf(fp, "      \"roce_device\": \"%s\",\n", dev->roce_device);
    fprintf(fp, "      \"eth_device\": \"%s\",\n", dev->eth_device);
    fprintf(fp, "      \"hw_type\": \"%s\",\n",
            dev->hw_config ? ((const RcclHwConfig*)dev->hw_config)->name : "unsupported");
    fprintf(fp, "      \"tx_bytes\": %lu,\n", (unsigned long)dev->tx_bytes);
    fprintf(fp, "      \"rx_bytes\": %lu,\n", (unsigned long)dev->rx_bytes);
    fprintf(fp, "      \"num_cq_errors\": %lu,\n", (unsigned long)dev->num_cq_errors);
    fprintf(fp, "      \"num_channels\": %d,\n", dev->num_channels);
    fprintf(fp, "      \"active_channels\": %d,\n", activeChannels);

    fprintf(fp, "      \"channels\": [\n");
    int chPrinted = 0;
    for (int c = 0; c < dev->num_channels; c++) {
      RcclChannelStats* ch = &dev->channels[c];

      if (ch->num_qps == 0 && ch->num_data_qp == 0 && ch->num_cts_qp == 0)
        continue;

      if (chPrinted > 0) fprintf(fp, ",\n");
      fprintf(fp, "        {\n");
      fprintf(fp, "          \"id\": %d,\n", ch->id);
      fprintf(fp, "          \"num_wqe_sent\": %lu,\n", (unsigned long)ch->num_wqe_sent);
      fprintf(fp, "          \"num_wqe_rcvd\": %lu,\n", (unsigned long)ch->num_wqe_rcvd);
      fprintf(fp, "          \"num_wqe_completed\": %lu,\n", (unsigned long)ch->num_wqe_completed);
      fprintf(fp, "          \"num_cts_sent\": %lu,\n", (unsigned long)ch->num_cts_sent);
      fprintf(fp, "          \"num_data_qp\": %d,\n", ch->num_data_qp);
      fprintf(fp, "          \"num_cts_qp\": %d,\n", ch->num_cts_qp);

      fprintf(fp, "          \"queue_pairs\": [\n");
      for (int q = 0; q < ch->num_qps; q++) {
        RcclQpStats* qp = &ch->qp[q];

        fprintf(fp, "            {\n");
        fprintf(fp, "              \"id\": %d,\n", qp->id);
        fprintf(fp, "              \"num_wqe_sent\": %lu,\n", (unsigned long)qp->num_wqe_sent);
        fprintf(fp, "              \"num_wqe_rcvd\": %lu,\n", (unsigned long)qp->num_wqe_rcvd);
        fprintf(fp, "              \"num_wqe_completed\": %lu,\n", (unsigned long)qp->num_wqe_completed);
        fprintf(fp, "              \"num_slot_miss\": %lu,\n", (unsigned long)qp->num_slot_miss);
        fprintf(fp, "              \"wqe_completion_ns_min\": %ld,\n", (long)qp->wqe_completion_ns_min);
        fprintf(fp, "              \"wqe_completion_ns_max\": %ld,\n", (long)qp->wqe_completion_ns_max);

        fprintf(fp, "              \"wqe_completion_histogram\": [\n");
        int max_buckets = rcclTelemetryCfg.histogram_max_buckets;
        if (max_buckets > RCCL_TELEMETRY_HISTOGRAM_SIZE)
          max_buckets = RCCL_TELEMETRY_HISTOGRAM_SIZE;
        for (int b = 0; b < max_buckets; b++) {
          int64_t latency_ns = (int64_t)(b + 1) * rcclTelemetryCfg.histogram_bucket_interval_ns;
          fprintf(fp, "                {\"latency_ns\": %ld, \"num_wqe\": %lu}%s\n",
                  (long)latency_ns, (unsigned long)qp->wqe_completion_histogram[b],
                  (b < max_buckets - 1) ? "," : "");
        }
        fprintf(fp, "              ]\n");

        fprintf(fp, "            }%s\n", (q < ch->num_qps - 1) ? "," : "");
      }
      fprintf(fp, "          ]\n");

      fprintf(fp, "        }");
      chPrinted++;
    }
    if (chPrinted > 0) fprintf(fp, "\n");
    fprintf(fp, "      ],\n");

    /* Hardware counters — only emit entries defined by the active HW table */
    fprintf(fp, "      \"hw_counters\": {\n");

    const RcclHwConfig* hw = (const RcclHwConfig*)dev->hw_config;
    if (hw != NULL) {
      for (int c = 0; c < hw->num_counters; c++) {
        fprintf(fp, "        \"%s\": %ld,\n",
                hw->counters[c].json_name, (long)dev->hw_counters[c]);
      }

      const RcclPfcPatterns* pfc = &hw->pfc;
      if (pfc->rx_frames_fmt)
        rcclTelemetryWriteJsonArray8(fp, "pfc_rx_frames",   dev->pfc_rx_frames,   1);
      if (pfc->tx_frames_fmt)
        rcclTelemetryWriteJsonArray8(fp, "pfc_tx_frames",   dev->pfc_tx_frames,   1);
      if (pfc->rx_pause_us_fmt)
        rcclTelemetryWriteJsonArray8(fp, "pfc_rx_pause_us", dev->pfc_rx_pause_us, 1);
      if (pfc->tx_pause_us_fmt)
        rcclTelemetryWriteJsonArray8(fp, "pfc_tx_pause_us", dev->pfc_tx_pause_us, 1);
    }

    fprintf(fp, "        \"delta_tx_bytes\": %ld,\n",   (long)dev->delta_tx_bytes);
    fprintf(fp, "        \"delta_rx_bytes\": %ld,\n",   (long)dev->delta_rx_bytes);
    fprintf(fp, "        \"delta_tx_packets\": %ld,\n", (long)dev->delta_tx_packets);
    fprintf(fp, "        \"delta_rx_packets\": %ld\n",  (long)dev->delta_rx_packets);

    fprintf(fp, "      }\n");

    fprintf(fp, "    }");
    devsPrinted++;
  }
  if (devsPrinted > 0) fprintf(fp, "\n");

  fprintf(fp, "  ]");

  /* Periodic HW-counter time series (absolute values; rates computed offline). */
  if (rcclTelemetryNumSamples > 0 && rcclTelemetrySamples != NULL) {
    fprintf(fp, ",\n  \"hw_samples\": [\n");
    for (int i = 0; i < rcclTelemetryNumSamples; i++) {
      RcclHwSample* s = &rcclTelemetrySamples[i];
      RcclDeviceStats* dev = &rcclTelemetryDevs[s->dev_idx];
      fprintf(fp, "    {\"ts_us\": %ld, \"device_id\": %d, \"roce_device\": \"%s\", "
                  "\"tx_bytes\": %lu, \"rx_bytes\": %lu",
              (long)s->ts_us, dev->device_id, dev->roce_device,
              (unsigned long)s->tx_bytes, (unsigned long)s->rx_bytes);
      for (int c = 0; c < RCCL_TEL_NUM_SAMPLED; c++)
        fprintf(fp, ", \"%s\": %ld", rcclTelSampledNames[c], (long)s->cong[c]);
      fprintf(fp, "}%s\n", (i < rcclTelemetryNumSamples - 1) ? "," : "");
    }
    fprintf(fp, "  ]\n");
  } else {
    fprintf(fp, "\n");
  }

  fprintf(fp, "}\n");
}
