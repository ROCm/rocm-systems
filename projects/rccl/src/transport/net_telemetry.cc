/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "net_telemetry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <net/if.h>
#include <linux/ethtool.h>
#include <linux/sockios.h>
#include <pthread.h>

/* RCCL_TEL_STATIC_ASSERT comes from net_telemetry.h. */

/* The output path is output_dir plus the file name appended to it, and both
 * bounds come from RCCL_TEL_PATH_MAX, so the join below cannot truncate. */
RCCL_TEL_STATIC_ASSERT(sizeof(((RcclTelemetryConfig*)0)->output_dir) + RCCL_TEL_FILENAME_MAX
                           <= RCCL_TEL_PATH_MAX,
                       "output_dir plus the JSON file name does not fit RCCL_TEL_PATH_MAX");

/* The device record is cache-line aligned and its hot data-path counters each
 * start on their own line, so neighbouring records and the read-mostly metadata
 * never share a line with a per-WQE or per-request write. */
RCCL_TEL_STATIC_ASSERT(alignof(RcclDeviceStats) == RCCL_TELEMETRY_CACHELINE,
                       "RcclDeviceStats must be cache-line aligned");
RCCL_TEL_STATIC_ASSERT(sizeof(RcclDeviceStats) % RCCL_TELEMETRY_CACHELINE == 0,
                       "RcclDeviceStats size must be a whole number of cache lines");
RCCL_TEL_STATIC_ASSERT(offsetof(RcclDeviceStats, wqe_size_histogram) % RCCL_TELEMETRY_CACHELINE == 0,
                       "wqe_size_histogram must start on a cache line");
RCCL_TEL_STATIC_ASSERT(offsetof(RcclDeviceStats, tx_bytes) % RCCL_TELEMETRY_CACHELINE == 0,
                       "tx_bytes must start on a cache line");
RCCL_TEL_STATIC_ASSERT(offsetof(RcclDeviceStats, rx_bytes) % RCCL_TELEMETRY_CACHELINE == 0,
                       "rx_bytes must start on a cache line");
RCCL_TEL_STATIC_ASSERT(offsetof(RcclDeviceStats, tx_bytes) != offsetof(RcclDeviceStats, rx_bytes),
                       "tx_bytes and rx_bytes must not share a cache line");

/* Upper bound on RCCL_TELEMETRY_LATENCY_SAMPLE. Well past the point where the
 * latency work has stopped costing anything; it exists so that the round-up to
 * a power of two cannot be handed an absurd value. */
#define RCCL_TEL_LATENCY_SAMPLE_MAX (1 << 24)

/* Global telemetry state */
int rcclTelemetryEnabled = 0;
RcclTelemetryConfig rcclTelemetryCfg;
RcclDeviceStats rcclTelemetryDevs[RCCL_TELEMETRY_MAX_DEVS];
int rcclTelemetryNumDevs = 0;

/* 1-in-N completion-latency sampling; see net_telemetry.h. The defaults are
 * "sample everything", so an un-configured run behaves and reports exactly as
 * it did before sampling existed. */
uint64_t rcclTelemetryLatencySampleMask = 0;
int rcclTelemetryLatencySampleN = 1;

/* Internal state */
static char rcclTelemetryStartTime[64];
static char rcclTelemetryProcessName[256];

/* ---- On-demand storage for channels and QP slots ------------------ */

/* Serializes allocation only; counter updates deliberately take no lock. */
static pthread_mutex_t rcclTelemetryAllocLock = PTHREAD_MUTEX_INITIALIZER;

/* Serializes the one-shot per-device baseline capture. Held across the counter
 * reads, deliberately: the baseline must be in place before the caller starts
 * driving traffic on the device, and this happens once per device on the
 * connection-setup path, never on the data path. It is a different lock from
 * rcclTelemetryAllocLock so that slow counter I/O cannot block slot allocation
 * for other channels. */
static pthread_mutex_t rcclTelemetrySnapshotLock = PTHREAD_MUTEX_INITIALIZER;

static void rcclTelemetryEnsureSnapshot(int devIdx);

/* Zeroed block of `entries` cache-line-aligned entries.
 *
 * Both RcclChannelStats and RcclQpStats are declared aligned to
 * RCCL_TELEMETRY_CACHELINE, so their sizeof() is a multiple of it and aligning
 * only the block base is enough to keep every entry on its own line(s). Plain
 * calloc() guarantees no more than 16-byte alignment, which would let one line
 * straddle two neighbouring slots and reintroduce false sharing between the
 * threads driving them. */
static void* rcclTelemetryAlignedCalloc(size_t entries, size_t entrySize) {
  size_t bytes = entries * entrySize;
  void* p = NULL;
  if (posix_memalign(&p, RCCL_TELEMETRY_CACHELINE, bytes) != 0 || p == NULL) return NULL;
  memset(p, 0, bytes);
  return p;
}

/* Appends blocks under rcclTelemetryAllocLock; publishes capacity last. */
static int rcclTelemetryBlocksGrow(void** blocks, int* block0Log2, int* capacity, int need, size_t entrySize) {
  if (*capacity == 0) {
    /* Size block 0 to the first request, so sparse shapes stay small. */
    int shift = RCCL_TELEMETRY_BLOCK0_MIN_LOG2;
    while ((1 << shift) < need && shift < 20) shift++;
    *block0Log2 = shift;
  }

  const int base = 1 << *block0Log2;
  int reached = *capacity;
  /* capacity == base * (2^numBlocks - 1) by construction. */
  int numBlocks = 0;
  while (numBlocks < RCCL_TELEMETRY_BLOCKS && base * ((1 << numBlocks) - 1) < reached) numBlocks++;

  while (reached < need && numBlocks < RCCL_TELEMETRY_BLOCKS) {
    int entries = base << numBlocks;
    /* Absurd request: stop rather than wrap the running total and spin. */
    if (entries <= 0 || reached > INT_MAX - entries) break;
    void* block = rcclTelemetryAlignedCalloc((size_t)entries, entrySize);
    if (block == NULL) break;
    blocks[numBlocks] = block;
    numBlocks++;
    reached += entries;
    __atomic_store_n(capacity, reached, __ATOMIC_RELEASE);
  }
  return reached;
}

/* Reserve the block table itself, once per parent. Caller holds the lock. */
static void** rcclTelemetryBlocksAlloc(void) {
  return (void**)calloc(RCCL_TELEMETRY_BLOCKS, sizeof(void*));
}

int rcclTelemetrySetupChannel(int devIdx, int chIdx, int numQps, int* numSlots) {
  if (numSlots) *numSlots = 0;
  if (!rcclTelemetryOn() || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS || numQps <= 0) {
    return -1;
  }

  RcclDeviceStats* dstat = &rcclTelemetryDevs[devIdx];

  /* First channel setup is the first use of this device, and it still precedes
   * any traffic on it, so this is where the HW-counter baseline belongs. Done
   * before the untracked-QP bail-out below so that a device whose slots could
   * not be allocated still reports valid hw_counter deltas. */
  rcclTelemetryEnsureSnapshot(devIdx);

  /* No storage for such an index; charge its QPs to the device. */
  if (chIdx < 0 || chIdx >= RCCL_TELEMETRY_MAX_CHANNELS) {
    __atomic_fetch_add(&dstat->num_qp_untracked, numQps, __ATOMIC_RELAXED);
    return -1;
  }

  pthread_mutex_lock(&rcclTelemetryAllocLock);

  if (dstat->channel_blocks == NULL) dstat->channel_blocks = rcclTelemetryBlocksAlloc();
  RcclChannelStats* ch = NULL;
  if (dstat->channel_blocks != NULL) {
    rcclTelemetryBlocksGrow(dstat->channel_blocks, &dstat->channel_block0_log2, &dstat->channel_capacity, chIdx + 1,
                            sizeof(RcclChannelStats));
    ch = rcclTelemetryChannel(devIdx, chIdx);
  }
  if (ch == NULL) {
    __atomic_fetch_add(&dstat->num_qp_untracked, numQps, __ATOMIC_RELAXED);
    pthread_mutex_unlock(&rcclTelemetryAllocLock);
    return -1;
  }

  ch->id = chIdx;

  int startSlot = ch->num_qps;
  int granted = 0;
  if (ch->qp_blocks != NULL || (ch->qp_blocks = rcclTelemetryBlocksAlloc()) != NULL) {
    int capacity = rcclTelemetryBlocksGrow(ch->qp_blocks, &ch->qp_block0_log2, &ch->qp_capacity, startSlot + numQps,
                                           sizeof(RcclQpStats));
    granted = capacity - startSlot;
    if (granted > numQps) granted = numQps;
    if (granted < 0) granted = 0;
  }

  /* Publish num_qps with a release store only after the slots exist. */
  for (int q = 0; q < granted; q++) {
    rcclTelemetryQpSlot(ch, startSlot + q)->id = startSlot + q;
  }
  __atomic_store_n(&ch->num_qps, startSlot + granted, __ATOMIC_RELEASE);

  if (granted < numQps) {
    /* Allocation failed: report the shortfall instead of dropping it. */
    __atomic_fetch_add(&ch->num_qp_untracked, numQps - granted, __ATOMIC_RELAXED);
    __atomic_fetch_add(&dstat->num_qp_untracked, numQps - granted, __ATOMIC_RELAXED);
  }

  int cur = __atomic_load_n(&dstat->num_channels, __ATOMIC_RELAXED);
  while (chIdx + 1 > cur) {
    if (__atomic_compare_exchange_n(&dstat->num_channels, &cur, chIdx + 1, 1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      break;
  }

  pthread_mutex_unlock(&rcclTelemetryAllocLock);

  if (granted == 0) return -1;
  if (numSlots) *numSlots = granted;
  return startSlot;
}

/* ---- Periodic HW-counter sampler (time series for congestion) ------ */
/* When RCCL_TELEMETRY_SAMPLE_MS > 0, a background thread samples a small
 * set of congestion-relevant IB-sysfs counters plus the atomic SW byte
 * counters at a fixed interval. Only cheap file reads + atomic loads are
 * done (no ethtool ioctl), so the hot path is undisturbed. The absolute
 * per-sample values are emitted as a "hw_samples" time series; rates are
 * computed offline by the trace merger. */

/*
 * One sampled slot. `name` is a canonical json_name, the same vocabulary the
 * per-HW tables below and the hw_counters JSON object use, never a raw sysfs
 * key: rcclTelemetrySampledTableIdx() matches on json_name and the sysfs key it
 * then reads comes from the resolved table row.
 *
 * `alt` is an optional second json_name for a signal the tables spell
 * differently, e.g. CNP-received is cnp_rcvd on ainic and cnp_handled on mlx5
 * and thor2. The series key stays `name` on every NIC, so a consumer gets one
 * stable column per signal instead of a NIC-dependent one.
 */
typedef struct {
  const char* name;
  const char* alt;
} RcclTelSampledDesc;

/* Trailing comment per row: which of the three tables resolves the slot. */
static const RcclTelSampledDesc rcclTelSampled[] = {
  {"ecn_marked_pkts", NULL},                      /* ainic, mlx5, thor2 */
  {"cnp_sent", NULL},                             /* ainic, mlx5, thor2 */
  {"cnp_rcvd", "cnp_handled"},                    /* ainic | mlx5, thor2 */
  {"out_of_buffer", NULL},                        /* ainic, mlx5 */
  {"oos_drop_count", NULL},                       /* ainic, mlx5, thor2 */
  {"seq_err_naks_rcvd", NULL},                    /* ainic, mlx5, thor2 */
  {"rnr_retry_err", NULL},                        /* ainic, mlx5, thor2 */
  {"local_ack_timeout_err", "to_retransmits"},    /* mlx5, thor2 | ainic */
};
#define RCCL_TEL_NUM_SAMPLED ((int)(sizeof(rcclTelSampled) / sizeof(rcclTelSampled[0])))
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

#define RCCL_TEL_HW_TABLE_SIZE(tbl) ((int)(sizeof(tbl) / sizeof((tbl)[0])))

/*
 * Define the config for one HW type from its counter table.
 *
 * A device's counters are stored in RcclDeviceStats::hw_counters[], indexed by
 * position in that table, so a table longer than RCCL_TELEMETRY_MAX_HWC would
 * write past the array. This macro is the only way a table becomes a config,
 * and it derives the counter count and asserts the bound in the same step, so
 * the two cannot drift and a newly added table cannot skip the check. Anything
 * after `hwname` is the rest of the RcclHwConfig initializer, passed through.
 */
#define RCCL_TEL_HW_CONFIG(cfgname, tblname, hwname, ...)                     \
  RCCL_TEL_STATIC_ASSERT(RCCL_TEL_HW_TABLE_SIZE(tblname) <= RCCL_TELEMETRY_MAX_HWC, \
                         hwname " counter table exceeds RCCL_TELEMETRY_MAX_HWC");   \
  static const RcclHwConfig cfgname = {                                       \
    hwname, tblname, RCCL_TEL_HW_TABLE_SIZE(tblname), __VA_ARGS__             \
  }

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

RCCL_TEL_HW_CONFIG(rcclHwConfigAinic, rcclHwcAinic, "ainic",
  { "frames_rx_pri_%d",        "frames_tx_pri_%d",
    "rx_pripause_%d_1us_count", "tx_pripause_%d_1us_count" },
  /* RoCE traffic only: the ethtool frames_*_ok keys are absent on this NIC, and
   * netdev-level byte counts miss RDMA traffic entirely. */
  { HWC_IB_SYSFS, "tx_rdma_ucast_bytes", "rx_rdma_ucast_bytes",
                  "tx_rdma_ucast_pkts",  "rx_rdma_ucast_pkts" });

/* ------------------------------------------------------------------ */
/* MLX5 (NVIDIA/Mellanox ConnectX, mlx5_core driver)                   */
/* ------------------------------------------------------------------ */
/* RoCE counters exposed under                                         */
/*   /sys/class/infiniband/<dev>/ports/<p>/hw_counters/                */
/* PFC per-priority pause frames/duration come from ethtool NIC stats. */

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

RCCL_TEL_HW_CONFIG(rcclHwConfigMlx5, rcclHwcMlx5, "mlx5",
  /* PFC per-priority pause frames + pause duration, from ethtool NIC stats. */
  { "rx_prio%d_pause",          "tx_prio%d_pause",
    "rx_prio%d_pause_duration", "tx_prio%d_pause_duration" },
  /* RoCE bypasses the kernel netdev stack, so ethtool tx_bytes/rx_bytes miss
   * almost all of it. The vport RDMA counters track the IB port counters
   * exactly (port_xmit_data * 4 == tx_vport_rdma_unicast_bytes). */
  { HWC_ETHTOOL, "tx_vport_rdma_unicast_bytes", "rx_vport_rdma_unicast_bytes",
                 "tx_vport_rdma_unicast_packets", "rx_vport_rdma_unicast_packets" });

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

RCCL_TEL_HW_CONFIG(rcclHwConfigThor2, rcclHwcThor2, "thor2",
  /* Broadcom bnxt_en per-priority pause frames via ethtool NIC stats. */
  { "rx_prio%d_pause",  "tx_prio%d_pause",
    NULL,               NULL },
  /* bnxt_re exposes rx/tx_bytes + rx/tx_pkts in IB sysfs hw_counters. */
  { HWC_IB_SYSFS, "tx_bytes", "rx_bytes", "tx_pkts", "rx_pkts" });

/* ------------------------------------------------------------------ */
/* Driver name → HW config resolution                                  */
/* ------------------------------------------------------------------ */

static const RcclHwConfig* rcclTelemetryResolveHw(const char* driver_name) {
  if (driver_name == NULL || driver_name[0] == '\0') return NULL;
  if (strcmp(driver_name, "ionic") == 0)               return &rcclHwConfigAinic;
  if (strcmp(driver_name, "mlx5_core") == 0)           return &rcclHwConfigMlx5;
  if (strcmp(driver_name, "bnxt_re") == 0)             return &rcclHwConfigThor2;
  /* Broadcom: the PCI device is bound to bnxt_en; bnxt_re is an auxiliary
     module, so /sys/class/infiniband/<dev>/device/driver resolves to bnxt_en. */
  if (strcmp(driver_name, "bnxt_en") == 0)             return &rcclHwConfigThor2;
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
static void rcclTelemetryGetEthDevice(const char* roce_device, char* eth_device, size_t eth_device_size);
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

/* Longest `ethtool -S` statistic name we ever ask for. The kernel bounds a
 * statistic name by ETH_GSTRING_LEN (32), and the per-priority names we format
 * stay well inside that, so this has room to spare. */
#define RCCL_TEL_ETHTOOL_KEY_MAX 64

/* The four PFC per-priority patterns (rx/tx frames, rx/tx pause duration) and
 * the four delta keys (tx/rx bytes, tx/rx packets) of an RcclHwConfig. */
#define RCCL_TEL_NUM_PFC_PATTERNS 4
#define RCCL_TEL_NUM_DELTA_KEYS   4

/*
 * Worst case of one batched ethtool query, derived rather than restated: every
 * counter of the widest possible table can contribute its key and its fallback,
 * every PFC pattern is asked for at every priority, and the delta keys are
 * added once. Deriving it means adding counters cannot silently overflow the
 * request and drop the ones past the end.
 */
#define RCCL_ETHTOOL_MAX_WANTED                                     \
  (2 * RCCL_TELEMETRY_MAX_HWC +                                     \
   RCCL_TEL_NUM_PFC_PATTERNS * RCCL_TELEMETRY_NUM_PFC_PRIO +        \
   RCCL_TEL_NUM_DELTA_KEYS)

typedef struct {
  char     key[RCCL_TEL_ETHTOOL_KEY_MAX];
  int64_t* target;
} RcclEthtoolWantedEx;

static void rcclTelemetryCollectEthtoolBatch(const char* eth_device,
                                              RcclEthtoolWantedEx* wanted,
                                              int num_wanted) {
  if (eth_device[0] == '\0' || num_wanted == 0) {
    return;
  }
  /* SIOCETHTOOL identifies the interface by ifr_name, which is IFNAMSIZ wide. */
  if (strlen(eth_device) >= IFNAMSIZ) {
    return;
  }

  int fd = socket(AF_INET, SOCK_DGRAM, 0);
  if (fd < 0) {
    return;
  }

  struct ifreq ifr;
  memset(&ifr, 0, sizeof(ifr));
  /* Length checked above; ifr is zeroed, so the copy stays NUL-terminated. */
  memcpy(ifr.ifr_name, eth_device, strlen(eth_device));

  /* n_stats from ETHTOOL_GDRVINFO is the length of the ETH_SS_STATS set, i.e.
   * the number of `ethtool -S` counters this NIC exposes. */
  struct ethtool_drvinfo drvinfo;
  memset(&drvinfo, 0, sizeof(drvinfo));
  drvinfo.cmd = ETHTOOL_GDRVINFO;
  ifr.ifr_data = (char*)&drvinfo;
  if (ioctl(fd, SIOCETHTOOL, &ifr) != 0) {
    close(fd);
    return;
  }
  unsigned int n_stats = drvinfo.n_stats;
  if (n_stats == 0) {
    close(fd);
    return;
  }

  /* GSTRINGS gives the counter names (ETH_GSTRING_LEN each), GSTATS the values
   * in the same order; index i of one lines up with index i of the other. */
  struct ethtool_gstrings* strings =
      (struct ethtool_gstrings*)calloc(1, sizeof(*strings) +
                                           (size_t)n_stats * ETH_GSTRING_LEN);
  struct ethtool_stats* stats =
      (struct ethtool_stats*)calloc(1, sizeof(*stats) +
                                        (size_t)n_stats * sizeof(uint64_t));
  if (strings == NULL || stats == NULL) {
    free(strings);
    free(stats);
    close(fd);
    return;
  }

  strings->cmd = ETHTOOL_GSTRINGS;
  strings->string_set = ETH_SS_STATS;
  strings->len = n_stats;
  ifr.ifr_data = (char*)strings;
  if (ioctl(fd, SIOCETHTOOL, &ifr) != 0) {
    free(strings);
    free(stats);
    close(fd);
    return;
  }

  stats->cmd = ETHTOOL_GSTATS;
  stats->n_stats = n_stats;
  ifr.ifr_data = (char*)stats;
  if (ioctl(fd, SIOCETHTOOL, &ifr) != 0) {
    free(strings);
    free(stats);
    close(fd);
    return;
  }

  int found = 0;
  for (unsigned int s = 0; s < n_stats && found < num_wanted; s++) {
    /* Kernel names need not be NUL-terminated when they fill the field, so copy
     * into a bounded buffer and terminate before comparing. */
    char name[ETH_GSTRING_LEN + 1];
    memcpy(name, strings->data + (size_t)s * ETH_GSTRING_LEN, ETH_GSTRING_LEN);
    name[ETH_GSTRING_LEN] = '\0';

    for (int i = 0; i < num_wanted; i++) {
      if (*(wanted[i].target) >= 0) continue;
      if (strcmp(name, wanted[i].key) != 0) continue;
      *(wanted[i].target) = (int64_t)stats->data[s];
      found++;
      break;
    }
  }

  free(strings);
  free(stats);
  close(fd);
}

/* ------------------------------------------------------------------ */
/* Init / Flush / Register                                            */
/* ------------------------------------------------------------------ */

/* atexit() takes void(void); the flush status is already logged by the flush
 * itself, and at exit there is no caller left to act on it. */
static void rcclTelemetryFlushAtExit(void) {
  (void)rcclTelemetryFlush();
}

/* The body runs exactly once. A second caller racing in blocks in pthread_once
 * until the winner returns, so it never observes rcclTelemetryEnabled still 0
 * after a successful init. An entry latch did not give that: the loser returned
 * before the flag was published, then went on to win netRefCount++ == 0 and skip
 * the one-time device registration, leaving the run paying for every hook while
 * writing "devices": []. */
static pthread_once_t rcclTelemetryInitOnceControl = PTHREAD_ONCE_INIT;
static int rcclTelemetryInitStatus = 0;

static void rcclTelemetryInitBody(void) {
  const char* enable_env = getenv("RCCL_TELEMETRY_ENABLE");
  if (enable_env == NULL || strcmp(enable_env, "1") != 0) {
    rcclTelemetryEnabled = 0;
    return;
  }

  strncpy(rcclTelemetryCfg.output_dir, "/tmp", sizeof(rcclTelemetryCfg.output_dir) - 1);
  rcclTelemetryCfg.output_dir[sizeof(rcclTelemetryCfg.output_dir) - 1] = '\0';
  rcclTelemetryCfg.histogram_max_buckets = 5;
  rcclTelemetryCfg.histogram_bucket_interval_ns = 30000;
  rcclTelemetryCfg.hw_counter_list[0] = '\0';

  const char* env_val;

  env_val = getenv("RCCL_TELEMETRY_OUTPUT_DIR");
  if (env_val != NULL && env_val[0] != '\0') {
    /* Truncating here would put the JSON in a directory the user did not ask
     * for, so refuse rather than write somewhere unintended. */
    if (strlen(env_val) >= sizeof(rcclTelemetryCfg.output_dir)) {
      fprintf(stderr,
              "RCCL NET_TELEMETRY: RCCL_TELEMETRY_OUTPUT_DIR is longer than %zu bytes; "
              "telemetry stays disabled\n",
              sizeof(rcclTelemetryCfg.output_dir) - 1);
      rcclTelemetryInitStatus = -1;
      return;
    }
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

  /* The interval is final here; derive the reciprocal the completion hook uses
   * in place of a division. */
  rcclTelemetryConfigDeriveHistogram();

  /* 1-in-N completion-latency sampling. The hot path selects a WQE with a
   * mask, so N has to be a power of two. Anything else is rounded up to the
   * next power of two and said out loud, rather than silently behaving as some
   * other interval; the value that ends up in effect is also written to the
   * JSON, so a consumer never has to reconstruct it from the environment. */
  env_val = getenv("RCCL_TELEMETRY_LATENCY_SAMPLE");
  if (env_val != NULL && env_val[0] != '\0') {
    long long requested = strtoll(env_val, NULL, 10);
    if (requested < 1) {
      fprintf(stderr,
              "RCCL NET_TELEMETRY: RCCL_TELEMETRY_LATENCY_SAMPLE=\"%s\" is not a positive integer; "
              "keeping 1 (every WQE sampled)\n",
              env_val);
    } else {
      long long clamped = requested > RCCL_TEL_LATENCY_SAMPLE_MAX ? RCCL_TEL_LATENCY_SAMPLE_MAX : requested;
      long long n = 1;
      while (n < clamped) n <<= 1;
      if (n != requested)
        fprintf(stderr,
                "RCCL NET_TELEMETRY: RCCL_TELEMETRY_LATENCY_SAMPLE=%lld is not a power of two in [1, %d]; "
                "sampling 1 WQE in %lld\n",
                requested, RCCL_TEL_LATENCY_SAMPLE_MAX, n);
      rcclTelemetryLatencySampleN = (int)n;
      rcclTelemetryLatencySampleMask = (uint64_t)n - 1;
    }
  }

  env_val = getenv("RCCL_TELEMETRY_HW_COUNTERS");
  if (env_val != NULL) {
    /* Unlike the output directory, a truncated filter only narrows what is
     * collected, so say so and keep going. */
    if (strlen(env_val) >= sizeof(rcclTelemetryCfg.hw_counter_list)) {
      fprintf(stderr,
              "RCCL NET_TELEMETRY: RCCL_TELEMETRY_HW_COUNTERS is longer than %zu bytes; "
              "the list is truncated and the counters past the cut are not collected\n",
              sizeof(rcclTelemetryCfg.hw_counter_list) - 1);
    }
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
    for (int p = 0; p < RCCL_TELEMETRY_NUM_PFC_PRIO; p++) {
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
  char proc_path[RCCL_TEL_PATH_MAX];
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

  /* Before the sampler, so that a failure here leaves nothing running. Without
   * this handler the run would pay for telemetry and never emit it, which is
   * worse than not collecting at all. */
  if (atexit(rcclTelemetryFlushAtExit) != 0) {
    fprintf(stderr,
            "RCCL NET_TELEMETRY: could not register the exit handler that writes the JSON; "
            "telemetry stays disabled\n");
    rcclTelemetryInitStatus = -1;
    return;
  }

  /* The sampler only needs rcclTelemetryNumDevs, which stays 0 until a device
   * registers, and registration is itself gated on the flag below. */
  rcclTelemetrySamplerStart();

  /* Publish last: every hot-path hook keys off this flag, so it must not be
   * observable as 1 before the table above is seeded. */
  __atomic_store_n(&rcclTelemetryEnabled, 1, __ATOMIC_RELEASE);
  rcclTelemetryInitStatus = 0;
}

int rcclTelemetryInit(void) {
  pthread_once(&rcclTelemetryInitOnceControl, rcclTelemetryInitBody);
  return rcclTelemetryInitStatus;
}

int rcclTelemetryFlush(void) {
  if (!rcclTelemetryOn()) {
    return 0;
  }

  static int flushed = 0;
  if (__atomic_exchange_n(&flushed, 1, __ATOMIC_SEQ_CST)) {
    return 0;
  }

  /* Stop the sampler first so the sample buffer is stable while we write. */
  rcclTelemetrySamplerStop();

  /* Only devices this rank actually used have a baseline to subtract, and
   * reading the others would cost a full ethtool stats read (ioctl) each to
   * produce counters that no baseline makes meaningful. They keep the -1/N/A
   * values installed at registration. */
  for (int i = 0; i < rcclTelemetryNumDevs; i++) {
    if (!__atomic_load_n(&rcclTelemetryDevs[i].snap_taken, __ATOMIC_ACQUIRE)) continue;
    rcclTelemetryCollectHwCounters(&rcclTelemetryDevs[i]);
  }

  if (getenv("RCCL_TELEMETRY_DEBUG") != NULL) {
    fprintf(stderr, "RCCL NET_TELEMETRY: flush pid=%d numDevs=%d\n",
            (int)getpid(), rcclTelemetryNumDevs);
    for (int i = 0; i < rcclTelemetryNumDevs; i++) {
      RcclDeviceStats* d = &rcclTelemetryDevs[i];
      uint64_t wqe_sent = 0, recv_wqe = 0, wqe_rcvd = 0, wqe_comp = 0;
      for (int c = 0; c < d->num_channels; c++) {
        RcclChannelStats* ch = rcclTelemetryChannel(i, c);
        if (ch == NULL) continue;
        RcclChannelAggregate agg;
        rcclTelemetryChannelAggregate(ch, &agg);
        wqe_sent += agg.num_wqe_sent;
        recv_wqe += agg.num_recv_wqe;
        wqe_rcvd += agg.num_wqe_rcvd;
        wqe_comp += agg.num_wqe_completed;
      }
      fprintf(stderr, "RCCL NET_TELEMETRY:   dev[%d] roce=%s eth=%s chans=%d "
              "tx=%lu rx=%lu wqe_sent=%lu recv_wqe=%lu wqe_rcvd=%lu wqe_comp=%lu cq_err=%lu\n",
              i, d->roce_device, d->eth_device, d->num_channels,
              (unsigned long)d->tx_bytes, (unsigned long)d->rx_bytes,
              (unsigned long)wqe_sent, (unsigned long)recv_wqe,
              (unsigned long)wqe_rcvd,
              (unsigned long)wqe_comp, (unsigned long)d->num_cq_errors);
    }
  }

  char hostname[HOST_NAME_MAX + 1];
  if (gethostname(hostname, sizeof(hostname)) != 0) {
    strncpy(hostname, "unknown", sizeof(hostname) - 1);
    hostname[sizeof(hostname) - 1] = '\0';
  }

  /* The uid keeps a shared default dir (e.g. /tmp) per-user: two users never
   * target the same path, so one can never hit the other's file permissions. */
  char filepath[RCCL_TEL_PATH_MAX];
  snprintf(filepath, sizeof(filepath), "%s/rccl_telemetry_%s_%u_%d.json",
           rcclTelemetryCfg.output_dir, hostname, (unsigned)getuid(), (int)getpid());

  FILE* fp = fopen(filepath, "w");
  if (fp == NULL) {
    fprintf(stderr, "RCCL NET_TELEMETRY: cannot write %s (%s); the run's telemetry is lost\n",
            filepath, strerror(errno));
    return -1;
  }
  rcclTelemetryWriteJson(fp);
  fclose(fp);
  return 0;
}

__attribute__((visibility("default")))
int rcclTelemetrySwCapture(RcclTelemetrySwSnapshot* out, int maxDevs) {
  if (!rcclTelemetryOn() || out == NULL || maxDevs <= 0) return 0;

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
    s->wqe_sent = s->recv_wqe = s->wqe_rcvd = s->wqe_completed = 0;
    s->wqe_sampled = 0;
    s->wqe_completion_ns_min = 0;
    s->wqe_completion_ns_max = 0;
    for (int b = 0; b < RCCL_TELEMETRY_HISTOGRAM_SIZE; b++)
      s->wqe_completion_histogram[b] = 0;

    int nch = __atomic_load_n(&dev->num_channels, __ATOMIC_RELAXED);
    for (int c = 0; c < nch; c++) {
      RcclChannelStats* ch = rcclTelemetryChannel(i, c);
      if (ch == NULL) continue;
      int nqp = __atomic_load_n(&ch->num_qps, __ATOMIC_RELAXED);
      for (int q = 0; q < nqp; q++) {
        RcclQpStats* qp = rcclTelemetryQp(ch, q);
        if (qp == NULL) break;
        s->wqe_sent      += __atomic_load_n(&qp->num_wqe_sent,      __ATOMIC_RELAXED);
        s->recv_wqe      += __atomic_load_n(&qp->num_recv_wqe,      __ATOMIC_RELAXED);
        s->wqe_rcvd      += __atomic_load_n(&qp->num_wqe_rcvd,      __ATOMIC_RELAXED);
        s->wqe_completed += __atomic_load_n(&qp->num_wqe_completed, __ATOMIC_RELAXED);

        int64_t qmin = __atomic_load_n(&qp->wqe_completion_ns_min, __ATOMIC_RELAXED);
        int64_t qmax = __atomic_load_n(&qp->wqe_completion_ns_max, __ATOMIC_RELAXED);
        if (qmin > 0 && (s->wqe_completion_ns_min == 0 || qmin < s->wqe_completion_ns_min))
          s->wqe_completion_ns_min = qmin;
        if (qmax > s->wqe_completion_ns_max)
          s->wqe_completion_ns_max = qmax;

        for (int b = 0; b < RCCL_TELEMETRY_HISTOGRAM_SIZE; b++) {
          uint64_t count = __atomic_load_n(&qp->wqe_completion_histogram[b], __ATOMIC_RELAXED);
          s->wqe_completion_histogram[b] += count;
          s->wqe_sampled += count;
        }
      }
    }
  }

  return num_devs;
}

int rcclTelemetryRegisterDevice(int device_id, const char* roce_device, const char* transport) {
  if (!rcclTelemetryOn()) {
    return -1;
  }

  /* The slot is the caller's device index, never an allocation counter: every
   * hot-path entry point addresses rcclTelemetryDevs[] with the transport's own
   * device index, so registration has to use that same numbering or the labels
   * and the counters end up describing different NICs. */
  if (device_id < 0 || device_id >= RCCL_TELEMETRY_MAX_DEVS) {
    /* No slot exists to charge this loss to, so warn once instead. */
    static int warned = 0;
    if (!__atomic_exchange_n(&warned, 1, __ATOMIC_RELAXED)) {
      fprintf(stderr,
              "RCCL NET_TELEMETRY: device %d is outside the %d telemetry device slots, "
              "its counters are not collected\n",
              device_id, RCCL_TELEMETRY_MAX_DEVS);
    }
    return -1;
  }
  int idx = device_id;

  RcclDeviceStats* dev = &rcclTelemetryDevs[idx];
  dev->device_id = device_id;

  if (roce_device != NULL) {
    strncpy(dev->roce_device, roce_device, sizeof(dev->roce_device) - 1);
    dev->roce_device[sizeof(dev->roce_device) - 1] = '\0';
  }

  /* The netdev name comes from the RoCE device, so resolve it here instead of
   * making every caller do the sysfs walk and carry a buffer for the result. */
  rcclTelemetryGetEthDevice(dev->roce_device, dev->eth_device, sizeof(dev->eth_device));

  if (getenv("RCCL_TELEMETRY_DEBUG") != NULL) {
    fprintf(stderr, "RCCL NET_TELEMETRY: RegisterDevice idx=%d id=%d roce=%s eth=%s transport=%s\n",
            idx, device_id, roce_device ? roce_device : "(null)", dev->eth_device,
            transport ? transport : "(null)");
  }

  if (transport != NULL) {
    strncpy(dev->transport, transport, sizeof(dev->transport) - 1);
    dev->transport[sizeof(dev->transport) - 1] = '\0';
  }

  /* Resolve HW config once at registration time; stays valid for process lifetime. */
  char driver_name[RCCL_TELEMETRY_DEV_NAME_MAX] = {0};
  rcclTelemetryGetDriverName(dev->roce_device, driver_name, sizeof(driver_name));
  dev->hw_config = rcclTelemetryResolveHw(driver_name);

  /* No baseline yet — it is captured on first use (see snap_taken). Until then
   * every HW counter and delta reads as -1/N/A rather than as a value that
   * would otherwise be an absolute count masquerading as a delta. */
  for (int c = 0; c < RCCL_TELEMETRY_MAX_HWC; c++) {
    dev->hw_counters[c] = -1;
    dev->snap_init_hw_counters[c] = -1;
  }
  for (int p = 0; p < RCCL_TELEMETRY_NUM_PFC_PRIO; p++) {
    dev->pfc_rx_frames[p] = dev->snap_init_pfc_rx_frames[p] = -1;
    dev->pfc_tx_frames[p] = dev->snap_init_pfc_tx_frames[p] = -1;
    dev->pfc_rx_pause_us[p] = dev->snap_init_pfc_rx_pause_us[p] = -1;
    dev->pfc_tx_pause_us[p] = dev->snap_init_pfc_tx_pause_us[p] = -1;
  }
  dev->snap_init_tx_bytes = dev->snap_init_rx_bytes = -1;
  dev->snap_init_tx_packets = dev->snap_init_rx_packets = -1;
  dev->delta_tx_bytes = dev->delta_rx_bytes = -1;
  dev->delta_tx_packets = dev->delta_rx_packets = -1;

  /* Publish the high-water mark last, so a reader that observes the new count
   * also observes a fully populated slot. */
  int nd = __atomic_load_n(&rcclTelemetryNumDevs, __ATOMIC_RELAXED);
  while (nd < idx + 1 &&
         !__atomic_compare_exchange_n(&rcclTelemetryNumDevs, &nd, idx + 1,
                                      true, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
  }

  return idx;
}

/* Map a RoCE device to its netdev name via sysfs. Internal: the only caller is
 * registration, which is where the pair used to be spelled out at every call
 * site. */
static void rcclTelemetryGetEthDevice(const char* roce_device, char* eth_device, size_t eth_device_size) {
  eth_device[0] = '\0';

  if (roce_device == NULL || roce_device[0] == '\0') {
    return;
  }

  char path[RCCL_TEL_PATH_MAX];
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
 * Capture the baseline for `devIdx` exactly once, on the device's first use.
 *
 * Registration deliberately does not do this: a rank registers every NIC it can
 * enumerate (8 on an MI300X node) but normally drives one or two, and every
 * baseline costs a full ethtool stats read (ioctl) plus IB-sysfs reads.
 * Deferring to first use makes that work proportional to the NICs the rank
 * actually drives.
 */
static void rcclTelemetryEnsureSnapshot(int devIdx) {
  if (devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return;
  RcclDeviceStats* dev = &rcclTelemetryDevs[devIdx];
  if (__atomic_load_n(&dev->snap_taken, __ATOMIC_ACQUIRE)) return;

  pthread_mutex_lock(&rcclTelemetrySnapshotLock);
  /* Re-check under the lock with the same atomic load used on the fast path and
   * the write below, so every access to snap_taken goes through __atomic_*. */
  if (!__atomic_load_n(&dev->snap_taken, __ATOMIC_ACQUIRE)) {
    rcclTelemetrySnapshotInit(dev);
    /* Release store: a flush that sees the flag also sees the baseline. */
    __atomic_store_n(&dev->snap_taken, 1, __ATOMIC_RELEASE);
  }
  pthread_mutex_unlock(&rcclTelemetrySnapshotLock);
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
  int64_t* pfc_out[RCCL_TEL_NUM_PFC_PATTERNS] = { pfc_rx_frames, pfc_tx_frames,
                                                  pfc_rx_pause_us, pfc_tx_pause_us };
  for (int k = 0; k < RCCL_TEL_NUM_PFC_PATTERNS; k++)
    for (int p = 0; p < RCCL_TELEMETRY_NUM_PFC_PRIO; p++) pfc_out[k][p] = -1;
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
  /* Bounded by construction: num_counters is asserted against
   * RCCL_TELEMETRY_MAX_HWC when the table becomes a config, and
   * RCCL_ETHTOOL_MAX_WANTED is the exact worst case of the three loops below
   * given that, so none of them needs a capacity test. */
  RcclEthtoolWantedEx ew[RCCL_ETHTOOL_MAX_WANTED];
  int ew_n = 0;
  for (int c = 0; c < hw->num_counters; c++) {
    const RcclHwCounterDesc* d = &hw->counters[c];
    if (d->source != HWC_ETHTOOL || d->key == NULL) continue;
    if (!rcclTelemetryIsCounterEnabled(d->json_name)) continue;
    snprintf(ew[ew_n].key, sizeof(ew[ew_n].key), "%s", d->key);
    ew[ew_n].target = &hwc[c]; ew_n++;
    if (d->key_fallback != NULL) {
      snprintf(ew[ew_n].key, sizeof(ew[ew_n].key), "%s", d->key_fallback);
      ew[ew_n].target = &hwc[c]; ew_n++;
    }
  }

  const RcclPfcPatterns* pfc = &hw->pfc;
  const char* pfc_fmt[RCCL_TEL_NUM_PFC_PATTERNS] = { pfc->rx_frames_fmt, pfc->tx_frames_fmt,
                                                     pfc->rx_pause_us_fmt, pfc->tx_pause_us_fmt };
  for (int pri = 0; pri < RCCL_TELEMETRY_NUM_PFC_PRIO; pri++) {
    for (int k = 0; k < RCCL_TEL_NUM_PFC_PATTERNS; k++) {
      if (pfc_fmt[k] == NULL) continue;
      snprintf(ew[ew_n].key, sizeof(ew[ew_n].key), pfc_fmt[k], pri);
      ew[ew_n].target = &pfc_out[k][pri]; ew_n++;
    }
  }

  const RcclDeltaPatterns* dp = &hw->delta;
  if (dp->source == HWC_ETHTOOL) {
    const char* delta_key[RCCL_TEL_NUM_DELTA_KEYS] = { dp->tx_bytes, dp->rx_bytes,
                                                       dp->tx_packets, dp->rx_packets };
    int64_t* delta_target[RCCL_TEL_NUM_DELTA_KEYS] = { tx_bytes, rx_bytes, tx_packets, rx_packets };
    for (int k = 0; k < RCCL_TEL_NUM_DELTA_KEYS; k++) {
      snprintf(ew[ew_n].key, sizeof(ew[ew_n].key), "%s", delta_key[k]);
      ew[ew_n].target = delta_target[k]; ew_n++;
    }
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
    char driver_name[RCCL_TELEMETRY_DEV_NAME_MAX];
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

  char link_path[RCCL_TEL_PATH_MAX];
  snprintf(link_path, sizeof(link_path), "/sys/class/infiniband/%s/device/driver", roce_device);

  char resolved[RCCL_TEL_PATH_MAX];
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
  char path[RCCL_TEL_PATH_MAX];
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

/* Resolve a sampled json_name, or its alternate, to a row of the HW table. */
static int rcclTelemetrySampledTableIdx(const RcclHwConfig* hw, int sampled) {
  const RcclTelSampledDesc* s = &rcclTelSampled[sampled];
  for (int c = 0; c < hw->num_counters; c++) {
    if (strcmp(hw->counters[c].json_name, s->name) == 0) return c;
  }
  if (s->alt == NULL) return -1;
  for (int c = 0; c < hw->num_counters; c++) {
    if (strcmp(hw->counters[c].json_name, s->alt) == 0) return c;
  }
  return -1;
}

/* Read one resolved row, honouring the table's fallback sysfs key. */
static int64_t rcclTelemetrySampleRow(const RcclDeviceStats* dev, const RcclHwCounterDesc* d) {
  int64_t v = rcclTelemetryReadHwCounter(dev->roce_device, d->key);
  if (v < 0 && d->key_fallback != NULL) v = rcclTelemetryReadHwCounter(dev->roce_device, d->key_fallback);
  return v;
}

/* Warn once about any sampled name that no configured device's table defines,
 * so a series that is silently all -1 cannot go unnoticed. idx is the sampler's
 * resolution cache; -2 means a device was not evaluated yet and is skipped. */
static void rcclTelemetryWarnUnresolvedSampled(const int idx[][RCCL_TEL_NUM_SAMPLED], int nd, int* warned) {
  for (int c = 0; c < RCCL_TEL_NUM_SAMPLED; c++) {
    if (warned[c]) continue;
    int evaluated = 0, resolved = 0;
    for (int i = 0; i < nd; i++) {
      if (idx[i][c] == -2) continue;
      evaluated++;
      if (idx[i][c] >= 0) resolved++;
    }
    if (evaluated == 0 || resolved > 0) continue;
    warned[c] = 1;
    fprintf(stderr,
            "RCCL NET_TELEMETRY: sampled counter \"%s\" is not defined by any "
            "configured device counter table, its hw_samples series stays -1\n",
            rcclTelSampled[c].name);
  }
}

static void* rcclTelemetrySamplerMain(void* arg) {
  (void)arg;
  /* Per-device cache of resolved table indices (-2 = not yet resolved). */
  int idx_cache[RCCL_TELEMETRY_MAX_DEVS][RCCL_TEL_NUM_SAMPLED];
  for (int i = 0; i < RCCL_TELEMETRY_MAX_DEVS; i++)
    for (int c = 0; c < RCCL_TEL_NUM_SAMPLED; c++) idx_cache[i][c] = -2;
  int warned[RCCL_TEL_NUM_SAMPLED] = {0};

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
        smp->cong[c] = (ti >= 0) ? rcclTelemetrySampleRow(dev, &hw->counters[ti]) : -1;
      }
      rcclTelemetryNumSamples = s + 1;
    }

    rcclTelemetryWarnUnresolvedSampled(idx_cache, nd, warned);

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

  char path[RCCL_TEL_PATH_MAX];
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

/* Replace eight absolute per-priority values with deltas vs. their baseline.
 * A -1 on either side means the counter or the baseline was unavailable, and
 * stays -1 rather than becoming a bogus difference. */
static void rcclTelemetryPfcDelta(int64_t* cur, const int64_t* init) {
  for (int p = 0; p < RCCL_TELEMETRY_NUM_PFC_PRIO; p++)
    cur[p] = (cur[p] >= 0 && init[p] >= 0) ? (cur[p] - init[p]) : -1;
}

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
  rcclTelemetryPfcDelta(dev->pfc_rx_frames, dev->snap_init_pfc_rx_frames);
  rcclTelemetryPfcDelta(dev->pfc_tx_frames, dev->snap_init_pfc_tx_frames);
  rcclTelemetryPfcDelta(dev->pfc_rx_pause_us, dev->snap_init_pfc_rx_pause_us);
  rcclTelemetryPfcDelta(dev->pfc_tx_pause_us, dev->snap_init_pfc_tx_pause_us);
}

/* ------------------------------------------------------------------ */
/* JSON writer                                                        */
/* ------------------------------------------------------------------ */

/* One JSON array per PFC counter, one element per priority. */
static void rcclTelemetryWriteJsonPfcArray(FILE* fp, const char* name,
                                           const int64_t arr[RCCL_TELEMETRY_NUM_PFC_PRIO],
                                           int trailing_comma) {
  fprintf(fp, "        \"%s\": [", name);
  for (int p = 0; p < RCCL_TELEMETRY_NUM_PFC_PRIO; p++)
    fprintf(fp, "%s%ld", p == 0 ? "" : ", ", (long)arr[p]);
  fprintf(fp, "]%s\n", trailing_comma ? "," : "");
}

static void rcclTelemetryWriteJson(FILE* fp) {
  char end_time[64];
  rcclTelemetryGetTimestamp(end_time, sizeof(end_time));

  char hostname[HOST_NAME_MAX + 1];
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

  /* Only when sampling is actually on. At the default N == 1 the histogram
   * covers every completion, num_wqe_sampled would repeat num_wqe_completed,
   * and the file stays byte-for-byte what it was before sampling existed. */
  const int sampleN = rcclTelemetryLatencySampleN;
  const int sampled = sampleN > 1;
  if (sampled) fprintf(fp, "  \"latency_sample_interval\": %d,\n", sampleN);

  fprintf(fp, "  \"devices\": [\n");

  int devsPrinted = 0;
  for (int d = 0; d < rcclTelemetryNumDevs; d++) {
    RcclDeviceStats* dev = &rcclTelemetryDevs[d];

    int activeChannels = 0;
    for (int c = 0; c < dev->num_channels; c++) {
      RcclChannelStats* ch = rcclTelemetryChannel(d, c);
      if (ch == NULL) continue;
      RcclChannelAggregate agg;
      rcclTelemetryChannelAggregate(ch, &agg);
      if (ch->num_qps > 0 || ch->num_qp_untracked > 0 || agg.num_wqe_sent || agg.num_recv_wqe || agg.num_wqe_rcvd ||
          agg.num_wqe_completed)
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
    fprintf(fp, "      \"cq_poll_count\": %lu,\n", (unsigned long)dev->cq_poll_count);
    if (dev->channels_unknown)
      fprintf(fp, "      \"num_channels\": null,\n");
    else
      fprintf(fp, "      \"num_channels\": %d,\n", dev->num_channels);
    fprintf(fp, "      \"active_channels\": %d,\n", activeChannels);
    fprintf(fp, "      \"num_qp_untracked\": %d,\n", dev->num_qp_untracked);

    /* WQE payload-size distribution; only populated buckets are emitted. */
    fprintf(fp, "      \"wqe_size_stats\": [");
    int sizesPrinted = 0;
    for (int b = 0; b < RCCL_TELEMETRY_WQE_SIZE_BUCKETS; b++) {
      if (dev->wqe_size_histogram[b] == 0) continue;
      /* Bucket b covers [2^(b-1), 2^b - 1]; bucket 0 is zero-length WQEs. */
      unsigned long long maxBytes = (b == 0) ? 0ULL : ((1ULL << b) - 1ULL);
      fprintf(fp, "%s\n        {\"max_wqe_size\": %llu, \"num_wqe\": %lu}",
              sizesPrinted ? "," : "", maxBytes, (unsigned long)dev->wqe_size_histogram[b]);
      sizesPrinted++;
    }
    fprintf(fp, "%s],\n", sizesPrinted ? "\n      " : "");

    fprintf(fp, "      \"channels\": [\n");
    int chPrinted = 0;
    for (int c = 0; c < dev->num_channels; c++) {
      RcclChannelStats* ch = rcclTelemetryChannel(d, c);
      if (ch == NULL) continue;

      if (ch->num_qps == 0 && ch->num_data_qp == 0 && ch->num_cts_qp == 0 && ch->num_qp_untracked == 0) continue;

      /* Channel WQE/CTS totals are sums over this channel's QP slots, computed
       * here rather than maintained on the hot path. */
      RcclChannelAggregate agg;
      rcclTelemetryChannelAggregate(ch, &agg);

      if (chPrinted > 0) fprintf(fp, ",\n");
      fprintf(fp, "        {\n");
      if (dev->channels_unknown)
        fprintf(fp, "          \"id\": null,\n");
      else
        fprintf(fp, "          \"id\": %d,\n", ch->id);
      fprintf(fp, "          \"num_wqe_sent\": %lu,\n", (unsigned long)agg.num_wqe_sent);
      fprintf(fp, "          \"num_recv_wqe\": %lu,\n", (unsigned long)agg.num_recv_wqe);
      fprintf(fp, "          \"num_wqe_rcvd\": %lu,\n", (unsigned long)agg.num_wqe_rcvd);
      fprintf(fp, "          \"num_wqe_completed\": %lu,\n", (unsigned long)agg.num_wqe_completed);
      if (sampled) fprintf(fp, "          \"num_wqe_sampled\": %lu,\n", (unsigned long)agg.num_wqe_sampled);
      fprintf(fp, "          \"num_cts_sent\": %lu,\n", (unsigned long)agg.num_cts_sent);
      fprintf(fp, "          \"num_req_completed\": %lu,\n", (unsigned long)ch->num_req_completed);
      fprintf(fp, "          \"num_data_qp\": %d,\n", ch->num_data_qp);
      fprintf(fp, "          \"num_cts_qp\": %d,\n", ch->num_cts_qp);
      fprintf(fp, "          \"num_qp_untracked\": %d,\n", ch->num_qp_untracked);

      /* Exactly the slots a counter update can reach, so totals add up. */
      int numQps = __atomic_load_n(&ch->num_qps, __ATOMIC_ACQUIRE);
      fprintf(fp, "          \"queue_pairs\": [\n");
      for (int q = 0; q < numQps; q++) {
        RcclQpStats* qp = rcclTelemetryQp(ch, q);

        fprintf(fp, "            {\n");
        fprintf(fp, "              \"id\": %d,\n", qp->id);
        fprintf(fp, "              \"data_qp\": %s,\n", qp->is_data_qp ? "true" : "false");
        fprintf(fp, "              \"num_wqe_sent\": %lu,\n", (unsigned long)qp->num_wqe_sent);
        fprintf(fp, "              \"num_recv_wqe\": %lu,\n", (unsigned long)qp->num_recv_wqe);
        fprintf(fp, "              \"num_wqe_rcvd\": %lu,\n", (unsigned long)qp->num_wqe_rcvd);
        fprintf(fp, "              \"num_wqe_completed\": %lu,\n", (unsigned long)qp->num_wqe_completed);
        fprintf(fp, "              \"num_slot_miss\": %lu,\n", (unsigned long)qp->num_slot_miss);
        fprintf(fp, "              \"num_cts_sent\": %lu,\n", (unsigned long)qp->num_cts_sent);
        fprintf(fp, "              \"num_cts_sent_signalled\": %lu,\n",
                (unsigned long)qp->num_cts_sent_signalled);
        fprintf(fp, "              \"num_cts_sent_unsignalled\": %lu,\n",
                (unsigned long)qp->num_cts_sent_unsignalled);
        fprintf(fp, "              \"num_write_wqe\": %lu,\n", (unsigned long)qp->num_write_wqe);
        fprintf(fp, "              \"num_write_imm_wqe\": %lu,\n", (unsigned long)qp->num_write_imm_wqe);
        /* The size of the sample the three latency fields below are computed
         * from, and by construction the sum of that histogram. */
        if (sampled)
          fprintf(fp, "              \"num_wqe_sampled\": %lu,\n", (unsigned long)rcclTelemetryQpSampledCount(qp));
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

        fprintf(fp, "            }%s\n", (q < numQps - 1) ? "," : "");
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
        rcclTelemetryWriteJsonPfcArray(fp, "pfc_rx_frames",   dev->pfc_rx_frames,   1);
      if (pfc->tx_frames_fmt)
        rcclTelemetryWriteJsonPfcArray(fp, "pfc_tx_frames",   dev->pfc_tx_frames,   1);
      if (pfc->rx_pause_us_fmt)
        rcclTelemetryWriteJsonPfcArray(fp, "pfc_rx_pause_us", dev->pfc_rx_pause_us, 1);
      if (pfc->tx_pause_us_fmt)
        rcclTelemetryWriteJsonPfcArray(fp, "pfc_tx_pause_us", dev->pfc_tx_pause_us, 1);
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
        fprintf(fp, ", \"%s\": %ld", rcclTelSampled[c].name, (long)s->cong[c]);
      fprintf(fp, "}%s\n", (i < rcclTelemetryNumSamples - 1) ? "," : "");
    }
    fprintf(fp, "  ]\n");
  } else {
    fprintf(fp, "\n");
  }

  fprintf(fp, "}\n");
}
