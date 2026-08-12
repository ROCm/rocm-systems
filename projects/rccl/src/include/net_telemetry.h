/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_TELEMETRY_H_
#define RCCL_NET_TELEMETRY_H_

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * RCCL Network Telemetry Subsystem
 *
 * This subsystem provides unified telemetry collection for RCCL network transports.
 * Telemetry is enabled at runtime via RCCL_TELEMETRY_ENABLE=1.
 * Optional configuration can be provided via RCCL_TELEMETRY_CONFIG=/path/to/cfg.json
 *
 * Output: JSON file written on process exit to
 *   /tmp/rccl_telemetry_<hostname>_<pid>.json
 *
 * Supported hardware:
 *   - AMD AINIC (driver: ionic)
 */

/* Maximum constants */
#define RCCL_TELEMETRY_MAX_DEVS       16
#define RCCL_TELEMETRY_MAX_CHANNELS   512
#define RCCL_TELEMETRY_MAX_QPS        128
#define RCCL_TELEMETRY_HISTOGRAM_SIZE 16

/* Power-of-two buckets for the WQE payload-size distribution: bucket b holds
 * WQEs of [2^(b-1), 2^b - 1] bytes, bucket 0 holds zero-length WQEs. */
#define RCCL_TELEMETRY_WQE_SIZE_BUCKETS 32

/*
 * Maximum number of scalar hardware counters stored per device.
 * Must be >= the largest per-HW counter table size (see net_telemetry.cc).
 */
#define RCCL_TELEMETRY_MAX_HWC        80

/* Runtime guard - 1 if telemetry is enabled, 0 otherwise */
extern int rcclTelemetryEnabled;

/* Initialize telemetry system - reads env vars, parses config, registers atexit handler */
void rcclTelemetryInit(void);

/* Flush telemetry to JSON file - called via atexit() or can be called manually.
 * HW counters report deltas vs the baseline captured at device registration, so
 * telemetry always covers the whole process lifetime. */
void rcclTelemetryFlush(void);

/*
 * Lightweight per-device software-counter snapshot.
 *
 * Reads only the atomic SW counters already maintained on the hot path. No file
 * I/O, no subprocess, no global reset — cheap enough to call around every
 * collective. A profiler plugin captures one at collective-start and one at
 * collective-stop, then subtracts to get per-collective deltas.
 */
typedef struct {
  int      device_id;
  uint64_t tx_bytes;
  uint64_t rx_bytes;
  uint64_t num_cq_errors;
  uint64_t wqe_sent;
  uint64_t recv_wqe;      /* receive WQEs posted */
  uint64_t wqe_rcvd;      /* completions drained from the CQ */
  uint64_t wqe_completed; /* completions matched to a tracked posting */
  int64_t  wqe_completion_ns_min;   /* min across QPs (0 = none seen) */
  int64_t  wqe_completion_ns_max;   /* max across QPs */
  uint64_t wqe_completion_histogram[RCCL_TELEMETRY_HISTOGRAM_SIZE]; /* summed across channels/QPs */
} RcclTelemetrySwSnapshot;

/*
 * Capture current SW counters for up to maxDevs devices into out[].
 * Returns the number of devices written (0 if telemetry disabled).
 * Does NOT reset the underlying counters (whole-run flush stays intact).
 */
__attribute__((visibility("default")))
int rcclTelemetrySwCapture(RcclTelemetrySwSnapshot* out, int maxDevs);

/*
 * Configuration structure - populated from RCCL_TELEMETRY_CONFIG JSON file
 * or uses defaults if not specified
 */
typedef struct {
  char    output_dir[512];              /* default: "/tmp" */
  int     histogram_max_buckets;        /* default: 5 */
  int64_t histogram_bucket_interval_ns; /* default: 30000 */
  char    hw_counter_list[1024];        /* default: "" means collect all */
} RcclTelemetryConfig;

extern RcclTelemetryConfig rcclTelemetryCfg;

/*
 * Per-QP statistics
 */
/*
 * Counter semantics follow the ANP plugin (ROCm/amd-anp, include/anp_state.h):
 *   num_recv_wqe      receive WQEs posted (incremented after ibv_post_recv)
 *   num_wqe_rcvd      completions drained from the CQ, send and recv alike
 *   num_wqe_completed completions matched to a tracked posting, i.e. those for
 *                     which a post timestamp was recorded and latency is
 *                     therefore computable
 */
typedef struct {
  int      id;
  int      is_data_qp;            /* 1 for data QPs, 0 for CTS QPs */
  uint64_t num_wqe_sent;
  uint64_t num_recv_wqe;
  uint64_t num_wqe_rcvd;
  uint64_t num_wqe_completed;
  uint64_t num_slot_miss;
  uint64_t num_cts_sent;
  uint64_t num_cts_sent_signalled;
  uint64_t num_cts_sent_unsignalled;
  uint64_t num_write_wqe;         /* RDMA_WRITE postings */
  uint64_t num_write_imm_wqe;     /* RDMA_WRITE_WITH_IMM postings */
  int64_t  wqe_completion_ns_min;
  int64_t  wqe_completion_ns_max;
  uint64_t wqe_completion_histogram[RCCL_TELEMETRY_HISTOGRAM_SIZE];
} RcclQpStats;

/*
 * Per-channel statistics
 */
typedef struct {
  int         id;
  uint64_t    num_wqe_sent;
  uint64_t    num_recv_wqe;
  uint64_t    num_wqe_rcvd;
  uint64_t    num_wqe_completed;
  uint64_t    num_cts_sent;
  /* Network requests completed on this channel. Not a WQE count: one request
   * spans one WQE per QP it is striped over, and one CQE completes every
   * sub-request of a multi-send, so this is neither an upper nor a lower bound
   * on the num_wqe_* counters above. */
  uint64_t    num_req_completed;
  int         num_data_qp;
  int         num_cts_qp;
  RcclQpStats qp[RCCL_TELEMETRY_MAX_QPS];
  int         num_qps;
  /* QPs this channel could not track because all RCCL_TELEMETRY_MAX_QPS slots
   * were taken. Their traffic is missing from every counter above. */
  int         num_qp_untracked;
} RcclChannelStats;

/*
 * Per-device statistics including hardware counters.
 *
 * The scalar hw_counters[] array is indexed by position in the active per-HW
 * descriptor table (see net_telemetry.cc). `hw_config` is an opaque pointer to
 * that table; consumers outside of net_telemetry.cc should not dereference it.
 */
typedef struct {
  int    device_id;
  char   roce_device[64];
  char   eth_device[64];
  char   transport[32];       /* e.g., "IB-CAST", "IB" */

  const void* hw_config;      /* opaque: points to active per-HW RcclHwConfig */

  uint64_t tx_bytes;
  uint64_t rx_bytes;
  uint64_t num_cq_errors;
  uint64_t cq_poll_count;
  uint64_t wqe_size_histogram[RCCL_TELEMETRY_WQE_SIZE_BUCKETS];
  int      num_channels;
  int      num_qp_untracked;  /* sum of the per-channel shortfalls */
  RcclChannelStats channels[RCCL_TELEMETRY_MAX_CHANNELS];

  /* Scalar hardware counters — filled at flush time, -1 means N/A.
   * Indexed by position in the active per-HW counter table. */
  int64_t hw_counters[RCCL_TELEMETRY_MAX_HWC];

  /* Per-priority PFC counters (priorities 0-7), -1 if not supported */
  int64_t pfc_rx_frames[8];
  int64_t pfc_tx_frames[8];
  int64_t pfc_rx_pause_us[8];
  int64_t pfc_tx_pause_us[8];

  /* Snapshot-based NIC deltas — internal init snapshots (not written to JSON) */
  int64_t snap_init_tx_bytes;
  int64_t snap_init_rx_bytes;
  int64_t snap_init_tx_packets;
  int64_t snap_init_rx_packets;

  /* Baselines for hw_counters[]/pfc_*[] — captured at device registration,
   * subtracted from the current values at flush so JSON reports deltas. */
  int64_t snap_init_hw_counters[RCCL_TELEMETRY_MAX_HWC];
  int64_t snap_init_pfc_rx_frames[8];
  int64_t snap_init_pfc_tx_frames[8];
  int64_t snap_init_pfc_rx_pause_us[8];
  int64_t snap_init_pfc_tx_pause_us[8];

  /* Snapshot deltas — written to JSON under hw_counters */
  int64_t delta_tx_bytes;
  int64_t delta_rx_bytes;
  int64_t delta_tx_packets;
  int64_t delta_rx_packets;
} RcclDeviceStats;

/* Global device statistics array */
extern RcclDeviceStats rcclTelemetryDevs[RCCL_TELEMETRY_MAX_DEVS];
extern int             rcclTelemetryNumDevs;

/*
 * Helper to register a device for telemetry collection.
 *
 * device_id doubles as the slot in rcclTelemetryDevs[], because that is the
 * index every hot-path helper below is called with. Callers must pass the same
 * device numbering they later use for devIdx, and must register after any
 * reordering of their own device list.
 *
 * Returns the device index (== device_id) or -1 on failure.
 */
int rcclTelemetryRegisterDevice(int device_id, const char* roce_device,
                                 const char* eth_device, const char* transport);

/*
 * Helper to map eth device name from roce device via sysfs
 * eth_device: output buffer (at least 64 bytes)
 */
void rcclTelemetryGetEthDevice(const char* roce_device, char* eth_device, size_t eth_device_size);

/* ------------------------------------------------------------------ */
/* Hot-path telemetry inline functions                                 */
/*                                                                     */
/* These functions encapsulate all telemetry instrumentation for the   */
/* data path. Using functions (vs macros) keeps the hot path code      */
/* clean and readable, similar to NCCL profiler pattern.               */
/*                                                                     */
/* ALGORITHM OVERVIEW:                                                 */
/* 1. SEND PATH (after ibv_post_send):                                 */
/*    - Increment tx_bytes by the payload size                         */
/*    - Record current timestamp for latency tracking                  */
/*                                                                     */
/* 2. RECV PATH (after ibv_post_recv):                                 */
/*    - Increment rx_bytes by the received payload size                */
/*    - Record current timestamp for latency tracking                  */
/*                                                                     */
/* 3. COMPLETION PATH (after ibv_poll_cq success):                     */
/*    - Increment wqe_completed counter                                */
/*    - Compute latency = current_time - post_timestamp                */
/*    - Update histogram bucket based on latency                       */
/*    - Update min/max latency values                                  */
/*                                                                     */
/* 4. ERROR PATH (on CQ error):                                        */
/*    - Increment cq_errors counter                                    */
/* ------------------------------------------------------------------ */

#include <time.h>

/**
 * Get current timestamp in nanoseconds.
 * Used for latency measurement between post and completion.
 */
static inline int64_t rcclTelemetryGetNs(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

/**
 * Record transferred bytes on a device.
 * Call after ibv_post_send succeeds (isSend=1) or when receive data is
 * available (isSend=0).
 *
 * @param devIdx   Device index in rcclTelemetryDevs array
 * @param isSend   1 to add to tx_bytes, 0 to add to rx_bytes
 * @param bytes    Number of bytes transferred
 */
static inline void rcclTelemetryBytes(int devIdx, int isSend, uint64_t bytes) {
  if (!rcclTelemetryEnabled || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return;
  uint64_t* field = isSend ? &rcclTelemetryDevs[devIdx].tx_bytes
                           : &rcclTelemetryDevs[devIdx].rx_bytes;
  __atomic_fetch_add(field, bytes, __ATOMIC_RELAXED);
}

/**
 * Record a CQ error.
 * Call when ibv_poll_cq returns an error completion.
 *
 * @param devIdx   Device index in rcclTelemetryDevs array
 */
static inline void rcclTelemetryCqError(int devIdx) {
  if (!rcclTelemetryEnabled || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return;
  __atomic_fetch_add(&rcclTelemetryDevs[devIdx].num_cq_errors, 1, __ATOMIC_RELAXED);
}

/**
 * Record a completion drained from the CQ, with latency tracking.
 * Call after ibv_poll_cq returns a successful completion.
 *
 * This function:
 * - Increments num_wqe_rcvd for every completion, send or recv
 * - Increments num_wqe_completed only when the completion is matched to a
 *   tracked posting (postTs > 0), mirroring the ANP wqe_id_tracker lookup
 * - Computes latency from post_ts to now for matched completions
 * - Updates the latency histogram bucket
 * - Updates min/max latency atomically
 *
 * Both counters are incremented on the channel as well as on the QP, under the
 * one bounds check, so that a channel total can never include a completion that
 * no QP slot accounts for.
 *
 * @param devIdx   Device index in rcclTelemetryDevs array
 * @param chIdx    Channel index
 * @param qpIdx    Queue pair index within channel
 * @param postTs   Timestamp when work request was posted (0 if unmatched)
 */
static inline void rcclTelemetryWqeComplete(int devIdx, int chIdx, int qpIdx, int64_t postTs) {
  if (!rcclTelemetryEnabled ||
      devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx  < 0 || chIdx  >= RCCL_TELEMETRY_MAX_CHANNELS ||
      qpIdx  < 0 || qpIdx  >= RCCL_TELEMETRY_MAX_QPS)
    return;
  
  RcclChannelStats* ch = &rcclTelemetryDevs[devIdx].channels[chIdx];
  RcclQpStats* qp = &ch->qp[qpIdx];
  __atomic_fetch_add(&qp->num_wqe_rcvd, 1, __ATOMIC_RELAXED);
  __atomic_fetch_add(&ch->num_wqe_rcvd, 1, __ATOMIC_RELAXED);
  
  if (postTs > 0) {
    __atomic_fetch_add(&qp->num_wqe_completed, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&ch->num_wqe_completed, 1, __ATOMIC_RELAXED);
    int64_t latency_ns = rcclTelemetryGetNs() - postTs;
    if (latency_ns > 0) {
      int bucket = (int)(latency_ns / rcclTelemetryCfg.histogram_bucket_interval_ns);
      if (bucket >= rcclTelemetryCfg.histogram_max_buckets)
        bucket = rcclTelemetryCfg.histogram_max_buckets - 1;
      if (bucket >= 0 && bucket < RCCL_TELEMETRY_HISTOGRAM_SIZE)
        __atomic_fetch_add(&qp->wqe_completion_histogram[bucket], 1, __ATOMIC_RELAXED);
      
      int64_t cur_min = __atomic_load_n(&qp->wqe_completion_ns_min, __ATOMIC_RELAXED);
      while (cur_min == 0 || latency_ns < cur_min) {
        if (__atomic_compare_exchange_n(&qp->wqe_completion_ns_min, &cur_min, latency_ns,
            1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
          break;
      }
      
      int64_t cur_max = __atomic_load_n(&qp->wqe_completion_ns_max, __ATOMIC_RELAXED);
      while (latency_ns > cur_max) {
        if (__atomic_compare_exchange_n(&qp->wqe_completion_ns_max, &cur_max, latency_ns,
            1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
          break;
      }
    }
  }
}

/**
 * Increment channel- and QP-level posted-WQE counters: num_wqe_sent for sends,
 * num_recv_wqe for receives. Call after ibv_post_send / ibv_post_recv.
 *
 * @param devIdx   Device index
 * @param chIdx    Channel index
 * @param qpIdx    QP index within channel
 * @param isSend   1 for send, 0 for recv
 */
static inline void rcclTelemetryWqePosted(int devIdx, int chIdx, int qpIdx, int isSend) {
  if (!rcclTelemetryEnabled ||
      devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx  < 0 || chIdx  >= RCCL_TELEMETRY_MAX_CHANNELS ||
      qpIdx  < 0 || qpIdx  >= RCCL_TELEMETRY_MAX_QPS)
    return;
  RcclChannelStats* ch = &rcclTelemetryDevs[devIdx].channels[chIdx];
  if (isSend) {
    __atomic_fetch_add(&ch->qp[qpIdx].num_wqe_sent, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&ch->num_wqe_sent, 1, __ATOMIC_RELAXED);
  } else {
    __atomic_fetch_add(&ch->qp[qpIdx].num_recv_wqe, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&ch->num_recv_wqe, 1, __ATOMIC_RELAXED);
  }
}

/**
 * Record a completed network request on a channel. Call once per request, when
 * its last outstanding event has been accounted for.
 *
 * This is a request count, not a WQE count: it does not aggregate the per-QP
 * num_wqe_* counters and must not be used as if it did.
 *
 * @param devIdx   Device index
 * @param chIdx    Channel index
 */
static inline void rcclTelemetryRequestCompleted(int devIdx, int chIdx) {
  if (!rcclTelemetryEnabled || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx < 0 || chIdx >= RCCL_TELEMETRY_MAX_CHANNELS) return;
  __atomic_fetch_add(&rcclTelemetryDevs[devIdx].channels[chIdx].num_req_completed, 1, __ATOMIC_RELAXED);
}

/**
 * Record a CTS FIFO slot miss.
 * Call when the sender finds the expected CTS slot not yet published and
 * returns without posting.
 */
static inline void rcclTelemetrySlotMiss(int devIdx, int chIdx, int qpIdx) {
  if (!rcclTelemetryEnabled ||
      devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx  < 0 || chIdx  >= RCCL_TELEMETRY_MAX_CHANNELS ||
      qpIdx  < 0 || qpIdx  >= RCCL_TELEMETRY_MAX_QPS)
    return;
  __atomic_fetch_add(&rcclTelemetryDevs[devIdx].channels[chIdx].qp[qpIdx].num_slot_miss,
                     1, __ATOMIC_RELAXED);
}

/**
 * Record a CTS posting on a CTS QP.
 * Call after ibv_post_send succeeds on the CTS path.
 *
 * @param signalled  1 if the work request carried IBV_SEND_SIGNALED
 */
static inline void rcclTelemetryCtsSent(int devIdx, int chIdx, int qpIdx, int signalled) {
  if (!rcclTelemetryEnabled ||
      devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx  < 0 || chIdx  >= RCCL_TELEMETRY_MAX_CHANNELS ||
      qpIdx  < 0 || qpIdx  >= RCCL_TELEMETRY_MAX_QPS)
    return;
  RcclChannelStats* ch = &rcclTelemetryDevs[devIdx].channels[chIdx];
  RcclQpStats* qp = &ch->qp[qpIdx];
  __atomic_fetch_add(&qp->num_cts_sent, 1, __ATOMIC_RELAXED);
  __atomic_fetch_add(&ch->num_cts_sent, 1, __ATOMIC_RELAXED);
  if (signalled)
    __atomic_fetch_add(&qp->num_cts_sent_signalled, 1, __ATOMIC_RELAXED);
  else
    __atomic_fetch_add(&qp->num_cts_sent_unsignalled, 1, __ATOMIC_RELAXED);
}

/**
 * Record an RDMA_WRITE (withImm=0) or RDMA_WRITE_WITH_IMM (withImm=1) posting.
 */
static inline void rcclTelemetryWriteWqe(int devIdx, int chIdx, int qpIdx, int withImm) {
  if (!rcclTelemetryEnabled ||
      devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx  < 0 || chIdx  >= RCCL_TELEMETRY_MAX_CHANNELS ||
      qpIdx  < 0 || qpIdx  >= RCCL_TELEMETRY_MAX_QPS)
    return;
  RcclQpStats* qp = &rcclTelemetryDevs[devIdx].channels[chIdx].qp[qpIdx];
  __atomic_fetch_add(withImm ? &qp->num_write_imm_wqe : &qp->num_write_wqe,
                     1, __ATOMIC_RELAXED);
}

/**
 * Record one ibv_poll_cq invocation on a device's completion queue.
 */
static inline void rcclTelemetryCqPoll(int devIdx) {
  if (!rcclTelemetryEnabled || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return;
  __atomic_fetch_add(&rcclTelemetryDevs[devIdx].cq_poll_count, 1, __ATOMIC_RELAXED);
}

/**
 * Record the payload size of a posted WQE into the per-device power-of-two
 * size distribution.
 */
static inline void rcclTelemetryWqeSize(int devIdx, uint64_t bytes) {
  if (!rcclTelemetryEnabled || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return;
  int bucket = 0;
  if (bytes > 0) {
    bucket = 64 - __builtin_clzll((unsigned long long)bytes);
    if (bucket >= RCCL_TELEMETRY_WQE_SIZE_BUCKETS)
      bucket = RCCL_TELEMETRY_WQE_SIZE_BUCKETS - 1;
  }
  __atomic_fetch_add(&rcclTelemetryDevs[devIdx].wqe_size_histogram[bucket], 1, __ATOMIC_RELAXED);
}

/**
 * Reserve a block of telemetry QP slots on a channel.
 * Call during connection setup (connect/accept) to register QPs for tracking.
 *
 * This only allocates slots; the data/CTS role is per QP and is assigned
 * separately via rcclTelemetrySetQpRole().
 *
 * A channel holds at most RCCL_TELEMETRY_MAX_QPS slots. When fewer than numQps
 * are left, the request is granted in part: *numSlots comes back below numQps
 * and the shortfall is added to num_qp_untracked on the channel and on the
 * device. The caller must leave telQpSlot == -1 on the QPs that got no slot, so
 * that no hot-path hook can charge their events anywhere. That, and counting
 * channel totals only where a QP slot is also counted, is what keeps every
 * channel aggregate equal to the sum over its QP slots.
 *
 * @param devIdx     Device index in rcclTelemetryDevs array
 * @param chIdx      Channel index (typically allocated via atomic counter)
 * @param numQps     Number of QPs the caller wants to register on this channel
 * @param numSlots   Out: number of slots actually reserved, 0 if none
 * @return           First reserved QP slot index, or -1 if no slot was reserved
 *
 * Usage (in ncclIbConnect/ncclIbAccept):
 *   int numSlots = 0;
 *   int qpSlot = rcclTelemetrySetupChannel(devIdx, chIdx, numQps, &numSlots);
 *   for (int q = 0; q < numSlots; q++) {
 *     comm->base.qps[q].telQpSlot = qpSlot + q;
 *     rcclTelemetrySetQpRole(devIdx, chIdx, qpSlot + q, comm->base.qps[q].isDataQp);
 *   }
 */
static inline int rcclTelemetrySetupChannel(int devIdx, int chIdx, int numQps, int* numSlots) {
  if (numSlots) *numSlots = 0;
  if (!rcclTelemetryEnabled || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS || chIdx < 0 ||
      chIdx >= RCCL_TELEMETRY_MAX_CHANNELS || numQps <= 0) {
    return -1;
  }
  
  RcclDeviceStats* dstat = &rcclTelemetryDevs[devIdx];
  RcclChannelStats* ch = &dstat->channels[chIdx];
  
  /* Set channel ID (idempotent if called multiple times for same channel) */
  ch->id = chIdx;
  
  /* Allocate QP slots atomically, as many as still fit */
  int startSlot = __atomic_fetch_add(&ch->num_qps, numQps, __ATOMIC_RELAXED);
  int granted = RCCL_TELEMETRY_MAX_QPS - startSlot;
  if (granted > numQps) granted = numQps;
  if (granted < 0) granted = 0;
  if (granted < numQps) {
    /* Give back what did not fit, so num_qps stays a slot count, and report the
     * QPs that will go untracked instead of dropping them silently. */
    __atomic_fetch_sub(&ch->num_qps, numQps - granted, __ATOMIC_RELAXED);
    __atomic_fetch_add(&ch->num_qp_untracked, numQps - granted, __ATOMIC_RELAXED);
    __atomic_fetch_add(&dstat->num_qp_untracked, numQps - granted, __ATOMIC_RELAXED);
  }
  if (granted == 0) return -1;
  
  /* Initialize QP slot IDs */
  for (int q = 0; q < granted; q++) {
    ch->qp[startSlot + q].id = startSlot + q;
  }
  
  /* Update device's channel count */
  int cur = __atomic_load_n(&dstat->num_channels, __ATOMIC_RELAXED);
  while (chIdx + 1 > cur) {
    if (__atomic_compare_exchange_n(&dstat->num_channels, &cur, chIdx + 1,
                                    1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      break;
  }
  
  if (numSlots) *numSlots = granted;
  return startSlot;
}

/**
 * Assign the data/CTS role of a single telemetry QP slot reserved by
 * rcclTelemetrySetupChannel(), and count it towards the channel's
 * num_data_qp / num_cts_qp totals.
 *
 * Must be called at most once per slot, since it bumps a channel total.
 *
 * @param devIdx     Device index
 * @param chIdx      Channel index
 * @param qpIdx      QP slot index within the channel
 * @param isDataQp   nonzero for a data QP, zero for a CTS QP
 */
static inline void rcclTelemetrySetQpRole(int devIdx, int chIdx, int qpIdx, int isDataQp) {
  if (!rcclTelemetryEnabled ||
      devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx  < 0 || chIdx  >= RCCL_TELEMETRY_MAX_CHANNELS ||
      qpIdx  < 0 || qpIdx  >= RCCL_TELEMETRY_MAX_QPS)
    return;
  RcclChannelStats* ch = &rcclTelemetryDevs[devIdx].channels[chIdx];
  ch->qp[qpIdx].is_data_qp = isDataQp ? 1 : 0;
  if (isDataQp)
    __atomic_fetch_add(&ch->num_data_qp, 1, __ATOMIC_RELAXED);
  else
    __atomic_fetch_add(&ch->num_cts_qp, 1, __ATOMIC_RELAXED);
}

#ifdef __cplusplus
}
#endif

#endif /* RCCL_NET_TELEMETRY_H_ */
