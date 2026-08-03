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

/*
 * Maximum number of scalar hardware counters stored per device.
 * Must be >= the largest per-HW counter table size (see net_telemetry.cc).
 */
#define RCCL_TELEMETRY_MAX_HWC        80

/* Runtime guard - 1 if telemetry is enabled, 0 otherwise */
extern int rcclTelemetryEnabled;

/* Initialize telemetry system - reads env vars, parses config, registers atexit handler */
void rcclTelemetryInit(void);

/* Flush telemetry to JSON file - called via atexit() or can be called manually */
void rcclTelemetryFlush(void);

/* Bracketed-snapshot API. Begin zeros runtime stats + re-baselines
 * ethtool; End collects HW counters, computes deltas, writes JSON.
 * Per-process state, mutex-serialized; output_path NULL = default. */
__attribute__((visibility("default")))
void rcclTelemetrySnapshotBegin(void);

__attribute__((visibility("default")))
void rcclTelemetrySnapshotEnd(const char* output_path);

/*
 * Lightweight per-device software-counter snapshot for per-collective bracketing.
 *
 * Unlike the SnapshotBegin/End pair (which forks ethtool and writes JSON), this
 * only reads the atomic SW counters already maintained on the hot path. No file
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
  uint64_t wqe_rcvd;
  uint64_t wqe_completed;
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
typedef struct {
  int      id;
  uint64_t num_wqe_sent;
  uint64_t num_wqe_rcvd;
  uint64_t num_wqe_completed;
  uint64_t num_slot_miss;
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
  uint64_t    num_wqe_rcvd;
  uint64_t    num_wqe_completed;
  uint64_t    num_cts_sent;
  int         num_data_qp;
  int         num_cts_qp;
  RcclQpStats qp[RCCL_TELEMETRY_MAX_QPS];
  int         num_qps;
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
  int      num_channels;
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

  /* Baselines for hw_counters[]/pfc_*[] — captured at SnapshotInit, subtracted
   * from the current values at flush/SnapshotEnd so JSON reports deltas. */
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
 * Helper to register a device for telemetry collection
 * Returns the device index or -1 on failure
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
  if (!rcclTelemetryEnabled || devIdx < 0) return;
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
  if (!rcclTelemetryEnabled || devIdx < 0) return;
  __atomic_fetch_add(&rcclTelemetryDevs[devIdx].num_cq_errors, 1, __ATOMIC_RELAXED);
}

/**
 * Record WQE completion with latency tracking.
 * Call after ibv_poll_cq returns a successful completion.
 *
 * This function:
 * - Increments wqe_completed counter
 * - Computes latency from post_ts to now
 * - Updates the latency histogram bucket
 * - Updates min/max latency atomically
 *
 * @param devIdx   Device index in rcclTelemetryDevs array
 * @param chIdx    Channel index
 * @param qpIdx    Queue pair index within channel
 * @param postTs   Timestamp when work request was posted (0 to skip latency)
 */
static inline void rcclTelemetryWqeComplete(int devIdx, int chIdx, int qpIdx, int64_t postTs) {
  if (!rcclTelemetryEnabled ||
      devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx  < 0 || chIdx  >= RCCL_TELEMETRY_MAX_CHANNELS ||
      qpIdx  < 0 || qpIdx  >= RCCL_TELEMETRY_MAX_QPS)
    return;
  
  RcclQpStats* qp = &rcclTelemetryDevs[devIdx].channels[chIdx].qp[qpIdx];
  __atomic_fetch_add(&qp->num_wqe_completed, 1, __ATOMIC_RELAXED);
  
  if (postTs > 0) {
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
 * Increment channel- and QP-level WQE sent (isSend=1) or received (isSend=0)
 * counters. Call after ibv_post_send / ibv_post_recv.
 *
 * @param devIdx   Device index
 * @param chIdx    Channel index
 * @param qpIdx    QP index within channel
 * @param isSend   1 for send, 0 for recv
 */
static inline void rcclTelemetryWqePosted(int devIdx, int chIdx, int qpIdx, int isSend) {
  if (!rcclTelemetryEnabled || devIdx < 0) return;
  RcclChannelStats* ch = &rcclTelemetryDevs[devIdx].channels[chIdx];
  if (isSend) {
    __atomic_fetch_add(&ch->qp[qpIdx].num_wqe_sent, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&ch->num_wqe_sent, 1, __ATOMIC_RELAXED);
  } else {
    __atomic_fetch_add(&ch->qp[qpIdx].num_wqe_rcvd, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&ch->num_wqe_rcvd, 1, __ATOMIC_RELAXED);
  }
}

/**
 * Increment channel-level WQE completed counter.
 *
 * @param devIdx   Device index
 * @param chIdx    Channel index
 */
static inline void rcclTelemetryChannelCompleted(int devIdx, int chIdx) {
  if (!rcclTelemetryEnabled || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx < 0 || chIdx >= RCCL_TELEMETRY_MAX_CHANNELS) return;
  __atomic_fetch_add(&rcclTelemetryDevs[devIdx].channels[chIdx].num_wqe_completed, 1, __ATOMIC_RELAXED);
}

/**
 * Setup a telemetry channel with QP slots.
 * Call during connection setup (connect/accept) to register QPs for tracking.
 *
 * @param devIdx     Device index in rcclTelemetryDevs array
 * @param chIdx      Channel index (typically allocated via atomic counter)
 * @param numQps     Number of QPs to register for this channel on this device
 * @param isDataQp   true for data QPs, false for CTS QPs
 * @return           Starting QP slot index, or -1 on failure
 *
 * Usage (in ncclIbConnect/ncclIbAccept):
 *   int qpSlot = rcclTelemetrySetupChannel(devIdx, chIdx, numQps, true);
 *   if (qpSlot >= 0) {
 *     for (int q = 0; q < numQps; q++)
 *       comm->base.qps[q].telQpSlot = qpSlot + q;
 *   }
 */
static inline int rcclTelemetrySetupChannel(int devIdx, int chIdx, int numQps, int isDataQp) {
  if (!rcclTelemetryEnabled || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS ||
      chIdx < 0 || chIdx >= RCCL_TELEMETRY_MAX_CHANNELS) {
    return -1;
  }
  
  RcclDeviceStats* dstat = &rcclTelemetryDevs[devIdx];
  RcclChannelStats* ch = &dstat->channels[chIdx];
  
  /* Set channel ID (idempotent if called multiple times for same channel) */
  ch->id = chIdx;
  
  /* Allocate QP slots atomically */
  int startSlot = __atomic_fetch_add(&ch->num_qps, numQps, __ATOMIC_RELAXED);
  if (startSlot + numQps > RCCL_TELEMETRY_MAX_QPS) {
    /* Rollback if we exceeded the limit */
    __atomic_fetch_sub(&ch->num_qps, numQps, __ATOMIC_RELAXED);
    return -1;
  }
  
  /* Initialize QP slot IDs */
  for (int q = 0; q < numQps && (startSlot + q) < RCCL_TELEMETRY_MAX_QPS; q++) {
    ch->qp[startSlot + q].id = startSlot + q;
  }
  
  /* Track data QPs vs CTS QPs */
  if (isDataQp) {
    __atomic_fetch_add(&ch->num_data_qp, numQps, __ATOMIC_RELAXED);
  } else {
    __atomic_fetch_add(&ch->num_cts_qp, numQps, __ATOMIC_RELAXED);
  }
  
  /* Update device's channel count */
  int cur = __atomic_load_n(&dstat->num_channels, __ATOMIC_RELAXED);
  while (chIdx + 1 > cur) {
    if (__atomic_compare_exchange_n(&dstat->num_channels, &cur, chIdx + 1,
                                    1, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
      break;
  }
  
  return startSlot;
}

#ifdef __cplusplus
}
#endif

#endif /* RCCL_NET_TELEMETRY_H_ */
