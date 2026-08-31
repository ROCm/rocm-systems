/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_NET_TELEMETRY_H_
#define RCCL_NET_TELEMETRY_H_

#include <stdint.h>
#include <stddef.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * RCCL Network Telemetry Subsystem
 *
 * This subsystem provides unified telemetry collection for RCCL network transports.
 * Telemetry is enabled at runtime via RCCL_TELEMETRY_ENABLE=1.
 * Configuration is read from environment variables (RCCL_TELEMETRY_OUTPUT_DIR,
 * RCCL_TELEMETRY_SAMPLE_MS, etc.); there is no JSON config file.
 *
 * Output: JSON file written on process exit to
 *   <output_dir>/rccl_telemetry_<hostname>_<uid>_<pid>.json
 *   (output_dir defaults to /tmp)
 *
 * Supported hardware (per-driver HW counter tables):
 *   - AMD AINIC (driver: ionic)
 *   - Mellanox/NVIDIA (driver: mlx5)
 *   - Broadcom Thor2 (driver: bnxt_re / bnxt_en)
 * Software counters are transport-level and driver-independent.
 */

/* One slot per device the transport can enumerate, since device_id doubles as
 * the slot index. Kept equal to the transports' own MAX_IB_DEVS: a device that
 * does not fit here cannot be registered, and all of its traffic silently
 * lands in num_qp_untracked. The transports static_assert against this. */
#define RCCL_TELEMETRY_MAX_DEVS       32
/* Sanity bound, not a reservation: channel storage is allocated on demand, and
 * this only rejects indices too absurd to be a real channel. Matches the
 * largest MAXCHANNELS RCCL is built with (src/include/device.h). */
#define RCCL_TELEMETRY_MAX_CHANNELS   512
#define RCCL_TELEMETRY_HISTOGRAM_SIZE 16

/* PFC is defined over eight traffic classes, so every per-priority counter
 * array is this wide and every loop over priorities runs this far. */
#define RCCL_TELEMETRY_NUM_PFC_PRIO   8

/* Every filesystem path telemetry builds (sysfs, debugfs, procfs, the output
 * file) uses one buffer size, so no call site has to reason about which limit
 * applies to it. PATH_MAX is 4096 on Linux; the fallback covers platforms
 * where limits.h leaves it undefined. */
#ifdef PATH_MAX
#define RCCL_TEL_PATH_MAX PATH_MAX
#else
#define RCCL_TEL_PATH_MAX 1024
#endif

/* POSIX puts this in <limits.h>, but it is optional and musl-based and some
 * BSD-derived libcs omit it. 255 is the Linux kernel bound. */
#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255
#endif

/* Worst case of the file name telemetry appends to output_dir, which is
 * "/rccl_telemetry_<hostname>_<uid>_<pid>.json" (40 covers the uid and pid
 * digits). */
#define RCCL_TEL_FILENAME_MAX (sizeof("/rccl_telemetry___.json") + HOST_NAME_MAX + 40)

/* Device names as sysfs exposes them: IBV_SYSFS_NAME_MAX is 64 for the RoCE
 * device, and IFNAMSIZ (16) bounds the netdev name, so one buffer covers both
 * without pulling verbs headers into this file. */
#define RCCL_TELEMETRY_DEV_NAME_MAX   64
/* Transport label, e.g. "IB-CAST" or "IB". */
#define RCCL_TELEMETRY_TRANSPORT_MAX  32

/* Power-of-two buckets for the WQE payload-size distribution: bucket b holds
 * WQEs of [2^(b-1), 2^b - 1] bytes, bucket 0 holds zero-length WQEs. */
#define RCCL_TELEMETRY_WQE_SIZE_BUCKETS 32

/* Assumed cache-line size. Everything a hot-path hook writes to is aligned to
 * this, so two threads working on different QPs (or different channels) never
 * contend for the same line. 64 on x86-64 and on aarch64 Linux; over-aligning
 * only wastes a little memory, under-aligning costs throughput. */
#define RCCL_TELEMETRY_CACHELINE 64

/*
 * Maximum number of scalar hardware counters stored per device.
 *
 * Every per-HW counter table becomes a config through RCCL_TEL_HW_CONFIG() in
 * net_telemetry.cc, which derives the table's counter count and asserts it
 * against this bound in one step, so a table cannot outgrow this array
 * without failing the build.
 */
#define RCCL_TELEMETRY_MAX_HWC        80

/* Bound on the longest json_name in any counter table, used to size the config
 * filter list below so that naming every counter still fits. */
#define RCCL_TELEMETRY_HWC_NAME_MAX   32

/* Compile-time assert usable from both the C++ implementation and any C
 * consumer of this header. */
#ifndef RCCL_TEL_STATIC_ASSERT
#ifdef __cplusplus
#define RCCL_TEL_STATIC_ASSERT(cond, msg) static_assert(cond, msg)
#else
#define RCCL_TEL_STATIC_ASSERT(cond, msg) _Static_assert(cond, msg)
#endif
#endif

/* Runtime guard - 1 if telemetry is enabled, 0 otherwise.
 * Published with a release store at the very end of rcclTelemetryInit(), so it
 * is never 1 while the tables behind it are still unseeded. */
extern int rcclTelemetryEnabled;

/* Acquire-load of the guard, pairing with that release store. Every telemetry
 * entry point tests this; read the call-site rule further down. */
static inline int rcclTelemetryOn(void) {
  return __atomic_load_n(&rcclTelemetryEnabled, __ATOMIC_ACQUIRE);
}

/*
 * ERROR MODEL
 *
 * Telemetry is diagnostic, so no failure inside it may fail the collective that
 * carries it: a run must produce the same result whether telemetry succeeded,
 * degraded or never started. The entry points therefore never return
 * ncclResult_t and are never wrapped in NCCLCHECK.
 *
 * They do report, so a failure is not silent. Each returns 0 on success and -1
 * when it could not do its job while enabled, having already logged the reason;
 * being disabled is success, not failure. Callers are expected to log the
 * status and carry on. Partial degradation has its own channel instead of a
 * return code: QPs that could not be given a slot are counted in
 * num_qp_untracked and reported in the JSON, and counters that could not be
 * read are emitted as -1 rather than as a plausible zero.
 */

/* Initialize telemetry system - reads env vars, parses config, registers atexit handler */
int rcclTelemetryInit(void);

/* Flush telemetry to JSON file - called via atexit() or can be called manually.
 * HW counters report deltas vs the baseline captured on each device's first use,
 * so telemetry always covers the whole period the device carried traffic.
 * Only the first call writes a file, and no telemetry storage is ever released,
 * so a manual flush cannot race a concurrent hot-path hook. */
int rcclTelemetryFlush(void);

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
  uint64_t wqe_sampled;   /* of those, the ones whose latency was measured;
                         * equals wqe_completed unless 1-in-N latency
                         * sampling is on, and always equals the sum of
                         * wqe_completion_histogram below */
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
 * Configuration structure - populated from environment variables
 * (RCCL_TELEMETRY_*), or uses defaults if not specified
 */
typedef struct {
  /* Directory the JSON is written to, default "/tmp". PATH_MAX bounds the
   * finished path, so the directory gets PATH_MAX minus the file name that is
   * appended to it, and the join can never truncate. */
  char    output_dir[RCCL_TEL_PATH_MAX - RCCL_TEL_FILENAME_MAX];
  int     histogram_max_buckets;        /* default: 5 */
  int64_t histogram_bucket_interval_ns; /* default: 30000 */

  /* Derived from histogram_bucket_interval_ns, never set directly: call
   * rcclTelemetryConfigDeriveHistogram() after changing the interval. They let
   * the completion hook reach the histogram bucket with a multiply instead of a
   * 64-bit division; see rcclTelemetryLatencyBucket() for the exactness proof.
   * Deliberately adjacent to the two fields above: the completion hook reads
   * max_buckets, recip and recip_max_ns, and they fit one cache line together.
   *
   * histogram_recip == 0 means "not derived yet" and selects the division, so
   * a zeroed config behaves exactly like the old code. */
  uint64_t histogram_recip;
  uint64_t histogram_recip_max_ns;

  /* Comma-separated json_names to collect; "" means collect all. Sized so that
   * naming every counter of the widest table still fits without truncation. */
  char    hw_counter_list[RCCL_TELEMETRY_MAX_HWC * (RCCL_TELEMETRY_HWC_NAME_MAX + 1)];
} RcclTelemetryConfig;

extern RcclTelemetryConfig rcclTelemetryCfg;

/* ------------------------------------------------------------------ */
/* 1-in-N completion-latency sampling                                  */
/* ------------------------------------------------------------------ */

/*
 * Everything except the latency of a WQE is a counter increment. The latency
 * costs two clock_gettime() calls per WQE — one on the post path, one on the
 * completion path — plus the bucket computation and the two min/max CAS loops,
 * and that family is the bulk of the measured per-WQE cost. Sampling one
 * posting in N removes all of it for the other N-1.
 *
 * The decision has to be made at post time, because that is where the first
 * timestamp would be taken. It is carried to the completion path in the post
 * timestamp itself, using a sentinel so that no existing counter changes
 * meaning:
 *
 *   postTs == 0   completion not matched to a tracked posting  (unchanged)
 *   postTs == 1   matched, but this WQE was not sampled        (new)
 *   postTs >  1   matched and sampled; a real CLOCK_MONOTONIC reading
 *
 * num_wqe_completed therefore still counts every postTs > 0 and stays exact
 * for any N, while all latency work is confined to postTs > 1. The second
 * clock read already sits inside the postTs > 0 branch, so an unsampled WQE
 * ends up reading the clock zero times on both sides.
 *
 * A real reading is nanoseconds since boot, so it can only collide with the
 * sentinel in the first nanosecond of uptime, before any QP exists.
 */
#define RCCL_TELEMETRY_TS_UNSAMPLED 1

/*
 * N - 1, so the test is a mask. 0 means N == 1, i.e. sample everything, which
 * is the default and reproduces the un-sampled behaviour exactly. Written once
 * during rcclTelemetryInit(), before the enable flag is published, and only
 * read afterwards. RCCL_TELEMETRY_LATENCY_SAMPLE sets it; see net_telemetry.cc
 * for how a non-power-of-two request is rounded.
 */
extern uint64_t rcclTelemetryLatencySampleMask;

/* The N actually in effect (a power of two, >= 1), reported in the JSON so a
 * consumer never has to guess what the histogram covers. */
extern int rcclTelemetryLatencySampleN;

/* Fixed-point position of histogram_recip. */
#define RCCL_TELEMETRY_RECIP_SHIFT 63

/*
 * Recompute the derived histogram fields from histogram_bucket_interval_ns.
 * Idempotent, and a no-op for a non-positive interval (which leaves
 * histogram_recip at 0, i.e. keeps the division). Must be called after every
 * change to the interval; config parsing does, and so must any test.
 */
static inline void rcclTelemetryConfigDeriveHistogram(void) {
  int64_t d = rcclTelemetryCfg.histogram_bucket_interval_ns;
  if (d <= 0) {
    rcclTelemetryCfg.histogram_recip = 0;
    rcclTelemetryCfg.histogram_recip_max_ns = 0;
    return;
  }
  const uint64_t two63 = (uint64_t)1 << RCCL_TELEMETRY_RECIP_SHIFT;
  /* ceil(2^63 / d) written as floor + 1: the "round up" reciprocal. Fits in 64
   * bits for every d >= 1 (worst case d == 1 gives 2^63 + 1). */
  rcclTelemetryCfg.histogram_recip = two63 / (uint64_t)d + 1;
  rcclTelemetryCfg.histogram_recip_max_ns = two63 / (uint64_t)d;
}

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
 *
 * This is the only structure the hot-path hooks write to, so it is cache-line
 * aligned: two threads driving neighbouring QPs must not share a line. The
 * blocks holding these slots are allocated with the same alignment.
 */
typedef struct __attribute__((aligned(RCCL_TELEMETRY_CACHELINE))) {
  int      id;
  int      is_data_qp;            /* 1 for data QPs, 0 for CTS QPs */
  uint64_t num_wqe_sent;
  uint64_t num_recv_wqe;
  uint64_t num_wqe_rcvd;
  uint64_t num_wqe_completed;
  /* Postings counted for 1-in-N latency sampling; see rcclTelemetryPostTs().
   * Deliberately on the first line, next to the counters the same post already
   * writes, so selecting a WQE costs no cache line the hook did not own. */
  uint64_t latency_sample_seq;
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

/* Doubling heap blocks. Growth only appends and nothing is ever freed, so a
 * reader needs no lock and no slot can be pulled out from under it. */
#define RCCL_TELEMETRY_BLOCKS 32
#define RCCL_TELEMETRY_BLOCK0_MIN_LOG2 3

/*
 * Per-channel statistics
 *
 * The channel-level WQE/CTS totals (num_wqe_sent, num_recv_wqe, num_wqe_rcvd,
 * num_wqe_completed, num_cts_sent) are NOT stored here. They are exact sums
 * over this channel's QP slots and are derived on demand by
 * rcclTelemetryChannelAggregate(); see the comment on that function for why.
 *
 * Cache-line aligned for the same reason as RcclQpStats: num_req_completed is
 * written from the data path, and neighbouring channels must not share a line.
 */
typedef struct __attribute__((aligned(RCCL_TELEMETRY_CACHELINE))) {
  /* --- First line: read-mostly. Every hook call reads qp_blocks, num_qps and
   * qp_block0_log2 to reach a QP slot; all writes to this line happen during
   * connection setup, never on the data path. --- */
  int id;
  int num_data_qp;
  int num_cts_qp;
  /* Blocks of RcclQpStats; stored before qp_capacity is raised. */
  void** qp_blocks;
  /* Allocation bound; runs ahead of num_qps since blocks are powers of 2. */
  int qp_capacity;
  int qp_block0_log2;
  /* Slots handed out; the bound every counter update tests. */
  int num_qps;
  /* QPs with no slot; their traffic is missing from every counter here. */
  int num_qp_untracked;

  /* --- Second line: the only field the data path writes. ---
   * Network requests completed on this channel. Not a WQE count: one request
   * spans one WQE per QP it is striped over, and one CQE completes every
   * sub-request of a multi-send, so this is neither an upper nor a lower bound
   * on the derived num_wqe_* totals. It is genuinely per-channel — no QP sum
   * produces it — so unlike those it stays a stored counter.
   *
   * It gets its own line deliberately: measured runs put it at 1-2.5x the
   * num_wqe_sent rate, so leaving it next to the read-mostly fields above would
   * invalidate, on every request, the line that every hook call reads. */
  uint64_t num_req_completed __attribute__((aligned(RCCL_TELEMETRY_CACHELINE)));
} RcclChannelStats;

/*
 * Per-device statistics including hardware counters.
 *
 * The scalar hw_counters[] array is indexed by position in the active per-HW
 * descriptor table (see net_telemetry.cc). `hw_config` is an opaque pointer to
 * that table; consumers outside of net_telemetry.cc should not dereference it.
 */
typedef struct __attribute__((aligned(RCCL_TELEMETRY_CACHELINE))) {
  /* --- Read-mostly / cold metadata. Written only at registration, first use or
   * flush, never by a per-WQE or per-request hook, so it may share a line. --- */
  int    device_id;
  char   roce_device[RCCL_TELEMETRY_DEV_NAME_MAX];
  char   eth_device[RCCL_TELEMETRY_DEV_NAME_MAX];
  char   transport[RCCL_TELEMETRY_TRANSPORT_MAX]; /* e.g., "IB-CAST", "IB" */

  const void* hw_config;      /* opaque: points to active per-HW RcclHwConfig */

  int      num_channels;
  /* Set when the transport has no real NCCL channel id (non-AINIC): every QP
   * then lands in one device-wide bucket, so num_channels is reported unknown
   * rather than a misleading 1. */
  int      channels_unknown;
  /* Per-channel shortfalls plus channels that could not be tracked. */
  int num_qp_untracked;
  /* Blocks of RcclChannelStats; stored before channel_capacity is raised. */
  void** channel_blocks;
  int channel_capacity;
  int channel_block0_log2;
  uint64_t num_cq_errors;
  /* Folded in at teardown from the per-comm counters, never on the data path,
   * so it stays with the cold fields above. */
  uint64_t cq_poll_count;

  /* --- Hot: bumped on every posted WQE. Own line(s) so it never shares with
   * the byte counters below or the metadata above. --- */
  uint64_t wqe_size_histogram[RCCL_TELEMETRY_WQE_SIZE_BUCKETS]
      __attribute__((aligned(RCCL_TELEMETRY_CACHELINE)));

  /* --- Hot: per-request byte counters. tx and rx take separate lines so a rank
   * driving both directions on one device does not ping-pong a shared line. --- */
  uint64_t tx_bytes __attribute__((aligned(RCCL_TELEMETRY_CACHELINE)));
  uint64_t rx_bytes __attribute__((aligned(RCCL_TELEMETRY_CACHELINE)));

  /* Scalar hardware counters — filled at flush time, -1 means N/A.
   * Indexed by position in the active per-HW counter table. */
  int64_t hw_counters[RCCL_TELEMETRY_MAX_HWC] __attribute__((aligned(RCCL_TELEMETRY_CACHELINE)));

  /* Per-priority PFC counters, -1 if not supported */
  int64_t pfc_rx_frames[RCCL_TELEMETRY_NUM_PFC_PRIO];
  int64_t pfc_tx_frames[RCCL_TELEMETRY_NUM_PFC_PRIO];
  int64_t pfc_rx_pause_us[RCCL_TELEMETRY_NUM_PFC_PRIO];
  int64_t pfc_tx_pause_us[RCCL_TELEMETRY_NUM_PFC_PRIO];

  /* Snapshot-based NIC deltas — internal init snapshots (not written to JSON) */
  int64_t snap_init_tx_bytes;
  int64_t snap_init_rx_bytes;
  int64_t snap_init_tx_packets;
  int64_t snap_init_rx_packets;

  /* 1 once the HW-counter baseline below has been captured.
   *
   * The baseline is taken on first use of the device (the first channel setup
   * on it) rather than at registration: a rank registers every NIC it can see
   * but normally drives only one or two, and each baseline costs a full ethtool
   * stats read (ioctl) plus IB-sysfs reads. Flush skips devices that never got a
   * baseline, so their hw_counters stay at the -1 (N/A) that registration
   * installed instead of being reported as bogus absolute values. */
  int snap_taken;

  /* Baselines for hw_counters[]/pfc_*[] — captured on first use of the device,
   * subtracted from the current values at flush so JSON reports deltas. */
  int64_t snap_init_hw_counters[RCCL_TELEMETRY_MAX_HWC];
  int64_t snap_init_pfc_rx_frames[RCCL_TELEMETRY_NUM_PFC_PRIO];
  int64_t snap_init_pfc_tx_frames[RCCL_TELEMETRY_NUM_PFC_PRIO];
  int64_t snap_init_pfc_rx_pause_us[RCCL_TELEMETRY_NUM_PFC_PRIO];
  int64_t snap_init_pfc_tx_pause_us[RCCL_TELEMETRY_NUM_PFC_PRIO];

  /* Snapshot deltas — written to JSON under hw_counters */
  int64_t delta_tx_bytes;
  int64_t delta_rx_bytes;
  int64_t delta_tx_packets;
  int64_t delta_rx_packets;
} RcclDeviceStats;

/* Layout tripwire: this hot-path struct is cache-line aligned and splits its
 * counters across lines to avoid false sharing. Pin the size so any field change
 * that alters the layout is a deliberate, reviewed edit, and keep it a whole
 * number of cache lines. */
RCCL_TEL_STATIC_ASSERT(sizeof(RcclDeviceStats) == 2560,
  "RcclDeviceStats size changed; re-check the cache-line layout");
RCCL_TEL_STATIC_ASSERT(sizeof(RcclDeviceStats) % RCCL_TELEMETRY_CACHELINE == 0,
  "RcclDeviceStats must stay a whole number of cache lines");

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
 * The netdev name is resolved from roce_device via sysfs here, so callers hold
 * no buffer for it.
 *
 * Returns the device index (== device_id) or -1 on failure.
 */
int rcclTelemetryRegisterDevice(int device_id, const char* roce_device, const char* transport);

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
/* The timestamps in 1 and 2, and everything in 3 that depends on      */
/* them, are subject to 1-in-N sampling; see rcclTelemetryPostTs().    */
/* The counters are not.                                               */
/*                                                                     */
/* 4. ERROR PATH (on CQ error):                                        */
/*    - Increment cq_errors counter                                    */
/*                                                                     */
/* FEATURE-FLAG RULE — the single convention every call site obeys:    */
/* 1. Every telemetry entry point is self-guarding. It tests           */
/*    rcclTelemetryOn() (directly, or through rcclTelemetryChannel()), */
/*    or validates the resolved handle it was given, and validates its */
/*    own indices, so a negative or out-of-range devIdx/chIdx/qpIdx,   */
/*    or a NULL handle, is a silent no-op. In particular, an untracked */
/*    QP carries a NULL handle and needs no caller check.              */
/* 2. Call sites therefore never test rcclTelemetryEnabled, and never  */
/*    test a handle for NULL, in order to decide whether to call a     */
/*    hook.                                                            */
/* 3. The one permitted call-site guard is rcclTelemetryOn() wrapped   */
/*    around work that exists solely to produce hook arguments         */
/*    (timestamps, QP lookups, per-device slot assignment). That is an */
/*    optimization for the disabled case, never a correctness check.   */
/* 4. Resolved handles (see "resolved slot handles" below) do not bend */
/*    any of the above. Obtaining a handle IS the guarded step: the    */
/*    resolver tests rcclTelemetryOn() and the indices exactly as the  */
/*    index-taking entry points do, and returns NULL when telemetry is */
/*    off or the slot does not exist. Every hook that takes a handle   */
/*    is then a silent no-op on NULL, which is why rule 2 forbids the  */
/*    caller from checking. A handle is obtained once, on the setup    */
/*    path that rule 3 already covers, and stored next to the object   */
/*    it belongs to.                                                   */
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

/* Block b holds base << b entries; valid only below the capacity. */
static inline void rcclTelemetryBlockIndex(int idx, int block0Log2, unsigned int* block, unsigned int* offset) {
  unsigned int base = 1u << block0Log2;
  unsigned int b = 31u - (unsigned int)__builtin_clz(((unsigned int)idx >> block0Log2) + 1u);
  *block = b;
  *offset = (unsigned int)idx - (base << b) + base;
}

/* Returns NULL when telemetry is off or the channel has no storage. */
static inline RcclChannelStats* rcclTelemetryChannel(int devIdx, int chIdx) {
  if (!rcclTelemetryOn() || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return NULL;
  RcclDeviceStats* dev = &rcclTelemetryDevs[devIdx];
  int capacity = __atomic_load_n(&dev->channel_capacity, __ATOMIC_ACQUIRE);
  if (chIdx < 0 || chIdx >= capacity) return NULL;
  unsigned int block, offset;
  rcclTelemetryBlockIndex(chIdx, dev->channel_block0_log2, &block, &offset);
  return &((RcclChannelStats*)dev->channel_blocks[block])[offset];
}

/* Setup path only; counter updates must use rcclTelemetryQp(). */
static inline RcclQpStats* rcclTelemetryQpSlot(RcclChannelStats* ch, int qpIdx) {
  int capacity = __atomic_load_n(&ch->qp_capacity, __ATOMIC_ACQUIRE);
  if (qpIdx < 0 || qpIdx >= capacity) return NULL;
  unsigned int block, offset;
  rcclTelemetryBlockIndex(qpIdx, ch->qp_block0_log2, &block, &offset);
  return &((RcclQpStats*)ch->qp_blocks[block])[offset];
}

/* Bounded by num_qps, not capacity: the last block's tail is unused. */
static inline RcclQpStats* rcclTelemetryQp(RcclChannelStats* ch, int qpIdx) {
  int numQps = __atomic_load_n(&ch->num_qps, __ATOMIC_ACQUIRE);
  if (qpIdx < 0 || qpIdx >= numQps) return NULL;
  unsigned int block, offset;
  rcclTelemetryBlockIndex(qpIdx, ch->qp_block0_log2, &block, &offset);
  return &((RcclQpStats*)ch->qp_blocks[block])[offset];
}

/*
 * Channel-level WQE/CTS totals, derived from the channel's QP slots.
 *
 * These five counters used to be maintained on the hot path as a second
 * __atomic_fetch_add next to the per-QP one. Because every QP on a channel
 * shares one RcclChannelStats cache line, that turned each per-WQE hook into a
 * contended line acquisition and capped hook throughput at a fixed rate no
 * matter how many threads were running. They are exact sums over the QP slots,
 * so nothing is lost by computing them here instead, on the flush/JSON path.
 *
 * Consistency: every counter is read with a relaxed atomic load, bounded by the
 * same num_qps acquire load the hot path uses, so a concurrent hook can never
 * make this read a slot that does not exist. A total may straddle concurrent
 * increments (it is a sum of independently-updated counters, not a snapshot),
 * exactly as the old channel counter could be read mid-increment. At flush time
 * the data path is quiescent, so the reported values are exact.
 */
typedef struct {
  uint64_t num_wqe_sent;
  uint64_t num_recv_wqe;
  uint64_t num_wqe_rcvd;
  uint64_t num_wqe_completed;
  uint64_t num_wqe_sampled;
  uint64_t num_cts_sent;
} RcclChannelAggregate;

/*
 * Completions on this QP whose latency was actually measured.
 *
 * Not a stored counter, and deliberately so: it is the sum of the QP's latency
 * histogram, which is by construction exactly the set of completions that
 * produced a latency sample. Deriving it makes the "bucket sum == sampled
 * count" invariant true by definition rather than by agreement between two
 * counters, and it keeps the sampled path from growing an extra atomic that
 * the default N == 1 would then pay on every single completion.
 *
 * Summing all RCCL_TELEMETRY_HISTOGRAM_SIZE buckets is the same as summing
 * only the histogram_max_buckets that are emitted, because
 * rcclTelemetryLatencyBucket() clamps every index below that bound.
 */
static inline uint64_t rcclTelemetryQpSampledCount(RcclQpStats* qp) {
  uint64_t n = 0;
  if (qp == NULL) return 0;
  for (int b = 0; b < RCCL_TELEMETRY_HISTOGRAM_SIZE; b++)
    n += __atomic_load_n(&qp->wqe_completion_histogram[b], __ATOMIC_RELAXED);
  return n;
}

static inline void rcclTelemetryChannelAggregate(RcclChannelStats* ch, RcclChannelAggregate* agg) {
  agg->num_wqe_sent = agg->num_recv_wqe = agg->num_wqe_rcvd = 0;
  agg->num_wqe_completed = agg->num_wqe_sampled = agg->num_cts_sent = 0;
  if (ch == NULL) return;
  int numQps = __atomic_load_n(&ch->num_qps, __ATOMIC_ACQUIRE);
  for (int q = 0; q < numQps; q++) {
    RcclQpStats* qp = rcclTelemetryQp(ch, q);
    if (qp == NULL) break;
    agg->num_wqe_sent += __atomic_load_n(&qp->num_wqe_sent, __ATOMIC_RELAXED);
    agg->num_recv_wqe += __atomic_load_n(&qp->num_recv_wqe, __ATOMIC_RELAXED);
    agg->num_wqe_rcvd += __atomic_load_n(&qp->num_wqe_rcvd, __ATOMIC_RELAXED);
    agg->num_wqe_completed += __atomic_load_n(&qp->num_wqe_completed, __ATOMIC_RELAXED);
    agg->num_wqe_sampled += rcclTelemetryQpSampledCount(qp);
    agg->num_cts_sent += __atomic_load_n(&qp->num_cts_sent, __ATOMIC_RELAXED);
  }
}

/* ------------------------------------------------------------------ */
/* Resolved slot handles                                              */
/* ------------------------------------------------------------------ */

/*
 * A single posted WQE used to drive three or four separate telemetry calls
 * (posting, opcode, size, CTS-ness), and each of them re-walked the same path
 * to the same slot: acquire-load a capacity, a clz, a dependent load into a
 * block array, twice over — channel, then QP. That repeated resolution was
 * shared by every hook family, and so showed up in all of their measured costs
 * at once.
 *
 * A slot's address never changes: blocks are appended, never freed or moved,
 * and a slot is only handed out after its block exists. So the resolution can
 * be done once, when the QP is registered, and the resulting pointer stored on
 * the QP. Every per-WQE hook then takes that pointer and does nothing but its
 * counter updates.
 *
 * Publication ordering needs nothing new. A handle is produced on the
 * connection-setup path, whose result — the whole comm — is already published
 * to the data-path threads by the existing setup/handoff synchronization, the
 * same reason those threads may read qp->qp. The resolver's acquire loads still
 * pair with the release stores in the allocator, so the slot a handle names is
 * fully constructed.
 *
 * Telemetry cannot be switched on after init (rcclTelemetryEnabled is published
 * once, before any connection is made), so a handle that is NULL because
 * telemetry was off can never become stale.
 */

/*
 * Resolve the statistics slot of a QP registered by
 * rcclTelemetrySetupChannel(). Returns NULL when telemetry is off, or when the
 * indices name no slot — in which case the QP is untracked and every hook below
 * is a no-op for it.
 */
static inline RcclQpStats* rcclTelemetryResolveQp(int devIdx, int chIdx, int qpIdx) {
  RcclChannelStats* ch = rcclTelemetryChannel(devIdx, chIdx);
  if (ch == NULL) return NULL;
  return rcclTelemetryQp(ch, qpIdx);
}

/*
 * Resolve the statistics slot of a channel. Same contract as above; this is
 * just rcclTelemetryChannel() under the name the handle callers use.
 */
static inline RcclChannelStats* rcclTelemetryResolveChannel(int devIdx, int chIdx) {
  return rcclTelemetryChannel(devIdx, chIdx);
}

/*
 * Post timestamp for one WQE about to be posted on `qp`: either a real
 * CLOCK_MONOTONIC reading, or RCCL_TELEMETRY_TS_UNSAMPLED for a WQE this
 * sampling interval skips. Call it on the post path, in place of the bare
 * rcclTelemetryGetNs() it replaced, under the same rcclTelemetryOn() guard.
 *
 * At the default N == 1 this is a load of a never-written global, a compare and
 * a perfectly-predicted branch ahead of the clock read that used to be
 * unconditional, and nothing else: no counter is touched, so the default path
 * is what it always was.
 *
 * Above N == 1 the selection rides a per-QP sequence counter. Two alternatives
 * were rejected. The value returned by the existing __atomic_fetch_add on
 * num_wqe_sent is free, but it is not available in time: that counter is
 * incremented after ibv_post_send, while the timestamp has to be taken before
 * it, and moving the increment would change what the counter means when a post
 * fails. A thread_local counter needs no memory of its own, but in a shared
 * library it defaults to the global-dynamic TLS model and reaches the variable
 * through a __tls_get_addr call; forcing initial-exec to avoid that risks
 * failing to load under dlopen once the static TLS surplus is gone.
 *
 * Measured cost of this decision, per call, in a dlopen'd shared object at
 * N == 16 (median of 9 trials of 20M calls each, MI300X node): per-QP counter
 * 3.8 ns, thread_local global-dynamic 5.0 ns, thread_local initial-exec
 * 3.4 ns, __atomic_fetch_add on the same counter 9.4 ns. So the per-QP counter
 * costs 0.4 ns more than the TLS model that cannot safely be used and 1.2 ns
 * less than the one that can, and an atomic read-modify-write would cost more
 * than twice as much. All of them replace an unconditional clock read of
 * 18.9 ns. At N == 1 every variant measures 18.3-18.5 ns, i.e. the clock read
 * and nothing else.
 *
 * The counter is read and written with relaxed atomics rather than an atomic
 * read-modify-write: a lost update between two threads driving one QP only
 * shifts which postings are sampled, never a reported count, so nothing on the
 * hot path needs a lock prefix. Relaxed atomic accesses are still race-free,
 * so this stays clean under ThreadSanitizer.
 *
 * qp may be NULL (an untracked QP); the completion hook is a no-op for it, so
 * there is no reason to spend a clock read on it.
 */
static inline int64_t rcclTelemetryPostTs(RcclQpStats* qp) {
  uint64_t mask = rcclTelemetryLatencySampleMask;
  if (__builtin_expect(mask == 0, 1)) return rcclTelemetryGetNs();
  if (qp == NULL) return RCCL_TELEMETRY_TS_UNSAMPLED;
  uint64_t seq = __atomic_load_n(&qp->latency_sample_seq, __ATOMIC_RELAXED) + 1;
  __atomic_store_n(&qp->latency_sample_seq, seq, __ATOMIC_RELAXED);
  if ((seq & mask) != 0) return RCCL_TELEMETRY_TS_UNSAMPLED;
  return rcclTelemetryGetNs();
}

/*
 * Histogram bucket for a completion latency.
 *
 * Bit-identical to what it replaced,
 *
 *   int bucket = (int)(latency_ns / rcclTelemetryCfg.histogram_bucket_interval_ns);
 *   if (bucket >= max_buckets) bucket = max_buckets - 1;
 *
 * for every latency_ns, including the int narrowing, but without the 64-bit
 * division that used to sit on every completion.
 *
 * Why the multiply is exact. Let d = histogram_bucket_interval_ns >= 1 and
 * m = floor(2^63/d) + 1, so m*d = 2^63 + e with 1 <= e <= d. For n >= 0,
 *
 *   n*m / 2^63 = n/d + n*e/(d * 2^63)
 *
 * and writing n = k*d + s with 0 <= s < d, floor(n*m / 2^63) == k exactly when
 * s + n*e/2^63 < d. Since s <= d-1 it is enough that n*e < 2^63, and since
 * e <= d it is enough that n*d < 2^63. So the multiply is exact for every
 * n < floor(2^63/d) == histogram_recip_max_ns, which is the bound tested here:
 * at n = recip_max_ns - 1 we get n*d <= 2^63 - d < 2^63.
 *
 * Above that bound — 3.5 days of latency at the default 30 us interval — the
 * division still runs, so the identity holds over the whole int64 range rather
 * than only over a plausible one. histogram_recip == 0 (config not derived)
 * takes the same path.
 */
static inline int rcclTelemetryLatencyBucket(int64_t latency_ns) {
  int64_t q;
  uint64_t recip = rcclTelemetryCfg.histogram_recip;
  if (__builtin_expect(recip != 0 && (uint64_t)latency_ns < rcclTelemetryCfg.histogram_recip_max_ns, 1)) {
    q = (int64_t)(uint64_t)(((__uint128_t)(uint64_t)latency_ns * recip) >> RCCL_TELEMETRY_RECIP_SHIFT);
  } else {
    q = latency_ns / rcclTelemetryCfg.histogram_bucket_interval_ns;
  }
  int bucket = (int)q;
  if (bucket >= rcclTelemetryCfg.histogram_max_buckets) bucket = rcclTelemetryCfg.histogram_max_buckets - 1;
  return bucket;
}

/**
 * Record a completion drained from the CQ against a resolved QP slot, with
 * latency tracking. See rcclTelemetryWqeComplete() for the counter semantics.
 */
static inline void rcclTelemetryQpWqeComplete(RcclQpStats* qp, int64_t postTs) {
  if (qp == NULL) return;

  __atomic_fetch_add(&qp->num_wqe_rcvd, 1, __ATOMIC_RELAXED);
  if (postTs <= 0) return;

  __atomic_fetch_add(&qp->num_wqe_completed, 1, __ATOMIC_RELAXED);
  /* Matched, so it counts; not sampled, so it carries no latency. Placed above
   * the clock read on purpose: an unsampled WQE costs no clock read on either
   * side. */
  if (postTs == RCCL_TELEMETRY_TS_UNSAMPLED) return;

  int64_t latency_ns = rcclTelemetryGetNs() - postTs;
  if (latency_ns <= 0) return;

  int bucket = rcclTelemetryLatencyBucket(latency_ns);
  if (bucket >= 0 && bucket < RCCL_TELEMETRY_HISTOGRAM_SIZE)
    __atomic_fetch_add(&qp->wqe_completion_histogram[bucket], 1, __ATOMIC_RELAXED);

  int64_t cur_min = __atomic_load_n(&qp->wqe_completion_ns_min, __ATOMIC_RELAXED);
  while (cur_min == 0 || latency_ns < cur_min) {
    if (__atomic_compare_exchange_n(&qp->wqe_completion_ns_min, &cur_min, latency_ns, 1, __ATOMIC_RELAXED,
                                    __ATOMIC_RELAXED))
      break;
  }

  int64_t cur_max = __atomic_load_n(&qp->wqe_completion_ns_max, __ATOMIC_RELAXED);
  while (latency_ns > cur_max) {
    if (__atomic_compare_exchange_n(&qp->wqe_completion_ns_max, &cur_max, latency_ns, 1, __ATOMIC_RELAXED,
                                    __ATOMIC_RELAXED))
      break;
  }
}

/**
 * Record everything one posted send WQE contributes to its QP: the posting
 * itself (num_wqe_sent) and its opcode (num_write_wqe or num_write_imm_wqe).
 *
 * This is the combined per-WQE entry point the send path uses. It exists so
 * that one posting costs one slot resolution and one branch instead of two of
 * each; the counters it touches, and their values, are exactly those of
 * rcclTelemetryWqePosted(..., 1) followed by rcclTelemetryWriteWqe(..., withImm).
 *
 * @param withImm  1 for RDMA_WRITE_WITH_IMM, 0 for plain RDMA_WRITE
 */
static inline void rcclTelemetryQpSendPosted(RcclQpStats* qp, int withImm) {
  if (qp == NULL) return;
  __atomic_fetch_add(&qp->num_wqe_sent, 1, __ATOMIC_RELAXED);
  __atomic_fetch_add(withImm ? &qp->num_write_imm_wqe : &qp->num_write_wqe, 1, __ATOMIC_RELAXED);
}

/**
 * Record a posted receive WQE (num_recv_wqe) against a resolved QP slot.
 */
static inline void rcclTelemetryQpRecvPosted(RcclQpStats* qp) {
  if (qp == NULL) return;
  __atomic_fetch_add(&qp->num_recv_wqe, 1, __ATOMIC_RELAXED);
}

/**
 * Record a CTS posting against a resolved QP slot.
 *
 * @param signalled  1 if the work request carried IBV_SEND_SIGNALED
 */
static inline void rcclTelemetryQpCtsSent(RcclQpStats* qp, int signalled) {
  if (qp == NULL) return;
  __atomic_fetch_add(&qp->num_cts_sent, 1, __ATOMIC_RELAXED);
  if (signalled) __atomic_fetch_add(&qp->num_cts_sent_signalled, 1, __ATOMIC_RELAXED);
  else __atomic_fetch_add(&qp->num_cts_sent_unsignalled, 1, __ATOMIC_RELAXED);
}

/**
 * Record a CTS FIFO slot miss against a resolved QP slot.
 */
static inline void rcclTelemetryQpSlotMiss(RcclQpStats* qp) {
  if (qp == NULL) return;
  __atomic_fetch_add(&qp->num_slot_miss, 1, __ATOMIC_RELAXED);
}

/**
 * Record a completed network request against a resolved channel slot.
 * See rcclTelemetryRequestCompleted() for what this counter is and is not.
 */
static inline void rcclTelemetryChRequestCompleted(RcclChannelStats* ch) {
  if (ch == NULL) return;
  __atomic_fetch_add(&ch->num_req_completed, 1, __ATOMIC_RELAXED);
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
  if (!rcclTelemetryOn() || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return;
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
  if (!rcclTelemetryOn() || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return;
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
 * - Computes latency from post_ts to now for matched, sampled completions
 * - Updates the latency histogram bucket
 * - Updates min/max latency atomically
 *
 * num_wqe_completed is independent of the sampling interval: a WQE the
 * interval skipped still arrives here with postTs == RCCL_TELEMETRY_TS_UNSAMPLED
 * and is still counted. Only the histogram and min/max shrink to the sample.
 *
 * Both counters are per-QP only. The matching channel totals are derived from
 * the QP slots by rcclTelemetryChannelAggregate() at flush time, so a channel
 * total can never include a completion that no QP slot accounts for.
 *
 * The index-taking form below resolves the slot and defers to
 * rcclTelemetryQpWqeComplete(). The data path holds a resolved handle and calls
 * that one directly; this form is for callers that only have indices.
 *
 * @param devIdx   Device index in rcclTelemetryDevs array
 * @param chIdx    Channel index
 * @param qpIdx    Queue pair index within channel
 * @param postTs   Timestamp when the work request was posted: 0 if the
 *                 completion matched no tracked posting,
 *                 RCCL_TELEMETRY_TS_UNSAMPLED if it matched one that latency
 *                 sampling skipped, a real reading otherwise
 */
static inline void rcclTelemetryWqeComplete(int devIdx, int chIdx, int qpIdx, int64_t postTs) {
  rcclTelemetryQpWqeComplete(rcclTelemetryResolveQp(devIdx, chIdx, qpIdx), postTs);
}

/**
 * Increment the QP-level posted-WQE counter: num_wqe_sent for sends,
 * num_recv_wqe for receives. Call after ibv_post_send / ibv_post_recv.
 *
 * The channel totals of the same name are derived from the QP slots at flush
 * time by rcclTelemetryChannelAggregate(), not incremented here.
 *
 * @param devIdx   Device index
 * @param chIdx    Channel index
 * @param qpIdx    QP index within channel
 * @param isSend   1 for send, 0 for recv
 */
static inline void rcclTelemetryWqePosted(int devIdx, int chIdx, int qpIdx, int isSend) {
  RcclQpStats* qp = rcclTelemetryResolveQp(devIdx, chIdx, qpIdx);
  if (qp == NULL) return;
  if (isSend) {
    __atomic_fetch_add(&qp->num_wqe_sent, 1, __ATOMIC_RELAXED);
  } else {
    __atomic_fetch_add(&qp->num_recv_wqe, 1, __ATOMIC_RELAXED);
  }
}

/**
 * Record a completed network request on a channel. Call once per request, when
 * its last outstanding event has been accounted for.
 *
 * This is a request count, not a WQE count: it is not a sum over the per-QP
 * num_wqe_* counters and must not be used as if it were. It is therefore the
 * one channel counter that is still stored rather than derived at flush.
 *
 * @param devIdx   Device index
 * @param chIdx    Channel index
 */
static inline void rcclTelemetryRequestCompleted(int devIdx, int chIdx) {
  rcclTelemetryChRequestCompleted(rcclTelemetryResolveChannel(devIdx, chIdx));
}

/**
 * Record a CTS FIFO slot miss.
 * Call when the sender finds the expected CTS slot not yet published and
 * returns without posting.
 */
static inline void rcclTelemetrySlotMiss(int devIdx, int chIdx, int qpIdx) {
  rcclTelemetryQpSlotMiss(rcclTelemetryResolveQp(devIdx, chIdx, qpIdx));
}

/**
 * Record a CTS posting on a CTS QP.
 * Call after ibv_post_send succeeds on the CTS path.
 *
 * QP-level only; the channel's num_cts_sent is derived at flush time.
 *
 * @param signalled  1 if the work request carried IBV_SEND_SIGNALED
 */
static inline void rcclTelemetryCtsSent(int devIdx, int chIdx, int qpIdx, int signalled) {
  rcclTelemetryQpCtsSent(rcclTelemetryResolveQp(devIdx, chIdx, qpIdx), signalled);
}

/**
 * Record an RDMA_WRITE (withImm=0) or RDMA_WRITE_WITH_IMM (withImm=1) posting.
 *
 * The send path does not call this: it posts through
 * rcclTelemetryQpSendPosted(), which folds this counter in.
 */
static inline void rcclTelemetryWriteWqe(int devIdx, int chIdx, int qpIdx, int withImm) {
  RcclQpStats* qp = rcclTelemetryResolveQp(devIdx, chIdx, qpIdx);
  if (qp == NULL) return;
  __atomic_fetch_add(withImm ? &qp->num_write_imm_wqe : &qp->num_write_wqe, 1, __ATOMIC_RELAXED);
}

/**
 * Fold a comm's accumulated ibv_poll_cq count into its device at teardown.
 *
 * cq_poll_count is bumped on every poll, including the empty ones, and has no
 * per-device meaning on the data path, so the hot path keeps a plain per-comm
 * counter (ncclIbNetCommBase::telCqPollCount) and only this rare teardown call
 * touches the shared device line.
 */
static inline void rcclTelemetryAddCqPolls(int devIdx, uint64_t polls) {
  if (!rcclTelemetryOn() || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS || polls == 0) return;
  __atomic_fetch_add(&rcclTelemetryDevs[devIdx].cq_poll_count, polls, __ATOMIC_RELAXED);
}

/**
 * Record the payload size of a posted WQE into the per-device power-of-two
 * size distribution.
 */
static inline void rcclTelemetryWqeSize(int devIdx, uint64_t bytes) {
  if (!rcclTelemetryOn() || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return;
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
 * QPs that got no slot must keep a NULL handle, so nothing charges them.
 *
 * @param devIdx     Device index in rcclTelemetryDevs array
 * @param chIdx      Channel index (typically allocated via atomic counter)
 * @param numQps     Number of QPs the caller wants to register on this channel
 * @param numSlots   Out: number of slots actually reserved, 0 if none
 * @return           First reserved QP slot index, or -1 if no slot was reserved
 *
 * Usage (in ncclIbConnect/ncclIbAccept). This is the one setup path that rule 3
 * above applies to, hence the rcclTelemetryOn() wrapper, and it is also where
 * rule 4's one-time handle resolution happens:
 *   if (rcclTelemetryOn()) {
 *     int numSlots = 0;
 *     int qpSlot = rcclTelemetrySetupChannel(devIdx, chIdx, numQps, &numSlots);
 *     for (int q = 0; q < numSlots; q++) {
 *       rcclTelemetrySetQpRole(devIdx, chIdx, qpSlot + q, comm->base.qps[q].isDataQp);
 *       comm->base.qps[q].telQpStats = rcclTelemetryResolveQp(devIdx, chIdx, qpSlot + q);
 *     }
 *   }
 */
int rcclTelemetrySetupChannel(int devIdx, int chIdx, int numQps, int* numSlots);

/**
 * Mark a device's channel ids as not real NCCL channel ids.
 *
 * Call from a transport that cannot supply a channel id (e.g. mlx5/thor2, where
 * the id only reaches the AINIC path): every QP then shares one device-wide
 * bucket, and flush reports num_channels as null so the single bucket is not
 * mistaken for a real per-channel breakdown.
 */
static inline void rcclTelemetryMarkChannelsUnknown(int devIdx) {
  if (!rcclTelemetryOn() || devIdx < 0 || devIdx >= RCCL_TELEMETRY_MAX_DEVS) return;
  __atomic_store_n(&rcclTelemetryDevs[devIdx].channels_unknown, 1, __ATOMIC_RELAXED);
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
  RcclChannelStats* ch = rcclTelemetryChannel(devIdx, chIdx);
  if (ch == NULL) return;
  RcclQpStats* qp = rcclTelemetryQp(ch, qpIdx);
  if (qp == NULL) return;
  qp->is_data_qp = isDataQp ? 1 : 0;
  if (isDataQp)
    __atomic_fetch_add(&ch->num_data_qp, 1, __ATOMIC_RELAXED);
  else
    __atomic_fetch_add(&ch->num_cts_qp, 1, __ATOMIC_RELAXED);
}

#ifdef __cplusplus
}
#endif

#endif /* RCCL_NET_TELEMETRY_H_ */
