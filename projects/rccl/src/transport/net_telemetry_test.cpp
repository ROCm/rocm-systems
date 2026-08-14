/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Standalone unit tests for the net_telemetry on-demand block allocator, the
 * "channel total == sum over its QPs" invariant, the resolved slot handles the
 * hot path uses, and the division-free histogram bucket index.
 *
 * The channel WQE/CTS totals are not stored: they are derived from the QP slots
 * by rcclTelemetryChannelAggregate() on the flush/JSON path. The invariant is
 * therefore what makes the derived totals trustworthy, and these tests check it
 * two ways: the aggregate must match an independent walk of the QP slots, and
 * in the single-threaded case it must also match the analytically known number
 * of hook calls the test issued.
 *
 * The hot path no longer resolves a slot per counter update: it resolves once
 * per QP at connection setup and keeps the pointer. Two things must therefore
 * hold, and are tested here: a handle-driven update must be indistinguishable
 * from the index-driven one it replaced, and a handle must stay valid while the
 * block chain grows underneath it.
 *
 * The completion hook reaches its histogram bucket with a multiply instead of a
 * 64-bit division. testLatencyBucket() is the proof obligation for that: the
 * bucket index, and the underlying quotient, must be bit-identical to the
 * division for every input, which is checked exhaustively at small magnitudes
 * and by wide randomized sampling over the whole int64 range.
 *
 * net_telemetry.cc depends only on libc + pthread, so this links directly
 * against it with no RCCL build system and no hardware:
 *
 *   g++ -std=c++17 -O2 -pthread -Wall -Wextra \
 *       -I ../include net_telemetry_test.cpp net_telemetry.cc \
 *       -o /tmp/net_telemetry_test && /tmp/net_telemetry_test
 *
 * Concurrency case is meant to be run under sanitizers as well:
 *
 *   g++ -std=c++17 -O1 -g -pthread -fsanitize=thread \
 *       -I ../include net_telemetry_test.cpp net_telemetry.cc \
 *       -o /tmp/net_telemetry_test_tsan && /tmp/net_telemetry_test_tsan
 *
 *   g++ -std=c++17 -O1 -g -pthread -fsanitize=address,undefined \
 *       -I ../include net_telemetry_test.cpp net_telemetry.cc \
 *       -o /tmp/net_telemetry_test_asan && /tmp/net_telemetry_test_asan
 *
 * The allocator never frees its blocks, so LeakSanitizer reporting them at exit
 * under ASan is expected and is not a defect.
 */

#include "net_telemetry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <thread>
#include <vector>
#include <atomic>

extern int rcclTelemetryEnabled;
extern RcclTelemetryConfig rcclTelemetryCfg;
extern RcclDeviceStats rcclTelemetryDevs[RCCL_TELEMETRY_MAX_DEVS];

static int g_failures = 0;

#define CHECK(cond) \
  do { \
    if (!(cond)) { \
      std::printf("  FAIL: %s  (line %d)\n", #cond, __LINE__); \
      g_failures++; \
    } \
  } while (0)

#define CHECK_EQ(a, b) \
  do { \
    long long va = (long long)(a), vb = (long long)(b); \
    if (va != vb) { \
      std::printf("  FAIL: %s (%lld) != %s (%lld)  (line %d)\n", #a, va, #b, vb, __LINE__); \
      g_failures++; \
    } \
  } while (0)

/* Unsigned form: the reciprocal checks run up to 2^63, which CHECK_EQ's
 * long long would render as a negative number. */
#define CHECK_EQ_U(a, b) \
  do { \
    unsigned long long va = (unsigned long long)(a); \
    unsigned long long vb = (unsigned long long)(b); \
    if (va != vb) { \
      std::printf("  FAIL: %s (%llu) != %s (%llu)  (line %d)\n", #a, va, #b, vb, __LINE__); \
      g_failures++; \
    } \
  } while (0)

/* Fresh telemetry state. Blocks already allocated leak; that is fine for a
 * short-lived test and mirrors the process-lifetime model of the allocator. */
static void resetState(void) {
  std::memset(rcclTelemetryDevs, 0, sizeof(rcclTelemetryDevs));
  rcclTelemetryCfg.histogram_max_buckets = 5;
  rcclTelemetryCfg.histogram_bucket_interval_ns = 30000;
  rcclTelemetryConfigDeriveHistogram();
  rcclTelemetryEnabled = 1;
}

/* ---- block index decomposition -------------------------------------- */
static void testBlockIndex(void) {
  std::printf("testBlockIndex\n");
  for (int b0 = RCCL_TELEMETRY_BLOCK0_MIN_LOG2; b0 <= 8; b0++) {
    unsigned int base = 1u << b0;
    for (int idx = 0; idx < 200000; idx++) {
      unsigned int block = 0, offset = 0;
      rcclTelemetryBlockIndex(idx, b0, &block, &offset);
      unsigned int blockEntries = base << block;
      unsigned int blockStart = (base << block) - base;
      CHECK(offset < blockEntries);
      CHECK_EQ(blockStart + offset, (unsigned int)idx);
      if (g_failures) return;
    }
  }
}

/* ---- growth across block boundaries --------------------------------- */
static void testGrowth(void) {
  std::printf("testGrowth\n");
  resetState();
  int n = 0;
  int start = rcclTelemetrySetupChannel(0, 0, 3, &n);
  CHECK_EQ(start, 0);
  CHECK_EQ(n, 3);

  start = rcclTelemetrySetupChannel(0, 0, 5, &n);
  CHECK_EQ(start, 3);
  CHECK_EQ(n, 5);

  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  CHECK(ch != nullptr);
  CHECK_EQ(ch->num_qps, 8);
  for (int q = 0; q < 8; q++) {
    RcclQpStats* qp = rcclTelemetryQp(ch, q);
    CHECK(qp != nullptr);
    if (qp) CHECK_EQ(qp->id, q);
  }
  CHECK(rcclTelemetryQp(ch, 8) == nullptr);
  CHECK_EQ(ch->num_qp_untracked, 0);
}

/* ---- the old 128-QP ceiling is gone --------------------------------- */
static void testOverOldCeiling(void) {
  std::printf("testOverOldCeiling\n");
  resetState();
  const int N = 300;
  int n = 0;
  int start = rcclTelemetrySetupChannel(0, 0, N, &n);
  CHECK_EQ(start, 0);
  CHECK_EQ(n, N);

  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  CHECK(ch != nullptr);
  CHECK_EQ(ch->num_qps, N);
  CHECK_EQ(ch->num_qp_untracked, 0);
  CHECK_EQ(rcclTelemetryDevs[0].num_qp_untracked, 0);

  rcclTelemetryWqeComplete(0, 0, 250, 0);
  CHECK_EQ(rcclTelemetryQp(ch, 250)->num_wqe_rcvd, 1);

  /* The derived total has to find a slot this far into the block chain. */
  RcclChannelAggregate agg;
  rcclTelemetryChannelAggregate(ch, &agg);
  CHECK_EQ(agg.num_wqe_rcvd, 1);
}

/* ---- sanity bound on the channel index ------------------------------ */
static void testChannelSanityBound(void) {
  std::printf("testChannelSanityBound\n");
  resetState();
  int n = -1;
  int start = rcclTelemetrySetupChannel(0, RCCL_TELEMETRY_MAX_CHANNELS, 4, &n);
  CHECK_EQ(start, -1);
  CHECK_EQ(n, 0);
  CHECK_EQ(rcclTelemetryDevs[0].num_qp_untracked, 4);

  start = rcclTelemetrySetupChannel(0, -1, 2, &n);
  CHECK_EQ(start, -1);
  CHECK_EQ(rcclTelemetryDevs[0].num_qp_untracked, 6);
}

/* ---- channel counter == sum over QPs (single threaded) -------------- */
static void driveHooks(int dev, int ch, int qp, int nSend, int nRecv, int nComplete, int nCts) {
  for (int i = 0; i < nSend; i++) rcclTelemetryWqePosted(dev, ch, qp, 1);
  for (int i = 0; i < nRecv; i++) rcclTelemetryWqePosted(dev, ch, qp, 0);
  for (int i = 0; i < nComplete; i++) rcclTelemetryWqeComplete(dev, ch, qp, 0);
  for (int i = 0; i < nCts; i++) rcclTelemetryCtsSent(dev, ch, qp, i & 1);
}

/* The aggregation path that produces the channel numbers in the JSON must agree
 * with an independent walk of the same QP slots. */
static void checkInvariant(RcclChannelStats* ch) {
  uint64_t sSent = 0, sRecv = 0, sRcvd = 0, sComp = 0, sCts = 0;
  uint64_t sSig = 0, sUnsig = 0;
  for (int q = 0; q < ch->num_qps; q++) {
    RcclQpStats* qp = rcclTelemetryQp(ch, q);
    sSent += qp->num_wqe_sent;
    sRecv += qp->num_recv_wqe;
    sRcvd += qp->num_wqe_rcvd;
    sComp += qp->num_wqe_completed;
    sCts += qp->num_cts_sent;
    sSig += qp->num_cts_sent_signalled;
    sUnsig += qp->num_cts_sent_unsignalled;
  }
  RcclChannelAggregate agg;
  rcclTelemetryChannelAggregate(ch, &agg);
  CHECK_EQ(agg.num_wqe_sent, sSent);
  CHECK_EQ(agg.num_recv_wqe, sRecv);
  CHECK_EQ(agg.num_wqe_rcvd, sRcvd);
  CHECK_EQ(agg.num_wqe_completed, sComp);
  CHECK_EQ(agg.num_cts_sent, sCts);
  CHECK_EQ(sSig + sUnsig, sCts);
}

static void testInvariantSingle(void) {
  std::printf("testInvariantSingle\n");
  resetState();
  const int N = 40;
  int n = 0;
  rcclTelemetrySetupChannel(0, 0, N, &n);
  CHECK_EQ(n, N);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  uint64_t eSent = 0, eRecv = 0, eRcvd = 0, eCts = 0;
  for (int q = 0; q < N; q++) {
    rcclTelemetrySetQpRole(0, 0, q, (q % 4) != 0);
    driveHooks(0, 0, q, q + 1, q, 2 * q, (q % 3));
    eSent += (uint64_t)(q + 1);
    eRecv += (uint64_t)q;
    eRcvd += (uint64_t)(2 * q);
    eCts += (uint64_t)(q % 3);
  }
  checkInvariant(ch);

  /* Independent of the QP slots: the totals the driver actually issued. Catches
   * an aggregation that is self-consistent but walks the wrong slots. */
  RcclChannelAggregate agg;
  rcclTelemetryChannelAggregate(ch, &agg);
  CHECK_EQ(agg.num_wqe_sent, eSent);
  CHECK_EQ(agg.num_recv_wqe, eRecv);
  CHECK_EQ(agg.num_wqe_rcvd, eRcvd);
  CHECK_EQ(agg.num_cts_sent, eCts);
  /* driveHooks passes postTs == 0, so no completion is ever matched. */
  CHECK_EQ(agg.num_wqe_completed, 0);
}

/* The derived totals must also hold across the block-growth boundaries, where
 * the QP slots of one channel live in several separately allocated blocks. */
static void testAggregateAcrossBlocks(void) {
  std::printf("testAggregateAcrossBlocks\n");
  resetState();
  const int N = 200;
  uint64_t expected = 0;
  for (int q = 0; q < N; q++) {
    int n = 0;
    rcclTelemetrySetupChannel(0, 0, 1, &n); /* one slot at a time -> many blocks */
    CHECK_EQ(n, 1);
    rcclTelemetryWqePosted(0, 0, q, 1);
    expected++;
  }
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  CHECK_EQ(ch->num_qps, N);
  RcclChannelAggregate agg;
  rcclTelemetryChannelAggregate(ch, &agg);
  CHECK_EQ(agg.num_wqe_sent, expected);
  checkInvariant(ch);
}

/* Every hot-path slot must sit on its own cache line, or the contention the
 * derived-aggregate change removed comes back as false sharing between QPs. */
static void testCacheLineAlignment(void) {
  std::printf("testCacheLineAlignment\n");
  resetState();
  CHECK_EQ(sizeof(RcclQpStats) % RCCL_TELEMETRY_CACHELINE, 0);
  CHECK_EQ(sizeof(RcclChannelStats) % RCCL_TELEMETRY_CACHELINE, 0);
  /* The one data-path-written channel field must not share the line the hooks
   * read to find a QP slot. */
  CHECK_EQ(offsetof(RcclChannelStats, num_req_completed) % RCCL_TELEMETRY_CACHELINE, 0);
  CHECK(offsetof(RcclChannelStats, num_req_completed) >= RCCL_TELEMETRY_CACHELINE);
  const int N = 40;
  int n = 0;
  rcclTelemetrySetupChannel(0, 0, N, &n);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  CHECK_EQ((uintptr_t)ch % RCCL_TELEMETRY_CACHELINE, 0);
  for (int q = 0; q < N; q++) {
    RcclQpStats* qp = rcclTelemetryQp(ch, q);
    CHECK(qp != nullptr);
    if (qp) CHECK_EQ((uintptr_t)qp % RCCL_TELEMETRY_CACHELINE, 0);
  }
}

/* ---- resolved slot handles ------------------------------------------ */

/* Every per-QP counter, so an equivalence check cannot miss one. */
static void qpSnapshot(const RcclQpStats* qp, uint64_t out[12]) {
  out[0] = qp->num_wqe_sent;
  out[1] = qp->num_recv_wqe;
  out[2] = qp->num_wqe_rcvd;
  out[3] = qp->num_wqe_completed;
  out[4] = qp->num_slot_miss;
  out[5] = qp->num_cts_sent;
  out[6] = qp->num_cts_sent_signalled;
  out[7] = qp->num_cts_sent_unsignalled;
  out[8] = qp->num_write_wqe;
  out[9] = qp->num_write_imm_wqe;
  out[10] = 0;
  out[11] = 0;
  for (int b = 0; b < RCCL_TELEMETRY_HISTOGRAM_SIZE; b++) out[10] += qp->wqe_completion_histogram[b];
  out[11] = (uint64_t)qp->id;
}

/* A handle must resolve to exactly the slot the index-taking path reaches, and
 * must be NULL in precisely the cases where that path would find nothing. */
static void testHandleResolution(void) {
  std::printf("testHandleResolution\n");
  resetState();
  const int N = 40;
  int n = 0;
  rcclTelemetrySetupChannel(0, 3, N, &n);
  CHECK_EQ(n, N);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 3);
  CHECK(ch != nullptr);
  CHECK(rcclTelemetryResolveChannel(0, 3) == ch);
  for (int q = 0; q < N; q++) CHECK(rcclTelemetryResolveQp(0, 3, q) == rcclTelemetryQp(ch, q));

  /* Out of range in each index, and one past the live slots. */
  CHECK(rcclTelemetryResolveQp(0, 3, N) == nullptr);
  CHECK(rcclTelemetryResolveQp(0, 3, -1) == nullptr);
  CHECK(rcclTelemetryResolveQp(0, 4, 0) == nullptr); /* channel exists, no slots */
  CHECK(rcclTelemetryResolveQp(-1, 3, 0) == nullptr);
  CHECK(rcclTelemetryResolveQp(RCCL_TELEMETRY_MAX_DEVS, 3, 0) == nullptr);
  /* Channel resolution is bounded by the allocated capacity, which runs ahead
   * of the channels actually set up, so pick an index past every block. */
  CHECK(rcclTelemetryResolveChannel(0, RCCL_TELEMETRY_MAX_CHANNELS) == nullptr);
  CHECK(rcclTelemetryResolveChannel(0, -1) == nullptr);
  CHECK(rcclTelemetryResolveChannel(0, ch->id) == ch);

  /* Obtaining a handle is the guarded step: with telemetry off there is none. */
  rcclTelemetryEnabled = 0;
  CHECK(rcclTelemetryResolveQp(0, 3, 0) == nullptr);
  CHECK(rcclTelemetryResolveChannel(0, 3) == nullptr);
  rcclTelemetryEnabled = 1;

  /* And a NULL handle is a silent no-op in every hook that takes one, which is
   * what lets call sites skip the check. */
  rcclTelemetryQpSendPosted(nullptr, 1);
  rcclTelemetryQpRecvPosted(nullptr);
  rcclTelemetryQpWqeComplete(nullptr, 12345);
  rcclTelemetryQpCtsSent(nullptr, 1);
  rcclTelemetryQpSlotMiss(nullptr);
  rcclTelemetryChRequestCompleted(nullptr);
  RcclChannelAggregate agg;
  rcclTelemetryChannelAggregate(ch, &agg);
  CHECK_EQ(agg.num_wqe_sent + agg.num_recv_wqe + agg.num_wqe_rcvd + agg.num_cts_sent, 0);
  CHECK_EQ(ch->num_req_completed, 0);
}

/* The hot path drives handles; the index-taking entry points stay as the
 * documented equivalent. Same call sequence through both must leave the two
 * QP slots byte-for-byte equal on every counter. */
static void testHandleEquivalence(void) {
  std::printf("testHandleEquivalence\n");
  resetState();
  int n = 0;
  rcclTelemetrySetupChannel(0, 0, 2, &n);
  CHECK_EQ(n, 2);
  RcclQpStats* viaHandle = rcclTelemetryResolveQp(0, 0, 1);
  CHECK(viaHandle != nullptr);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);

  for (int i = 0; i < 500; i++) {
    int withImm = i & 1;
    int signalled = (i % 3) == 0;

    /* index-taking form on slot 0 */
    rcclTelemetryWqePosted(0, 0, 0, 1);
    rcclTelemetryWriteWqe(0, 0, 0, withImm);
    rcclTelemetryWqePosted(0, 0, 0, 0);
    rcclTelemetryWqeComplete(0, 0, 0, 0);
    rcclTelemetryCtsSent(0, 0, 0, signalled);
    rcclTelemetrySlotMiss(0, 0, 0);

    /* combined handle form on slot 1 */
    rcclTelemetryQpSendPosted(viaHandle, withImm);
    rcclTelemetryQpRecvPosted(viaHandle);
    rcclTelemetryQpWqeComplete(viaHandle, 0);
    rcclTelemetryQpCtsSent(viaHandle, signalled);
    rcclTelemetryQpSlotMiss(viaHandle);
  }

  uint64_t a[12], b[12];
  qpSnapshot(rcclTelemetryQp(ch, 0), a);
  qpSnapshot(viaHandle, b);
  for (int f = 0; f < 11; f++) CHECK_EQ_U(a[f], b[f]);
  CHECK_EQ_U(a[0], 500);  /* the combined form counted every posting once */
  CHECK_EQ_U(a[8] + a[9], 500);

  /* Same for the channel-level request counter. */
  for (int i = 0; i < 100; i++) rcclTelemetryRequestCompleted(0, 0);
  for (int i = 0; i < 100; i++) rcclTelemetryChRequestCompleted(ch);
  CHECK_EQ(ch->num_req_completed, 200);
}

/* A handle is resolved once and kept for the life of the process, so growth
 * must never move the slot it names. */
static void testHandleStableAcrossGrowth(void) {
  std::printf("testHandleStableAcrossGrowth\n");
  resetState();
  int n = 0;
  rcclTelemetrySetupChannel(0, 0, 8, &n);
  CHECK_EQ(n, 8);
  RcclQpStats* early[8];
  for (int q = 0; q < 8; q++) {
    early[q] = rcclTelemetryResolveQp(0, 0, q);
    CHECK(early[q] != nullptr);
  }
  RcclChannelStats* chEarly = rcclTelemetryResolveChannel(0, 0);

  /* Many growths, well past the first few block boundaries. */
  for (int round = 0; round < 100; round++) {
    int got = 0;
    rcclTelemetrySetupChannel(0, 0, 6, &got);
    CHECK_EQ(got, 6);
  }
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);
  CHECK(ch == chEarly);
  CHECK_EQ(ch->num_qps, 8 + 100 * 6);
  for (int q = 0; q < 8; q++) {
    CHECK(early[q] == rcclTelemetryResolveQp(0, 0, q));
    rcclTelemetryQpSendPosted(early[q], 0);
    CHECK_EQ(rcclTelemetryQp(ch, q)->num_wqe_sent, 1);
  }
  /* Channel storage grows too; that handle must survive it as well. */
  for (int c = 1; c < 40; c++) {
    int got = 0;
    rcclTelemetrySetupChannel(0, c, 2, &got);
  }
  CHECK(rcclTelemetryResolveChannel(0, 0) == chEarly);
}

/* ---- histogram bucket index, division vs. reciprocal ---------------- */

/* Exactly the expression rcclTelemetryLatencyBucket() replaced. */
static int refLatencyBucket(int64_t latency_ns, int64_t interval, int maxBuckets) {
  int bucket = (int)(latency_ns / interval);
  if (bucket >= maxBuckets) bucket = maxBuckets - 1;
  return bucket;
}

/* The claim the whole change rests on: strictly below histogram_recip_max_ns,
 * the multiply-and-shift is the division. */
static void checkRecipQuotient(uint64_t n, int64_t interval) {
  uint64_t viaRecip = (uint64_t)(((__uint128_t)n * rcclTelemetryCfg.histogram_recip) >> RCCL_TELEMETRY_RECIP_SHIFT);
  CHECK_EQ_U(viaRecip, n / (uint64_t)interval);
}

/* Bit-identical means bit-identical: compared unconditionally, including the
 * inputs whose quotient does not fit in an int and where both expressions
 * narrow it the same wrong way before clamping. */
static void checkBucket(int64_t latency_ns, int64_t interval, int maxBuckets) {
  CHECK_EQ(rcclTelemetryLatencyBucket(latency_ns), refLatencyBucket(latency_ns, interval, maxBuckets));
}

static void testLatencyBucket(void) {
  std::printf("testLatencyBucket\n");
  resetState();

  /* The default, powers of two, odd, prime, and intervals past 2^32 so the
   * reciprocal is exercised where it is smallest. */
  static const int64_t intervals[] = {
    1, 2, 3, 5, 7, 8, 30000, 1000, 4096, 12345, 65537, 999983, 1000000, 2147483647LL, 4294967311LL, 1099511627776LL
  };
  static const int maxBucketSets[] = {1, 2, 5, RCCL_TELEMETRY_HISTOGRAM_SIZE};

  uint64_t seed = 0x9E3779B97F4A7C15ULL;
  auto nextRand = [&seed]() {
    seed ^= seed >> 12;
    seed ^= seed << 25;
    seed ^= seed >> 27;
    return seed * 0x2545F4914F6CDD1DULL;
  };

  for (size_t ii = 0; ii < sizeof(intervals) / sizeof(intervals[0]); ii++) {
    const int64_t d = intervals[ii];
    rcclTelemetryCfg.histogram_bucket_interval_ns = d;
    rcclTelemetryConfigDeriveHistogram();
    const uint64_t recipMax = rcclTelemetryCfg.histogram_recip_max_ns;
    CHECK_EQ_U(recipMax, ((uint64_t)1 << RCCL_TELEMETRY_RECIP_SHIFT) / (uint64_t)d);
    CHECK_EQ_U(rcclTelemetryCfg.histogram_recip, recipMax + 1);

    /* Exhaustive at small magnitudes. */
    for (uint64_t v = 0; v < 100000 && v < recipMax; v++) checkRecipQuotient(v, d);

    /* Every bucket edge, and one either side of it. */
    for (uint64_t k = 0; k < 4096; k++) {
      if (recipMax != 0 && k > recipMax / (uint64_t)d + 2) break;
      uint64_t edge = k * (uint64_t)d;
      if (edge >= recipMax) break;
      if (edge > 0) checkRecipQuotient(edge - 1, d);
      checkRecipQuotient(edge, d);
      if (edge + 1 < recipMax) checkRecipQuotient(edge + 1, d);
    }

    /* The top of the proven domain, where the error term is largest. */
    for (uint64_t back = 1; back <= 4096 && back <= recipMax; back++) checkRecipQuotient(recipMax - back, d);

    /* Wide randomized: inside the proven domain, and over the whole positive
     * int64 range so the division fallback is covered too. */
    for (int i = 0; i < 100000; i++) {
      if (recipMax != 0) checkRecipQuotient(nextRand() % recipMax, d);
    }

    for (size_t mi = 0; mi < sizeof(maxBucketSets) / sizeof(maxBucketSets[0]); mi++) {
      rcclTelemetryCfg.histogram_max_buckets = maxBucketSets[mi];
      const int maxB = maxBucketSets[mi];

      for (int64_t v = 0; v < 20000; v++) checkBucket(v, d, maxB);
      for (int64_t k = 0; k <= maxB + 3; k++) {
        int64_t edge = k * d; /* d <= 2^40 and k <= 19, so this cannot overflow */
        if (edge > 0) checkBucket(edge - 1, d, maxB);
        checkBucket(edge, d, maxB);
        checkBucket(edge + 1, d, maxB);
      }
      /* Straddle the reciprocal/division boundary. */
      for (uint64_t back = 0; back < 64 && back < recipMax; back++) checkBucket((int64_t)(recipMax - back), d, maxB);
      for (uint64_t fwd = 0; fwd < 64; fwd++) {
        uint64_t v = recipMax + fwd;
        if (v <= (uint64_t)INT64_MAX) checkBucket((int64_t)v, d, maxB);
      }
      checkBucket(INT64_MAX, d, maxB);
      for (int i = 0; i < 30000; i++) {
        checkBucket((int64_t)(nextRand() >> 1), d, maxB);
        if (recipMax != 0) checkBucket((int64_t)(nextRand() % recipMax), d, maxB);
      }
    }
    if (g_failures) return;
  }

  /* A zeroed config keeps the division, so nothing depends on the derived
   * fields having been computed. */
  rcclTelemetryCfg.histogram_bucket_interval_ns = 30000;
  rcclTelemetryCfg.histogram_max_buckets = 5;
  rcclTelemetryCfg.histogram_recip = 0;
  rcclTelemetryCfg.histogram_recip_max_ns = 0;
  for (int64_t v = 0; v < 200000; v++) checkBucket(v, 30000, 5);

  /* A non-positive interval must not leave a reciprocal behind that a later
   * completion would divide by. */
  rcclTelemetryCfg.histogram_bucket_interval_ns = 0;
  rcclTelemetryConfigDeriveHistogram();
  CHECK_EQ_U(rcclTelemetryCfg.histogram_recip, 0);
  CHECK_EQ_U(rcclTelemetryCfg.histogram_recip_max_ns, 0);

  resetState();
}

/* The three fields the completion hook reads must stay on one cache line: the
 * reciprocal only pays off if reaching it does not cost an extra miss. */
static void testConfigLayout(void) {
  std::printf("testConfigLayout\n");
  size_t first = offsetof(RcclTelemetryConfig, histogram_max_buckets);
  size_t last = offsetof(RcclTelemetryConfig, histogram_recip_max_ns) + sizeof(uint64_t) - 1;
  CHECK_EQ(first / RCCL_TELEMETRY_CACHELINE, last / RCCL_TELEMETRY_CACHELINE);
}

/* ---- concurrent growth vs. reads (run under TSan/ASan) -------------- */
static void testConcurrent(void) {
  std::printf("testConcurrent\n");
  resetState();
  const int START = 8;
  const int MAX = 600;
  int n = 0;
  rcclTelemetrySetupChannel(0, 0, START, &n);
  RcclChannelStats* ch = rcclTelemetryChannel(0, 0);

  /* Resolved before any growth, as the data path resolves at setup time: these
   * must stay usable while the grower appends blocks. */
  RcclQpStats* pinned[START];
  for (int q = 0; q < START; q++) pinned[q] = rcclTelemetryResolveQp(0, 0, q);
  RcclChannelStats* pinnedCh = rcclTelemetryResolveChannel(0, 0);

  std::atomic<bool> stop{false};

  std::thread grower([&]() {
    for (int target = START + 8; target <= MAX; target += 8) {
      int got = 0;
      rcclTelemetrySetupChannel(0, 0, 8, &got);
      std::this_thread::yield();
    }
    stop.store(true, std::memory_order_release);
  });

  std::vector<std::thread> workers;
  for (int w = 0; w < 4; w++) {
    workers.emplace_back([&, w]() {
      unsigned int seed = 0x1234 + w;
      while (!stop.load(std::memory_order_acquire)) {
        int live = __atomic_load_n(&ch->num_qps, __ATOMIC_ACQUIRE);
        if (live <= 0) continue;
        for (int i = 0; i < 256; i++) {
          seed = seed * 1103515245u + 12345u;
          int q = (int)(seed % (unsigned int)live);
          rcclTelemetryWqePosted(0, 0, q, 1);
          rcclTelemetryWqeComplete(0, 0, q, 0);
          rcclTelemetryCtsSent(0, 0, q, i & 1);
        }
        /* The hot-path shape: resolve once, then update through the handle. */
        for (int i = 0; i < 256; i++) {
          seed = seed * 1103515245u + 12345u;
          RcclQpStats* qp = rcclTelemetryResolveQp(0, 0, (int)(seed % (unsigned int)live));
          rcclTelemetryQpSendPosted(qp, i & 1);
          rcclTelemetryQpRecvPosted(qp);
          rcclTelemetryQpWqeComplete(qp, 0);
          rcclTelemetryQpCtsSent(qp, i & 1);
          rcclTelemetryQpSendPosted(pinned[w], i & 1);
          rcclTelemetryChRequestCompleted(pinnedCh);
        }
      }
    });
  }

  grower.join();
  for (auto& t : workers) t.join();

  CHECK_EQ(ch->num_qp_untracked, 0);
  checkInvariant(ch);
  /* The pre-growth handles still name the slots they named at the start. */
  for (int q = 0; q < START; q++) CHECK(pinned[q] == rcclTelemetryResolveQp(0, 0, q));
  CHECK(pinnedCh == rcclTelemetryResolveChannel(0, 0));
  CHECK(ch->num_req_completed > 0);
}

int main(void) {
  std::printf("net_telemetry unit tests\n");
  testBlockIndex();
  testGrowth();
  testOverOldCeiling();
  testChannelSanityBound();
  testInvariantSingle();
  testAggregateAcrossBlocks();
  testCacheLineAlignment();
  testHandleResolution();
  testHandleEquivalence();
  testHandleStableAcrossGrowth();
  testLatencyBucket();
  testConfigLayout();
  testConcurrent();

  if (g_failures == 0) {
    std::printf("ALL PASSED\n");
    return 0;
  }
  std::printf("%d CHECK(s) FAILED\n", g_failures);
  return 1;
}
