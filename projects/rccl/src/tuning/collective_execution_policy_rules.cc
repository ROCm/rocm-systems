/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collective_execution_policy_internal.h"

namespace rcclExecutionPolicyRules {

// Adding a rule
// -------------
// Each entry is {{match fields}, {atomic execution profile}, score}. Array
// index is both the build-local rule ID and priority; lower indices win. Insert
// a rule at the priority where it should be evaluated, noting that this
// renumbers all following rules.
//
// Score is EFFICIENCY(peak measured bus GB/s over the rule's inputs, platform
// peak bus GB/s), or UNMEASURED. The build rejects a table where:
//   - two rules that can match the same input differ in byte range or
//     in-place constraint (scores would describe different inputs), or
//   - a measured rule outscores an earlier measured rule matching the same
//     input by more than kScoreOrderMargin (the earlier rule would shadow it).
// After a forced-transport sweep, tools/scripts/execution_policy_scores.py
// splits rules to satisfy the first check, rewrites every score, and lists
// the rules that must move to satisfy the second.
//
// Match fields, in order:
//   scope              P2P for P2P-backed planning, COLL for normal collectives.
//   arch               IsArchMatch pattern; e.g. "gfx120" covers that family.
//   collective         ALL_TO_ALL, ALL_GATHER, REDUCE_SCATTER, etc.
//   bytes              Inclusive {minimum, maximum} aggregate bytes per rank.
//                      Use kAnyBytes when size is irrelevant.
//   nodes              Node-count constraint: EQ(n), GE(n), or kAnyInt.
//   ranks              Rank-count constraint: EQ(n), GE(n), or kAnyInt.
//   available channels Provisioned-channel constraint: EQ(n), GE(n), or kAnyInt.
//   candidate transport IPC, SHM, NET, COLLNET, MIXED, or ANY_TRANSPORT.
//                      Concrete candidates are filtered against the adapter's
//                      eligible transport mask and become the policy output.
//   in-place           BOOL_TRUE when buffers overlap, BOOL_FALSE when
//                      disjoint, or ANY_BOOL when placement is irrelevant.
//
// Profile fields, in order:
//   channels           Scope-local channel count.
//   algorithm          RING or another NCCL_ALGO_* alias.
//   protocol           SIMPLE, LL, or another NCCL_PROTO_* alias.
//   path               SENDRECV or AUTO_PATH.
//   transfer mode      READ, WRITE, or AUTO_XFER.
// KEEP omits that assignment and preserves RCCL's existing choice. AUTO is an
// explicit -1 automatic selection for channels, algorithm, or protocol. A
// profile is selected atomically; fields from different rules are never merged.
//
// P2P example (single-node, eight-rank, >=1 MiB, two-channel write):
//   {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, SIZE_MAX}, EQ(1), EQ(8),
//     kAnyInt, IPC, ANY_BOOL},
//    {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE},
//    EFFICIENCY(20.33, kGfx120xPeakBusBwGBps)},
//
// Collective example (single-node, eight-rank Ring/Simple):
//   {{COLL, "gfx120", ALL_REDUCE, {128, 8ULL << 30}, EQ(1), EQ(8),
//     kAnyInt, IPC, ANY_BOOL},
//    {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},

// PCIe Gen5 x16 per direction: 32 GT/s * 16 lanes * 128/130 / 8.
constexpr double kGfx120xPeakBusBwGBps = 63.0;

constexpr RuleSpec kCollectiveExecutionRules[] = {
  // Preserve the gfx110x AllToAll variance workaround from #11803 as a
  // direction-local channel cap. PATH remains AUTO so existing collective
  // routing (pivot, CE, GDA, GIN, and DDA) is not overridden.
  {{P2P, "gfx110", ALL_TO_ALL, kAnyBytes, kAnyInt, kAnyInt,
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(1), KEEP, AUTO, AUTO_PATH, AUTO_XFER}, UNMEASURED},

  // Direct Send/Recv remains on IPC below the measured 2 MiB per-peer
  // crossover. P2P policy facts are rank-aggregate bytes, hence 16 MiB here
  // for the eight-rank gfx120x tuning topology. Keep the ordinary Send/Recv
  // work shape automatic; this row selects only its transport.
  {{P2P, "gfx120", ncclFuncSend, {0, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, KEEP, AUTO, AUTO_PATH, AUTO_XFER}, EFFICIENCY(17.04, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ncclFuncRecv, {0, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, KEEP, AUTO, AUTO_PATH, AUTO_XFER}, EFFICIENCY(17.04, kGfx120xPeakBusBwGBps)},

  // Runtime IPC/SHM crossover candidates for P2P-backed collectives. Each
  // endpoint resolves the same per-link byte count and placement, so the
  // selected connector index remains symmetric. Bands require a >=3% win in
  // both placements at two or more adjacent sizes in the September 24 sweep.
  // Placement-specific rows are used only where the measurements diverge.
  {{P2P, "gfx1201", GATHER, {1ULL << 20, (1536ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_FALSE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(35.75, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx1201", GATHER, {1536ULL << 10, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_FALSE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(60.06, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx1201", GATHER, {1536ULL << 10, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_TRUE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(58.64, kGfx120xPeakBusBwGBps)},

  {{P2P, "gfx1201", SCATTER, {48ULL << 10, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, UNMEASURED},
  {{P2P, "gfx1201", SCATTER, {64ULL << 10, (768ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(14.99, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx1201", SCATTER, {96ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(56.95, kGfx120xPeakBusBwGBps)},

  {{P2P, "gfx1201", ALL_TO_ALL, {48ULL << 10, (128ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_FALSE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(1.24, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx1201", ALL_TO_ALL, {128ULL << 10, (768ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_FALSE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(7.38, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx1201", ALL_TO_ALL, {48ULL << 10, (128ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_TRUE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(1.32, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx1201", ALL_TO_ALL, {128ULL << 10, (768ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_TRUE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(7.60, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx1201", ALL_TO_ALL, {12ULL << 20, (96ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_FALSE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(20.21, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx1201", ALL_TO_ALL, {12ULL << 20, (96ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_TRUE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(20.23, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx1201", ALL_TO_ALL, {384ULL << 20, (512ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_FALSE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, UNMEASURED},
  {{P2P, "gfx1201", ALL_TO_ALL, {512ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_FALSE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(19.39, kGfx120xPeakBusBwGBps)},

  // Large out-of-place AllToAll is stable on one channel; two channels produce
  // severe long-tail stalls while in-place execution remains stable.
  {{P2P, "gfx120", ALL_TO_ALL, {512ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, IPC, BOOL_FALSE},
   {CHANNELS(1), KEEP, AUTO, SENDRECV, WRITE}, EFFICIENCY(18.10, kGfx120xPeakBusBwGBps)},

  // Provisioning the wide read-mode pool also widens the default P2P pool.
  // Keep the measured low-size range on two active channels while preserving
  // automatic protocol and transfer-mode selection. AllToAll placement rows
  // are intentionally separate so future out-of-place tuning can diverge.
  {{P2P, "gfx120", ALL_TO_ALL, {128ULL << 10, (768ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(4.91, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ALL_TO_ALL, {768ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, AUTO, SENDRECV, AUTO_XFER}, UNMEASURED},
  {{P2P, "gfx120", ALL_TO_ALL, {128ULL << 10, (768ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(5.03, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ALL_TO_ALL, {768ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, AUTO, SENDRECV, AUTO_XFER}, UNMEASURED},
  {{P2P, "gfx120", SCATTER, {64ULL << 10, (768ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, AUTO, SENDRECV, AUTO_XFER}, EFFICIENCY(7.92, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", SCATTER, {768ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, AUTO, SENDRECV, AUTO_XFER}, UNMEASURED},

  // Read mode only wins these fanout collectives when peers are spread across
  // the wide active pool. Restrict the experiment to the measured topology and
  // size range; the generic write rules below remain the fallback.
  {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, 4ULL << 20}, EQ(1), EQ(8),
    kAnyInt, IPC, BOOL_FALSE},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}, EFFICIENCY(15.92, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, 4ULL << 20}, EQ(1), EQ(8),
    kAnyInt, IPC, BOOL_TRUE},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}, EFFICIENCY(15.93, kGfx120xPeakBusBwGBps)},
  // AllToAllv's per-peer tasks are asymmetric, so transfer mode cannot depend
  // on an individual task's byte count without risking endpoint disagreement.
  {{P2P, "gfx120", ALL_TO_ALL_V, kAnyBytes, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}, EFFICIENCY(17.26, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", SCATTER, {1ULL << 20, (96ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}, EFFICIENCY(53.88, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", SCATTER, {96ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}, EFFICIENCY(54.28, kGfx120xPeakBusBwGBps)},

  // Runtime IPC/SHM crossover candidates from the September 24 forced-path
  // sweep. SHM is ranked ahead of the broader IPC candidates only where it won
  // by >=3% in both placements at two or more adjacent measured sizes.
  // Arithmetic midpoints provide smooth handoffs between measured sizes.
  {{COLL, "gfx1201", REDUCE, {24ULL << 20, (1536ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(24), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(26.71, kGfx120xPeakBusBwGBps)},
  // AllReduce, AllGather, and ReduceScatter have no stable SHM-winning
  // crossover. Broadcast also remains IPC below 16 MiB; all four intentionally
  // fall through to their IPC profiles.

  // Eight-rank, single-node gfx1201 P2P/IPC matrix. Ring/protocol ranges use
  // both placements; channel overrides require a >=3% gain in each placement
  // across at least two adjacent measured sizes.
  {{COLL, "gfx120", ALL_GATHER, {128, (16ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, LL, AUTO_PATH, KEEP}, EFFICIENCY(0.27, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", ALL_GATHER, {16ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, AUTO, AUTO_PATH, KEEP}, EFFICIENCY(0.50, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", ALL_GATHER, {32ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(2.81, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", ALL_GATHER, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(8.37, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", ALL_GATHER, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(19.96, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", ALL_GATHER, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.75, kGfx120xPeakBusBwGBps)},

  {{COLL, "gfx120", ALL_REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, LL, AUTO_PATH, KEEP}, EFFICIENCY(1.27, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", ALL_REDUCE, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(3.80, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", ALL_REDUCE, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(10.23, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", ALL_REDUCE, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(19.65, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", ALL_REDUCE, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.56, kGfx120xPeakBusBwGBps)},

  {{COLL, "gfx120", BROADCAST, {128, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(1.50, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", BROADCAST, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(4.77, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", BROADCAST, {256ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(7.32, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", BROADCAST, {512ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(12.38, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", BROADCAST, {1ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(27.47, kGfx120xPeakBusBwGBps)},

  {{COLL, "gfx120", REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(1.47, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(4.53, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE, {256ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(7.02, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE, {512ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(10.29, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.89, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE, {16ULL << 20, (24ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(24.20, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE, {24ULL << 20, (1536ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(24.75, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE, {1536ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.69, kGfx120xPeakBusBwGBps)},

  {{COLL, "gfx120", REDUCE_SCATTER, {128, (8ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, LL, AUTO_PATH, KEEP}, EFFICIENCY(0.14, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE_SCATTER, {8ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, AUTO, AUTO_PATH, KEEP}, EFFICIENCY(0.52, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE_SCATTER, {32ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(3.02, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE_SCATTER, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(8.33, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE_SCATTER, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(20.05, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE_SCATTER, {16ULL << 20, (2ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.65, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE_SCATTER, {2ULL << 30, (8ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(6), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.18, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx120", REDUCE_SCATTER, {8ULL << 30, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.35, kGfx120xPeakBusBwGBps)},

  {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, 4ULL << 20}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},
  {{P2P, "gfx120", ALL_TO_ALL, {4194305, (12ULL << 20) - 1}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(14.69, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ALL_TO_ALL, {12ULL << 20, (96ULL << 20) - 1}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(19.24, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ALL_TO_ALL, {96ULL << 20, (384ULL << 20) - 1}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(19.82, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ALL_TO_ALL, {384ULL << 20, (512ULL << 20) - 1}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},
  {{P2P, "gfx120", ALL_TO_ALL, {512ULL << 20, 8ULL << 30}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},
  {{P2P, "gfx120", ALL_TO_ALL, {8589934593, SIZE_MAX}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},
  {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, 4ULL << 20}, EQ(1), GE(3),
    GE(3), IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},
  {{P2P, "gfx120", ALL_TO_ALL, {4194305, (12ULL << 20) - 1}, EQ(1), GE(3),
    GE(3), IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(17.18, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ALL_TO_ALL, {12ULL << 20, (96ULL << 20) - 1}, EQ(1), GE(3),
    GE(3), IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(19.61, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ALL_TO_ALL, {96ULL << 20, SIZE_MAX}, EQ(1), GE(3),
    GE(3), IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(20.34, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", ALL_GATHER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3),
    GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(22.75, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", REDUCE_SCATTER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3),
    GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(22.65, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", GATHER, {1ULL << 20, (1536ULL << 10) - 1}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(30.01, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", GATHER, {1536ULL << 10, 8ULL << 30}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(47.34, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", GATHER, {8589934593, SIZE_MAX}, EQ(1), GE(3),
    GE(3), IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},
  {{P2P, "gfx120", GATHER, {1ULL << 20, (1536ULL << 10) - 1}, EQ(1), GE(3),
    GE(3), IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(30.00, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", GATHER, {1536ULL << 10, 8ULL << 30}, EQ(1), GE(3),
    GE(3), IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, EFFICIENCY(47.68, kGfx120xPeakBusBwGBps)},
  {{P2P, "gfx120", GATHER, {8589934593, SIZE_MAX}, EQ(1), GE(3),
    GE(3), IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},
  {{P2P, "gfx120", SCATTER, {1ULL << 20, (96ULL << 20) - 1}, EQ(1), GE(3),
    GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},
  {{P2P, "gfx120", SCATTER, {96ULL << 20, 8ULL << 30}, EQ(1), GE(3),
    GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},
  {{P2P, "gfx120", SCATTER, {8589934593, SIZE_MAX}, EQ(1), GE(3),
    GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}, UNMEASURED},

  // Eight-rank, single-node gfx1201 pure-SHM matrix. These ranges follow stable
  // Ring/Simple bus-bandwidth plateaus and merge isolated point winners into
  // smooth size bands.
  {{COLL, "gfx1201", ALL_REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(0.80, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", ALL_REDUCE, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(2.72, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", ALL_REDUCE, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(4), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(7.46, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", ALL_REDUCE, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(12), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(18.10, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", ALL_REDUCE, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(24), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.86, kGfx120xPeakBusBwGBps)},

  {{COLL, "gfx1201", ALL_GATHER, {128, (16ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(0.16, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", ALL_GATHER, {16ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(0.32, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", ALL_GATHER, {32ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(2.38, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", ALL_GATHER, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(6), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(6.52, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", ALL_GATHER, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(12), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(18.05, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", ALL_GATHER, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(28), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.18, kGfx120xPeakBusBwGBps)},

  {{COLL, "gfx1201", REDUCE_SCATTER, {128, (8ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(0.08, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE_SCATTER, {8ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(0.34, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE_SCATTER, {32ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(2.46, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE_SCATTER, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(2), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(6.65, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE_SCATTER, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(12), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(17.05, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE_SCATTER, {16ULL << 20, (2ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(28), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.28, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE_SCATTER, {2ULL << 30, (8ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(28), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(21.43, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE_SCATTER, {8ULL << 30, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(28), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(21.20, kGfx120xPeakBusBwGBps)},

  {{COLL, "gfx1201", REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(1.30, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(3.68, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE, {256ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(2), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(6.39, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE, {512ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(2), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(8.48, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(8), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(21.14, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE, {16ULL << 20, (24ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(24), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(23.67, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", REDUCE, {24ULL << 20, (1536ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(24), RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE, {1536ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(24), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.48, kGfx120xPeakBusBwGBps)},

  // Broadcast's sub-64 KiB channel winner was noisy. Select only its measured
  // Ring/Simple protocol there, then use the stable channel plateaus.
  {{COLL, "gfx1201", BROADCAST, {128, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(1.34, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", BROADCAST, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(3.69, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", BROADCAST, {256ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(2), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(6.45, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", BROADCAST, {512ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(2), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(8.62, kGfx120xPeakBusBwGBps)},
  {{COLL, "gfx1201", BROADCAST, {1ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(8), RING, SIMPLE, AUTO_PATH, KEEP}, EFFICIENCY(22.68, kGfx120xPeakBusBwGBps)},

  // If a user or per-call channel bound rejects a measured channel profile,
  // retain the SHM algorithm/protocol choice without partially applying the
  // rejected profile. Explicit NCCL_ALGO/NCCL_PROTO conflicts reject these
  // fallbacks as well and preserve the user's selection.
  {{COLL, "gfx1201", ALL_REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_REDUCE, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_REDUCE, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_REDUCE, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_REDUCE, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_GATHER, {128, (16ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_GATHER, {16ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_GATHER, {32ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_GATHER, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_GATHER, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", ALL_GATHER, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE_SCATTER, {128, (8ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE_SCATTER, {8ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE_SCATTER, {32ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE_SCATTER, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE_SCATTER, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE_SCATTER, {16ULL << 20, (2ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE_SCATTER, {2ULL << 30, (8ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE_SCATTER, {8ULL << 30, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE, {256ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE, {512ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE, {16ULL << 20, (24ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE, {24ULL << 20, (1536ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", REDUCE, {1536ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", BROADCAST, {128, (64ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", BROADCAST, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", BROADCAST, {256ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", BROADCAST, {512ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
  {{COLL, "gfx1201", BROADCAST, {1ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}, UNMEASURED},
};

constexpr size_t kCollectiveExecutionRuleCount =
  sizeof(kCollectiveExecutionRules) / sizeof(kCollectiveExecutionRules[0]);
constexpr size_t kFirstInvalidRule = firstInvalidRule(kCollectiveExecutionRules);
constexpr size_t kFirstIncompleteP2pTransportShape =
  firstIncompleteP2pTransportShape(kCollectiveExecutionRules);
constexpr size_t kFirstMismatchedAllToAllVShape =
  firstMismatchedAllToAllVShape(kCollectiveExecutionRules);
static_assert(kCollectiveExecutionRuleCount <= RCCL_EXECUTION_POLICY_NO_RULE,
              "Execution policy rule count exceeds the RuleId range");
static_assert(kFirstInvalidRule == kCollectiveExecutionRuleCount,
              "Invalid execution policy rule; compiler diagnostic shows its array index");
static_assert(kFirstIncompleteP2pTransportShape == kCollectiveExecutionRuleCount,
              "P2P transport candidates must carry a complete work shape");
static_assert(kFirstMismatchedAllToAllVShape == kCollectiveExecutionRuleCount,
              "AllToAllV P2P rules must share one non-transport work shape");
static_assert(firstMisalignedRule(kCollectiveExecutionRules) == kCollectiveExecutionRuleCount,
              "Rules that can match one input must cover identical sizes and placement; "
              "run tools/scripts/execution_policy_scores.py");
static_assert(firstMisorderedRule(kCollectiveExecutionRules) == kCollectiveExecutionRuleCount,
              "A measurably faster rule is shadowed by an earlier rule matching the same input; "
              "move it earlier");

const auto kExpandedCollectiveExecutionRules = expandRules(kCollectiveExecutionRules);
const RuleTableView kCollectiveExecutionRuleTable = {
  kCollectiveExecutionRules,
  kExpandedCollectiveExecutionRules.data(),
  kCollectiveExecutionRuleCount,
};

} // namespace rcclExecutionPolicyRules
