/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "collective_execution_policy_internal.h"

namespace rcclExecutionPolicyRules {

// Adding a rule
// -------------
// Each entry is {{match fields}, {atomic execution profile}}. Array index is
// both the build-local rule ID and priority; lower indices win. Insert a rule
// at the priority where it should be evaluated, noting that this renumbers all
// following rules.
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
//    {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
//
// Collective example (single-node, eight-rank Ring/Simple):
//   {{COLL, "gfx120", ALL_REDUCE, {128, 8ULL << 30}, EQ(1), EQ(8),
//     kAnyInt, IPC, ANY_BOOL},
//    {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},
constexpr RuleSpec kCollectiveExecutionRules[] = {
  // Preserve the gfx110x AllToAll variance workaround from #11803 as a
  // direction-local channel cap. PATH remains AUTO so existing collective
  // routing (pivot, CE, GDA, GIN, and DDA) is not overridden.
  {{P2P, "gfx110", ALL_TO_ALL, kAnyBytes, kAnyInt, kAnyInt, kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(1), KEEP, AUTO, AUTO_PATH, AUTO_XFER}},

  // Direct Send/Recv remains on IPC below the measured 2 MiB per-peer
  // crossover. P2P policy facts are rank-aggregate bytes, hence 16 MiB here
  // for the eight-rank gfx120x tuning topology. Keep the ordinary Send/Recv
  // work shape automatic; this row selects only its transport.
  {{P2P, "gfx120", ncclFuncSend, {0, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, KEEP, AUTO, AUTO_PATH, AUTO_XFER}},
  {{P2P, "gfx120", ncclFuncRecv, {0, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, KEEP, AUTO, AUTO_PATH, AUTO_XFER}},

  // Runtime IPC/SHM crossover candidates for P2P-backed collectives. Each
  // endpoint resolves the same per-link byte count and placement, so the
  // selected connector index remains symmetric. Bands require a >=3% win in
  // both placements at two or more adjacent sizes in the September 24 sweep.
  // Placement-specific rows are used only where the measurements diverge.
  {{P2P, "gfx1201", GATHER, {1ULL << 20, (1536ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_FALSE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}},
  {{P2P, "gfx1201", GATHER, {1536ULL << 10, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}},

  {{P2P, "gfx1201", SCATTER, {48ULL << 10, (768ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}},
  {{P2P, "gfx1201", SCATTER, {96ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}},

  {{P2P, "gfx1201", ALL_TO_ALL, {48ULL << 10, (768ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}},
  {{P2P, "gfx1201", ALL_TO_ALL, {12ULL << 20, (96ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}},
  {{P2P, "gfx1201", ALL_TO_ALL, {384ULL << 20, 8ULL << 30}, EQ(1), EQ(8),
    kAnyInt, SHM, BOOL_FALSE},
   {CHANNELS(32), KEEP, AUTO, SENDRECV, AUTO_XFER}},

  // Large out-of-place AllToAll is stable on one channel; two channels produce
  // severe long-tail stalls while in-place execution remains stable.
  {{P2P, "gfx120", ALL_TO_ALL, {512ULL << 20, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, IPC,
    BOOL_FALSE},
   {CHANNELS(1), KEEP, AUTO, SENDRECV, WRITE}},

  // Provisioning the wide read-mode pool also widens the default P2P pool.
  // Keep the measured low-size range on two active channels while preserving
  // automatic protocol and transfer-mode selection. AllToAll placement rows
  // are intentionally separate so future out-of-place tuning can diverge.
  {{P2P, "gfx120", ALL_TO_ALL, {128ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8), kAnyInt,
    IPC, BOOL_FALSE},
   {CHANNELS(2), KEEP, AUTO, SENDRECV, AUTO_XFER}},
  {{P2P, "gfx120", ALL_TO_ALL, {128ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8), kAnyInt,
    IPC, BOOL_TRUE},
   {CHANNELS(2), KEEP, AUTO, SENDRECV, AUTO_XFER}},
  {{P2P, "gfx120", SCATTER, {64ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {CHANNELS(2), KEEP, AUTO, SENDRECV, AUTO_XFER}},

  // Read mode only wins these fanout collectives when peers are spread across
  // the wide active pool. Restrict the experiment to the measured topology and
  // size range; the generic write rules below remain the fallback.
  {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, 4ULL << 20}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}},
  // AllToAllv's per-peer tasks are asymmetric, so transfer mode cannot depend
  // on an individual task's byte count without risking endpoint disagreement.
  {{P2P, "gfx120", ALL_TO_ALL_V, kAnyBytes, EQ(1), EQ(8), kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}},
  {{P2P, "gfx120", SCATTER, {1ULL << 20, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}},

  // Runtime IPC/SHM crossover candidates from the September 24 forced-path
  // sweep. SHM is ranked ahead of the broader IPC candidates only where it won
  // by >=3% in both placements at two or more adjacent measured sizes.
  // Arithmetic midpoints provide smooth handoffs between measured sizes.
  {{COLL, "gfx1201", REDUCE, {24ULL << 20, (1536ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(24), RING, SIMPLE, AUTO_PATH, KEEP}},
  // AllReduce, AllGather, and ReduceScatter have no stable SHM-winning
  // crossover. Broadcast also remains IPC below 16 MiB; all four intentionally
  // fall through to their IPC profiles.

  // Eight-rank, single-node gfx1201 P2P/IPC matrix. Ring/protocol ranges use
  // both placements; channel overrides require a >=3% gain in each placement
  // across at least two adjacent measured sizes.
  {{COLL, "gfx120", ALL_GATHER, {128, (16ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {AUTO, RING, LL, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", ALL_GATHER, {16ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    IPC, ANY_BOOL},
   {AUTO, RING, AUTO, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", ALL_GATHER, {32ULL << 10, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx120", ALL_REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {AUTO, RING, LL, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", ALL_REDUCE, {64ULL << 10, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx120", BROADCAST, {128, (64ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", BROADCAST, {64ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    IPC, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", BROADCAST, {512ULL << 10, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx120", REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE, {64ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    IPC, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE, {512ULL << 10, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx120", REDUCE_SCATTER, {128, (8ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, IPC,
    ANY_BOOL},
   {AUTO, RING, LL, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE_SCATTER, {8ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, AUTO, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE_SCATTER, {32ULL << 10, (2ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE_SCATTER, {2ULL << 30, (8ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, IPC, ANY_BOOL},
   {CHANNELS(6), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE_SCATTER, {8ULL << 30, 8ULL << 30}, EQ(1), EQ(8), kAnyInt,
    IPC, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
  {{P2P, "gfx120", ALL_GATHER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
  {{P2P, "gfx120", REDUCE_SCATTER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
  {{P2P, "gfx120", GATHER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
  {{P2P, "gfx120", SCATTER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), IPC, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},

  // Eight-rank, single-node gfx1201 pure-SHM matrix. These ranges follow stable
  // Ring/Simple bus-bandwidth plateaus and merge isolated point winners into
  // smooth size bands.
  {{COLL, "gfx1201", ALL_REDUCE, {128, (256ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", ALL_REDUCE, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(4), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", ALL_REDUCE, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(12), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", ALL_REDUCE, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(24), RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx1201", ALL_GATHER, {128, (256ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", ALL_GATHER, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(6), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", ALL_GATHER, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(12), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", ALL_GATHER, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(28), RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx1201", REDUCE_SCATTER, {128, (256ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", REDUCE_SCATTER, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(2), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", REDUCE_SCATTER, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(12), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", REDUCE_SCATTER, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(28), RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx1201", REDUCE, {128, (256ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", REDUCE, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(2), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", REDUCE, {1ULL << 20, (16ULL << 20) - 1}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(8), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", REDUCE, {16ULL << 20, 8ULL << 30}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(24), RING, SIMPLE, AUTO_PATH, KEEP}},

  // Broadcast's sub-64 KiB channel winner was noisy. Select only its measured
  // Ring/Simple protocol there, then use the stable channel plateaus.
  {{COLL, "gfx1201", BROADCAST, {128, (64ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", BROADCAST, {64ULL << 10, (256ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", BROADCAST, {256ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8),
    kAnyInt, SHM, ANY_BOOL},
   {CHANNELS(2), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", BROADCAST, {1ULL << 20, 8ULL << 30}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {CHANNELS(8), RING, SIMPLE, AUTO_PATH, KEEP}},

  // If a user or per-call channel bound rejects a measured channel profile,
  // retain the SHM algorithm/protocol choice without partially applying the
  // rejected profile. Explicit NCCL_ALGO/NCCL_PROTO conflicts reject these
  // fallbacks as well and preserve the user's selection.
  {{COLL, "gfx1201", ALL_REDUCE, {128, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, SHM,
    ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", ALL_GATHER, {128, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, SHM,
    ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", REDUCE_SCATTER, {128, 8ULL << 30}, EQ(1), EQ(8), kAnyInt,
    SHM, ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", REDUCE, {128, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, SHM,
    ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx1201", BROADCAST, {128, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, SHM,
    ANY_BOOL},
   {KEEP, RING, SIMPLE, AUTO_PATH, KEEP}},
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

const auto kExpandedCollectiveExecutionRules = expandRules(kCollectiveExecutionRules);
const RuleTableView kCollectiveExecutionRuleTable = {
  kCollectiveExecutionRules,
  kExpandedCollectiveExecutionRules.data(),
  kCollectiveExecutionRuleCount,
};

} // namespace rcclExecutionPolicyRules
