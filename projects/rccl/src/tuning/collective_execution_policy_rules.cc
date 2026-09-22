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
//   P2P enabled        BOOL_TRUE, BOOL_FALSE, or ANY_BOOL.
//   in-place           BOOL_TRUE when buffers overlap, BOOL_FALSE when
//                      disjoint, or ANY_BOOL when placement is irrelevant.
//
// Profile fields, in order:
//   channels           Scope-local channel count.
//   algorithm          RING or another NCCL_ALGO_* alias.
//   protocol           SIMPLE, LL, or another NCCL_PROTO_* alias.
//   path               SENDRECV or AUTO_PATH.
//   transfer mode      READ, WRITE, or AUTO_XFER.
//
// KEEP omits that assignment and preserves RCCL's existing choice. AUTO is an
// explicit -1 automatic selection for channels, algorithm, or protocol. A
// profile is selected atomically; fields from different rules are never merged.
//
// P2P example (single-node, eight-rank, >=1 MiB, two-channel write):
//   {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, SIZE_MAX}, EQ(1), EQ(8),
//     kAnyInt, BOOL_TRUE, ANY_BOOL},
//    {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
//
// Collective example (single-node, eight-rank Ring/Simple):
//   {{COLL, "gfx120", ALL_REDUCE, {128, 8ULL << 30}, EQ(1), EQ(8),
//     kAnyInt, BOOL_TRUE, ANY_BOOL},
//    {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},
constexpr RuleSpec kCollectiveExecutionRules[] = {
  // Preserve the gfx110x AllToAll variance workaround from #11803 as a
  // direction-local channel cap. PATH remains AUTO so existing collective
  // routing (pivot, CE, GDA, GIN, and DDA) is not overridden.
  {{P2P, "gfx110", ALL_TO_ALL, kAnyBytes, kAnyInt, kAnyInt, kAnyInt, BOOL_TRUE, ANY_BOOL},
   {CHANNELS(1), KEEP, KEEP, KEEP, KEEP}},

  // Large out-of-place AllToAll is stable on one channel; two channels produce
  // severe long-tail stalls while in-place execution remains stable.
  {{P2P, "gfx120", ALL_TO_ALL, {512ULL << 20, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    BOOL_FALSE},
   {CHANNELS(1), KEEP, KEEP, SENDRECV, WRITE}},

  // Provisioning the wide read-mode pool also widens the default P2P pool.
  // Keep the measured low-size range on two active channels while preserving
  // automatic protocol and transfer-mode selection. AllToAll placement rows
  // are intentionally separate so future out-of-place tuning can diverge.
  {{P2P, "gfx120", ALL_TO_ALL, {128ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8), kAnyInt,
    BOOL_TRUE, BOOL_FALSE},
   {CHANNELS(2), KEEP, KEEP, SENDRECV, AUTO_XFER}},
  {{P2P, "gfx120", ALL_TO_ALL, {128ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8), kAnyInt,
    BOOL_TRUE, BOOL_TRUE},
   {CHANNELS(2), KEEP, KEEP, SENDRECV, AUTO_XFER}},
  {{P2P, "gfx120", SCATTER, {64ULL << 10, (1ULL << 20) - 1}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {CHANNELS(2), KEEP, KEEP, SENDRECV, KEEP}},

  // Read mode only wins these fanout collectives when peers are spread across
  // the wide active pool. Restrict the experiment to the measured topology and
  // size range; the generic write rules below remain the fallback.
  {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, 4ULL << 20}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}},
  // AllToAllv's per-peer tasks are asymmetric, so transfer mode cannot depend
  // on an individual task's byte count without risking endpoint disagreement.
  {{P2P, "gfx120", ALL_TO_ALL_V, kAnyBytes, EQ(1), EQ(8), kAnyInt, BOOL_TRUE, ANY_BOOL},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}},
  {{P2P, "gfx120", SCATTER, {1ULL << 20, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {CHANNELS(8), KEEP, SIMPLE, SENDRECV, READ}},

  // Eight-rank, single-node gfx1201 P2P/IPC matrix. Ring/protocol ranges use
  // both placements; channel overrides require a >=3% gain in each placement
  // across at least two adjacent measured sizes.
  {{COLL, "gfx120", ALL_GATHER, {128, (16ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {AUTO, RING, LL, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", ALL_GATHER, {16ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    BOOL_TRUE, ANY_BOOL},
   {AUTO, RING, AUTO, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", ALL_GATHER, {32ULL << 10, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx120", ALL_REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {AUTO, RING, LL, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", ALL_REDUCE, {64ULL << 10, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx120", BROADCAST, {128, (64ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", BROADCAST, {64ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    BOOL_TRUE, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", BROADCAST, {512ULL << 10, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx120", REDUCE, {128, (64ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE, {64ULL << 10, (512ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt,
    BOOL_TRUE, ANY_BOOL},
   {CHANNELS(1), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE, {512ULL << 10, 8ULL << 30}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{COLL, "gfx120", REDUCE_SCATTER, {128, (8ULL << 10) - 1}, EQ(1), EQ(8), kAnyInt, BOOL_TRUE,
    ANY_BOOL},
   {AUTO, RING, LL, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE_SCATTER, {8ULL << 10, (32ULL << 10) - 1}, EQ(1), EQ(8),
    kAnyInt, BOOL_TRUE, ANY_BOOL},
   {AUTO, RING, AUTO, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE_SCATTER, {32ULL << 10, (2ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, BOOL_TRUE, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE_SCATTER, {2ULL << 30, (8ULL << 30) - 1}, EQ(1), EQ(8),
    kAnyInt, BOOL_TRUE, ANY_BOOL},
   {CHANNELS(6), RING, SIMPLE, AUTO_PATH, KEEP}},
  {{COLL, "gfx120", REDUCE_SCATTER, {8ULL << 30, 8ULL << 30}, EQ(1), EQ(8), kAnyInt,
    BOOL_TRUE, ANY_BOOL},
   {AUTO, RING, SIMPLE, AUTO_PATH, KEEP}},

  {{P2P, "gfx120", ALL_TO_ALL, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), BOOL_TRUE, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
  {{P2P, "gfx120", ALL_GATHER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), BOOL_TRUE, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
  {{P2P, "gfx120", REDUCE_SCATTER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), BOOL_TRUE, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
  {{P2P, "gfx120", GATHER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), BOOL_TRUE, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
  {{P2P, "gfx120", SCATTER, {1ULL << 20, SIZE_MAX}, EQ(1), GE(3), GE(3), BOOL_TRUE, ANY_BOOL},
   {CHANNELS(2), KEEP, SIMPLE, SENDRECV, WRITE}},
};

constexpr size_t kCollectiveExecutionRuleCount =
  sizeof(kCollectiveExecutionRules) / sizeof(kCollectiveExecutionRules[0]);
constexpr size_t kFirstInvalidRule = firstInvalidRule(kCollectiveExecutionRules);
static_assert(kCollectiveExecutionRuleCount <= RCCL_EXECUTION_POLICY_NO_RULE,
              "Execution policy rule count exceeds the RuleId range");
static_assert(kFirstInvalidRule == kCollectiveExecutionRuleCount,
              "Invalid execution policy rule; compiler diagnostic shows its array index");

const auto kExpandedCollectiveExecutionRules = expandRules(kCollectiveExecutionRules);
const RuleTableView kCollectiveExecutionRuleTable = {
  kCollectiveExecutionRules,
  kExpandedCollectiveExecutionRules.data(),
  kCollectiveExecutionRuleCount,
};

} // namespace rcclExecutionPolicyRules
