/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// RCCL backend scheduler.
//
// Every collective is decided by one loop (rcclSchedule) over its row in the
// comm's arch policy (arch/). A rule is a backend, a size window and the facts it needs or
// avoids. Facts and size windows are in sched_conditions.cc; each backend's own
// checks and launch shape are in backends/. Adding a backend is one struct plus a rule.

#ifndef RCCL_SCHEDULER_H_
#define RCCL_SCHEDULER_H_

#include <cstddef>
#include <cstdint>
#include <string>

#include "comm.h"
#include "rccl_arch_thresholds.h"
#include "rccl_common.h"
#include "rccl_decision.h"

// Facts are evaluated on first use and cached. Within a rule, `avoid` facts
// are tested before `need` facts, each in enum order, stopping at the first
// that rejects. CAPTURING is the only fact with a side effect (the live probe
// ticks the CE graph latch), so row prologues control when it runs.
enum rcclFact : uint8_t {
  RCCL_FACT_ALGO_ENV,         // NCCL_ALGO is set: RCCL backends stand aside
  RCCL_FACT_IN_GROUP,         // inside ncclGroupStart/End
  RCCL_FACT_MUST_ENQUEUE,     // check mode, suspended or revoked comm
  RCCL_FACT_CTA_ZERO,         // CTAPolicy has NCCL_CTA_POLICY_ZERO
  RCCL_FACT_FABRIC,           // gfx1250 (only the symk reroutes ask; rows are per arch)
  RCCL_FACT_IN_PLACE,
  RCCL_FACT_CAPTURING,        // graph capture (live probe, or the query's hint)
  RCCL_FACT_CE_AR_GRAPH_OK,   // the CE AllReduce graph latch allows CE
  RCCL_FACT_SYSMEM,           // an operand window has a sysmem segment
  RCCL_FACT_SYMK_REQUESTED,   // both operands are symmetric windows symk accepts
  RCCL_FACT_SYMK_ELIGIBLE,    // requested and inside the symk size window
  RCCL_FACT_SYMK_VETO,        // symk claims the call, so DDA and host-composed paths stand aside
  RCCL_FACT_CE_AR_2SHOT_OK,   // rcclUseCeAr2Shot
  RCCL_FACT_CE_RS_OK,         // CE ReduceScatter gates (rcclUseCeReduceScatter, graph, group)
  RCCL_FACT_REROUTE_RING,     // gfx1250: symk withdrawn under capture, DDA replaces Ring
  RCCL_FACT_REROUTE_CE_REG,   // gfx1250: symk withdrawn eager, DDA LL128 replaces CE registered
  RCCL_FACT_COUNT
};
typedef uint32_t rcclFactMask;
static_assert(RCCL_FACT_COUNT <= 32, "rcclFactMask is 32 bits");
constexpr rcclFactMask rcclFactBit(rcclFact f) { return 1u << f; }
constexpr rcclFactMask kAlgoEnv = rcclFactBit(RCCL_FACT_ALGO_ENV);
constexpr rcclFactMask kInGroup = rcclFactBit(RCCL_FACT_IN_GROUP);
constexpr rcclFactMask kMustEnqueue = rcclFactBit(RCCL_FACT_MUST_ENQUEUE);
constexpr rcclFactMask kCtaZero = rcclFactBit(RCCL_FACT_CTA_ZERO);
constexpr rcclFactMask kInPlace = rcclFactBit(RCCL_FACT_IN_PLACE);
constexpr rcclFactMask kCapturing = rcclFactBit(RCCL_FACT_CAPTURING);
constexpr rcclFactMask kCeArGraphOk = rcclFactBit(RCCL_FACT_CE_AR_GRAPH_OK);
constexpr rcclFactMask kSysmem = rcclFactBit(RCCL_FACT_SYSMEM);
constexpr rcclFactMask kSymkRequested = rcclFactBit(RCCL_FACT_SYMK_REQUESTED);
constexpr rcclFactMask kSymkEligible = rcclFactBit(RCCL_FACT_SYMK_ELIGIBLE);
constexpr rcclFactMask kSymkVeto = rcclFactBit(RCCL_FACT_SYMK_VETO);
constexpr rcclFactMask kCeAr2ShotOk = rcclFactBit(RCCL_FACT_CE_AR_2SHOT_OK);
constexpr rcclFactMask kCeRsOk = rcclFactBit(RCCL_FACT_CE_RS_OK);
constexpr rcclFactMask kRerouteRing = rcclFactBit(RCCL_FACT_REROUTE_RING);
constexpr rcclFactMask kRerouteCeReg = rcclFactBit(RCCL_FACT_REROUTE_CE_REG);

// Named byte limits. Each resolves against the comm's arch table and the env
// overrides; the loop compares the call's total bytes.
enum rcclLimit : uint8_t {
  RCCL_LIMIT_NONE,
  RCCL_LIMIT_DDA_ENTRY,         // widest enabled DDA tier
  RCCL_LIMIT_DDA_LL,
  RCCL_LIMIT_DDA_LL128,
  RCCL_LIMIT_DDA_VMM,           // fabric VMM on gfx1250, IPC elsewhere
  RCCL_LIMIT_DDA_AR_VMM_FIRST,  // AllReduce VMM ahead of LL on large comms
  RCCL_LIMIT_DDA_SCRATCH,       // fits comm->ddaScratch
  RCCL_LIMIT_CE_REG,
  RCCL_LIMIT_CE_NONREG,
  RCCL_LIMIT_CE_REG_AG,         // AllGather registered CE: recv registered and in the CE window
  RCCL_LIMIT_SYMK,
  RCCL_LIMIT_COUNT
};

enum rcclFit : uint8_t {
  RCCL_FIT_OFF,  // the limit disables the rule for this call
  RCCL_FIT_OUT,  // outside the tuning window
  RCCL_FIT_IN,
};

// Per-call scheduling context.
struct rcclSchedCtx {
  struct ncclComm* comm;
  ncclFunc_t func;
  const char* name;  // "AR", "AG", ...
  const void* sendbuff;
  void* recvbuff;
  size_t count;
  ncclDataType_t datatype;
  ncclRedOp_t op;
  int root;
  cudaStream_t stream;
  bool query;
  bool graphCapturingHint;

  size_t typeSize;
  size_t countBytes;
  size_t totalBytes;  // countBytes * nRanks for AG, RS and A2A
  const rcclArchThresholds* archTable;

  // Runtime bits every candidate carries (ceCapturing, ceArGraphAllowed, symkTdmOnly).
  rcclCollDecision base;
  // Launch bound when DDA replaces another backend.
  rcclDdaLaunchCap ddaCap;

  bool windowsFound;
  struct ncclDevrWindow* sendWin;
  struct ncclDevrWindow* recvWin;
  ncclSymRegType_t winRegType;
  bool hasSysmemSegment;

  bool capturing;
  rcclFactMask factsKnown;
  rcclFactMask factsTrue;
  bool trace;  // RCCL_SCHED_TRACE
};

void rcclSchedCtxInit(rcclSchedCtx* ctx, struct ncclComm* comm, ncclFunc_t func, const void* sendbuff,
                      void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, cudaStream_t stream,
                      bool query, bool graphCapturingHint);
ncclResult_t rcclSchedFindWindows(rcclSchedCtx& ctx);
ncclResult_t rcclFactHolds(rcclSchedCtx& ctx, rcclFact fact, bool* holds);
ncclResult_t rcclLimitFit(rcclSchedCtx& ctx, rcclLimit limit, rcclFit* fit);

// A backend. eligible holds the backend's own checks (including hard size
// limits); choose fills the decision from ctx.base and may decline (*ok = false).
// prepare runs on the live path only, before eligible, and may change comm state.
struct rcclBackend {
  const char* name;        // "ce.registered", "dda.fabric.ll", ...
  rcclBackendMask_t kind;  // RCCL_BACKEND_* bit a preference selects it by; 0 for general
  bool (*eligible)(rcclSchedCtx& ctx);
  ncclResult_t (*choose)(rcclSchedCtx& ctx, rcclCollDecision* d, bool* ok);
  ncclResult_t (*prepare)(rcclSchedCtx& ctx);
  bool usesWindows;
  bool outright = false;  // decides even when another kind is preferred
};

struct rcclRuleGroup;

// A rule fits a call when none of `avoid` holds, all of `need` hold and the
// size is inside `window`. AnySizeWhen facts skip the window; YieldTo facts
// count as outside it. A group rule stands for its rules in order, each with
// the group's conditions added.
struct rcclRule {
  const rcclBackend* backend;
  const rcclRuleGroup* group;
  rcclLimit window;
  rcclFactMask need;
  rcclFactMask avoid;
  rcclFactMask yield;
  rcclFactMask anySize;

  constexpr rcclRule If(rcclFactMask m) const { rcclRule r = *this; r.need |= m; return r; }
  constexpr rcclRule Not(rcclFactMask m) const { rcclRule r = *this; r.avoid |= m; return r; }
  constexpr rcclRule YieldTo(rcclFactMask m) const { rcclRule r = *this; r.yield |= m; return r; }
  constexpr rcclRule AnySizeWhen(rcclFactMask m) const { rcclRule r = *this; r.anySize |= m; return r; }
};
constexpr rcclRule Rule(const rcclBackend& b, rcclLimit window = RCCL_LIMIT_NONE) {
  return rcclRule{&b, nullptr, window, 0, 0, 0, 0};
}
constexpr rcclRule Rule(const rcclRuleGroup& g) { return rcclRule{nullptr, &g, RCCL_LIMIT_NONE, 0, 0, 0, 0}; }
// A group's rule as evaluated: the group's conditions added to its own.
constexpr rcclRule rcclRuleInGroup(const rcclRule& group, const rcclRule& r) {
  return rcclRule{r.backend,          r.group,           r.window, r.need | group.need, r.avoid | group.avoid,
                  r.yield | group.yield, r.anySize | group.anySize};
}

struct rcclRuleGroup {
  const rcclRule* rules;
  size_t nRules;
};

// Row prologue: facts evaluated up front, in order. SkipRulesIf goes straight
// to the fallbacks when its fact holds.
struct rcclStep {
  rcclFact fact;
  bool skipRules;
};
constexpr rcclStep Eval(rcclFact f) { return rcclStep{f, false}; }
constexpr rcclStep SkipRulesIf(rcclFact f) { return rcclStep{f, true}; }

// One collective's policy. Without `preferred` the first rule that fits and
// whose backend is eligible decides. With it, a backend of a preferred kind wins
// even outside its window, else the first one inside its window. Fallbacks are
// tried in order when no rule decides; the last one always decides.
struct rcclRow {
  const char* name;
  const rcclStep* prologue;
  size_t nPrologue;
  const rcclRule* rules;
  size_t nRules;
  const rcclBackend* const* fallbacks;
  size_t nFallbacks;
  rcclBackendMask_t (*preferred)(const struct ncclComm* comm);
};

ncclResult_t rcclSchedule(rcclSchedCtx& ctx, const rcclRow& row, rcclCollDecision* decision);

template <size_t R, size_t F>
constexpr rcclRow Row(const char* name, const rcclRule (&rules)[R], const rcclBackend* const (&fallbacks)[F]) {
  return rcclRow{name, nullptr, 0, rules, R, fallbacks, F, nullptr};
}
template <size_t P, size_t R, size_t F>
constexpr rcclRow Row(const char* name, const rcclStep (&prologue)[P], const rcclRule (&rules)[R],
                      const rcclBackend* const (&fallbacks)[F],
                      rcclBackendMask_t (*preferred)(const struct ncclComm*) = nullptr) {
  return rcclRow{name, prologue, P, rules, R, fallbacks, F, preferred};
}
bool rcclLimitIn(rcclSchedCtx& ctx, rcclLimit limit);

// The rows for one arch (arch/*_rules.cc, next to its arch/*_thresholds.h).
// Partitions share an arch's rows; their thresholds differ through the arch table.
struct rcclPolicy {
  const char* arch;  // nullptr matches any arch
  const rcclRow* allReduce;
  const rcclRow* allGather;
  const rcclRow* reduceScatter;
  const rcclRow* alltoAll;
  const rcclRow* broadcast;
};
extern const rcclPolicy kRcclGfx1250Policy;
extern const rcclPolicy kRcclDefaultPolicy;
const rcclPolicy& rcclPolicyFor(const struct ncclComm* comm);
// Text form of a policy (sched_describe.cc): one line per prologue step, rule
// (groups expanded) and fallback. The host tests pin it.
const char* rcclFactName(rcclFact fact);
const char* rcclLimitName(rcclLimit limit);
std::string rcclPolicyDescribe(const rcclPolicy& policy);

// Backends (backends/*_backends.cc).
extern const rcclBackend kGeneralRing;
extern const rcclBackend kGeneralNative;
extern const rcclBackend kGeneralDirect;
extern const rcclBackend kGeneralHier;
extern const rcclBackend kSymmetricKernel;
extern const rcclBackend kCe2Shot;
extern const rcclBackend kCeStagingInit;
extern const rcclBackend kCeRegistered;
extern const rcclBackend kCeStaged;
extern const rcclBackend kCeHier;
extern const rcclBackend kCeScratch;
extern const rcclBackend kDdaFabricLL;
extern const rcclBackend kDdaFabricLL128;
extern const rcclBackend kDdaFabricVmm;
extern const rcclBackend kDdaIpc;
extern const rcclRuleGroup kDdaFabricTiers;  // LL, then LL128, then VMM, each in its window
extern const rcclBackend kGinSdma;
extern const rcclBackend kGinGda;

// Mechanism queries the facts are built from.
bool rcclSymkRequested(rcclSchedCtx& ctx);
bool rcclSymkEligible(rcclSchedCtx& ctx);
bool rcclCeArGraphOk(rcclSchedCtx& ctx);
bool rcclCeArRegisteredAvailable(rcclSchedCtx& ctx);
bool rcclCeArRegisteredInWindow(rcclSchedCtx& ctx);
int rcclCeArRegisteredBlocks(rcclSchedCtx& ctx);
bool rcclGinArYieldsToDda(rcclSchedCtx& ctx);
ncclResult_t rcclRingFallback(struct ncclComm* comm, const void* sendbuff, void* recvbuff, ncclFunc_t func,
                              size_t count, ncclDataType_t datatype, bool query, struct rcclCollDecision* decision,
                              uint32_t* threadsPerBlock = nullptr);
bool rcclSymkQuery(struct ncclComm* comm, ncclFunc_t coll, uint64_t count, ncclDataType_t dataType, ncclRedOp_t op,
                   int* algo, int* protocol, int* maxChannels, bool symAligned16B = true);

extern ncclResult_t getAlgoInfo(struct ncclComm* comm, struct ncclTaskColl* task, int collNetSupport, int nvlsSupport,
                                int numPipeOps, ncclSimInfo_t* simInfo = NULL);
int rcclKernelPackedChannels(struct ncclComm* comm, ncclFunc_t func, size_t count, ncclDataType_t datatype,
                             int protocol, int nMaxChannels);

// Arch-table thresholds (sched_conditions.cc).
const rcclArchThresholds* extAlgoArchTable(const ncclComm* comm);
bool ddaThresholdFromEnv(int64_t param, size_t* threshold);
size_t funcThresholdFromTable(const size_t* caps, ncclFunc_t func);
size_t rcclSymMaxR2CapTab(const rcclArchThresholds* table, ncclFunc_t func, bool graphMode);
size_t rcclSymMinR2CapTab(const rcclArchThresholds* table, ncclFunc_t func);
size_t rcclAgSymMaxR2Cap(struct ncclComm* comm, const rcclArchThresholds* table, bool graphMode);
size_t rcclAgSymMinR2Cap(struct ncclComm* comm, const rcclArchThresholds* table);
bool rcclSymTdmWindowContains(const rcclArchThresholds* table, ncclFunc_t func, size_t totalBytes, bool spansTrays,
                              bool graphMode);
size_t rcclCeRegMaxTab(const rcclArchThresholds* table, ncclFunc_t func);
size_t rcclCeRegMinTab(const rcclArchThresholds* table, ncclFunc_t func);
size_t rcclCeNonRegMaxTab(const rcclArchThresholds* table, ncclFunc_t func);
size_t rcclCeNonRegMinTab(const rcclArchThresholds* table, ncclFunc_t func);
bool rcclAllGatherCeRegisteredWindowTab(const rcclArchThresholds* table, size_t totalBytes,
                                        ncclSymRegType_t winRegType, bool graphMode);
bool rcclDdaRankScaled(const rcclArchThresholds* table, int nRanks);
size_t ddaTierCapFromTable(const rcclArchThresholds* table, const size_t* totals, const size_t* perRank,
                           ncclFunc_t func, int nRanks);
size_t rcclDdaLLThresholdTab(const rcclArchThresholds* table, ncclFunc_t func, int nRanks);
size_t rcclDdaLL128ThresholdTab(const rcclArchThresholds* table, ncclFunc_t func, int nRanks);
size_t rcclDdaLLTwoShotThresholdTab(const rcclArchThresholds* table, int nRanks);
size_t rcclDdaLL128TwoShotThresholdTab(const rcclArchThresholds* table, int nRanks);
size_t rcclDdaVmmThresholdTab(const rcclArchThresholds* table, ncclFunc_t func, int nRanks);
size_t rcclDdaVmmThresholdCtxTab(const rcclArchThresholds* table, ncclFunc_t func, ncclSymRegType_t winRegType,
                                 bool graphMode, int nRanks);
size_t rcclDdaEntryThresholdTab(const rcclArchThresholds* table, ncclFunc_t func, int nRanks);
size_t rcclDdaArRegCaptureMaxBytesTab(const rcclArchThresholds* table, int nRanks);
bool rcclDdaArRegEagerShardFitsTab(const rcclArchThresholds* table, int nRanks, size_t bytes);
bool rcclDdaArVmmFirstTab(const rcclArchThresholds* table, int nRanks, size_t msgBytes, ncclSymRegType_t winRegType);
bool rcclEnvIsSet(const char* name);

// Scheduler knobs not declared in rccl_common.h (rccl_scheduler.cc).
RCCL_PARAM_DECLARE(DirectAllGatherThreshold);
RCCL_PARAM_DECLARE(DirectAllGatherDisable);
RCCL_PARAM_DECLARE(DirectReduceScatterDisable);
RCCL_PARAM_DECLARE(ForceSymmetric);
RCCL_PARAM_DECLARE(ForceDda);
RCCL_PARAM_DECLARE(DdaLLOneShotThreshold);
RCCL_PARAM_DECLARE(DdaLLTwoShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128OneShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128TwoShotThreshold);
RCCL_PARAM_DECLARE(DdaLL128AsyncWire);
RCCL_PARAM_DECLARE(DdaLL128ArPeerParallel);
RCCL_PARAM_DECLARE(DdaBc);
RCCL_PARAM_DECLARE(DdaBcAllowIpc);
RCCL_PARAM_DECLARE(IgnoreArchTable);
RCCL_PARAM_DECLARE(SchedTrace);
RCCL_PARAM_DECLARE(HierarchicalAllGatherMinBytesPerRank);
RCCL_PARAM_DECLARE(HierarchicalReduceScatterMinBytesPerRank);
RCCL_PARAM_DECLARE(ForceCe);

#endif  // RCCL_SCHEDULER_H_
