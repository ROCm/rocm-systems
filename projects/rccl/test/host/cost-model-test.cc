/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/tuning/cost_model.cc -- the NCCL 2.31.2 unified
// cost model's model registry and dispatch (AICOMRCCL-2007).
//
// The unit under test is modelMap[], which flattens three kernel families into
// one id space:
//   [0, 21)                  general algo*NCCL_NUM_PROTOCOLS + proto
//   [21, 39)                 ncclSymkKernelId_*   (device-API symmetric kernels)
//   [39, NCCL_TUNING_COUNT)  ncclCeMethodId_*     (copy engine)
// Every id is priced by one dispatch, so an off-by-one anywhere in that layout
// silently mis-prices a whole family rather than failing to build. These tests
// pin the layout and the dispatch to it.
//
// Scoped to avoid the neighbours: ncclTuningExpandId's own decode/reject
// behaviour is tuning-general-test.cc (AICOMRCCL-2400), and ncclTuningCompute's
// selection logic is AICOMRCCL-2448. What is tested here is the registry table
// itself and ncclTuningCostModelSimModel's routing into it.

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <memory>

#include "comm.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "sym_kernels.h"

// src/init.cc:114 defines this and nothing on the micro link line does. parseList()
// takes it as its prefix table, so NCCL_ALGO="allreduce:tree" cannot parse without
// it. Mirrors production exactly; RcclAssertSourceLine in CMakeLists.txt pins the
// first line, and the static_assert below pins the count, so a sixth collective
// whose name lands on a continuation line cannot slip past both.
const char* ncclFuncStr[NCCL_NUM_FUNCTIONS + 4] = {"Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce",
                                                   "AlltoAllPivot", "AlltoAllGda", "AlltoAllvGda",
                                                   "SendRecv"};
static_assert(NCCL_NUM_FUNCTIONS == 5, "ncclFuncStr mirrors src/init.cc; add the new function name");

// src/sym_kernels.cc:390 defines this, and nothing on this link line does: the generated
// sym_kernels_host.cc carries only ncclSymkGetKernelIndex and the kernel-list tables.
// cost_model.cc uses it solely in a TRACE, so mirror production's lookup rather than pulling
// in sym_kernels_index_fakes.cc, whose ncclSymkGetKernelIndex/ncclSymkKernelList* would
// collide with the generated file.
const char* ncclSymkKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymkKernelId_Count) return "Unknown";
  return ncclSymKernelStr[kernelId];
}

namespace {

// ---------------------------------------------------------------------------
// Model-function seams.
//
// modelMap[] stores pointers to the per-family model functions, which live in
// sibling TUs (ring.cc, tree.cc, ...) that are not on this link line. Defining
// them here is both what makes the TU link and what makes dispatch observable:
// the table points at these, so a call through modelMap[] lands in the matching
// counter below.
// ---------------------------------------------------------------------------

enum class ModelKind { None, Tree, Ring, Collnet, Nvls, Pat, Symk, Ce };

ModelKind g_lastModel = ModelKind::None;
float g_modelTimeUs = 42.0f;
ncclResult_t g_modelRet = ncclSuccess;
int g_initCalls[NCCL_TUNING_COUNT];
int g_initCallTotal = 0;

void ResetModelFakes() {
  g_lastModel = ModelKind::None;
  g_modelTimeUs = 42.0f;
  g_modelRet = ncclSuccess;
  memset(g_initCalls, 0, sizeof(g_initCalls));
  g_initCallTotal = 0;
}

ncclResult_t RecordInit(int id) {
  if (id >= 0 && id < NCCL_TUNING_COUNT) g_initCalls[id]++;
  g_initCallTotal++;
  return ncclSuccess;
}

ncclResult_t RecordSim(ModelKind kind, struct ncclTuningResult_t* tuning) {
  g_lastModel = kind;
  tuning->timeUs = g_modelTimeUs;
  return g_modelRet;
}

}  // namespace

// Name is both the production symbol infix and the ModelKind enumerator; the five match
// exactly, so one parameter cannot be paired with the wrong family.
#define DEFINE_MODEL_FAKE(Name)                                                                  \
  ncclResult_t ncclTuning##Name##ModelInit(struct ncclComm*, int id, int[NCCL_NUM_FUNCTIONS]) {   \
    return RecordInit(id);                                                                       \
  }                                                                                              \
  ncclResult_t ncclTuning##Name##ModelSim(struct ncclTuningInput_t* const,                       \
                                          struct ncclTuningResult_t* const tuning) {             \
    return RecordSim(ModelKind::Name, tuning);                                                   \
  }

DEFINE_MODEL_FAKE(Tree)
DEFINE_MODEL_FAKE(Ring)
DEFINE_MODEL_FAKE(Collnet)
DEFINE_MODEL_FAKE(Nvls)
DEFINE_MODEL_FAKE(Pat)
#undef DEFINE_MODEL_FAKE

// Symk and CE have no init entry in modelMap[], only a sim.
ncclResult_t ncclTuningSymkModelSim(struct ncclTuningInput_t* const, struct ncclTuningResult_t* const tuning) {
  return RecordSim(ModelKind::Symk, tuning);
}
ncclResult_t ncclTuningCeModelSim(struct ncclTuningInput_t* const, struct ncclTuningResult_t* const tuning) {
  return RecordSim(ModelKind::Ce, tuning);
}

#include COST_MODEL_CC_PATH
// After cost_model.cc: algorithm_registry.cc defines unguarded ALGBIT/F_* macros
// and only the lookup helpers below are needed from it.
#include ALGORITHM_REGISTRY_CC_PATH

namespace {

// The id of the first CE method; the registry deliberately stops here.
constexpr int kCeOffset = NCCL_TUNING_CE_METHOD_ID_OFFSET;
constexpr int kSymOffset = NCCL_TUNING_SYM_KERNEL_ID_OFFSET;
// The one general row every dispatch case drives; Ring/Simple is modelled on every config.
constexpr int kRingSimple = NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE;

class CostModelMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetEnvFakes();
    ResetModelFakes();
    comm_ = MakeComm();
  }

  void TearDown() override {
    ResetModelFakes();
    ResetEnvFakes();
    ResetNcclFakes();
  }

  // LL128's default-enable rule reads minCompCap/maxCompCap; a matched pair above
  // Hopper keeps isLL128Enabled() from vetoing for reasons unrelated to the test.
  static std::unique_ptr<ncclComm> MakeComm() {
    auto comm = std::make_unique<ncclComm>();
    comm->minCompCap = comm->maxCompCap = 90;
    comm->nRanks = 8;
    return comm;
  }

  static void SetGraphPaths(ncclComm* comm, int typeInter, int typeIntra) {
    for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
      comm->graphs[a].typeInter = typeInter;
      comm->graphs[a].typeIntra = typeIntra;
    }
  }

  // Drive one id through the dispatcher with everything enabled.
  ncclTuningResult_t Sim(int id, ncclFunc_t func = ncclFuncAllReduce) {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.func = func;
    ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
    result.id = id;
    result.valid = 1;
    // Production's only caller decodes the id into result before dispatching
    // (tuning.cc:141); it never passes an out-of-range id, so ignore that reject.
    (void)ncclTuningExpandId(id, &result.algo, &result.proto, &result.symKernelId, &result.ceMethodId);
    g_lastModel = ModelKind::None;
    lastRet_ = ncclTuningCostModelSimModel(id, &input, &result);
    return result;
  }

  void EnableEverything() {
    for (int i = 0; i < NCCL_TUNING_COUNT; i++) {
      for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) comm_->tuningContext.enabled[i][f] = 1;
    }
  }

  // The family a given id must route to, derived from the id space rather than
  // from modelMap[] -- otherwise the test would just restate the table.
  static ModelKind ExpectedKind(int id) {
    if (id >= kCeOffset) return ModelKind::Ce;
    if (id >= kSymOffset) return ModelKind::Symk;
    switch (id / NCCL_NUM_PROTOCOLS) {
    case NCCL_ALGO_TREE: return ModelKind::Tree;
    case NCCL_ALGO_RING: return ModelKind::Ring;
    case NCCL_ALGO_COLLNET_DIRECT:
    case NCCL_ALGO_COLLNET_CHAIN: return ModelKind::Collnet;
    case NCCL_ALGO_NVLS:
    case NCCL_ALGO_NVLS_TREE: return ModelKind::Nvls;
    case NCCL_ALGO_PAT: return ModelKind::Pat;
    default: return ModelKind::None;
    }
  }

  std::unique_ptr<ncclComm> comm_;
  ncclResult_t lastRet_ = ncclSuccess;
};

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

// getModelEntry() bounds-checks against NCCL_TUNING_COUNT and indexes modelMap[] with the
// result, so a table shorter than the id space reads out of bounds instead of failing.
// Adding an ncclSymkKernelId_* without a modelMap row is exactly that mistake, and it must
// stop the build rather than let 17 other cases index past the end first.
static_assert(sizeof(modelMap) / sizeof(modelMap[0]) == static_cast<size_t>(NCCL_TUNING_COUNT),
              "modelMap[] must carry exactly one row per tuning id");

// The three family offsets are pure macro algebra over the same two counts (tuning.h:17-19),
// as are the masks derived from them, so these are build-time facts, not runtime ones.
static_assert(kSymOffset == NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS, "sym kernels follow the general rows");
static_assert(kCeOffset == kSymOffset + ncclSymkKernelId_Count, "CE methods follow the sym kernels");
static_assert(NCCL_TUNING_COUNT == kCeOffset + ncclCeMethodId_Count, "the id space ends after the CE methods");
static_assert(NCCL_TUNING_MASK_ALL ==
                  (NCCL_TUNING_MASK_GENERAL_KERNELS | NCCL_TUNING_MASK_SYM_KERNELS | NCCL_TUNING_MASK_CE),
              "the three family masks must cover the id space");
static_assert((NCCL_TUNING_MASK_GENERAL_KERNELS & NCCL_TUNING_MASK_SYM_KERNELS) == 0 &&
                  (NCCL_TUNING_MASK_GENERAL_KERNELS & NCCL_TUNING_MASK_CE) == 0 &&
                  (NCCL_TUNING_MASK_SYM_KERNELS & NCCL_TUNING_MASK_CE) == 0,
              "the three family masks must not overlap");

TEST_F(CostModelMicrotest, FamilyMasksPartitionTheIdSpace) {
  // Callers pass these straight to ncclTuningCompute as tuningMask, so each
  // family's bits must sit where ncclTuningExpandId decodes that family. This is
  // the half that can go red: tuning_general.cc:212/:216 spell the two family
  // boundaries as their own expressions rather than reusing the macros.
  for (int i = 0; i < NCCL_TUNING_COUNT; i++) {
    uint64_t bit = 1ull << i;
    int algo = -9, proto = -9, sym = -9, ce = -9;
    ASSERT_EQ(ncclSuccess, ncclTuningExpandId(i, &algo, &proto, &sym, &ce)) << "id " << i;
    if (bit & NCCL_TUNING_MASK_GENERAL_KERNELS) {
      EXPECT_NE(NCCL_ALGO_UNDEF, algo) << "id " << i;
    } else if (bit & NCCL_TUNING_MASK_SYM_KERNELS) {
      EXPECT_NE(ncclSymkKernelId_Count, sym) << "id " << i;
    } else {
      EXPECT_NE(ncclCeMethodId_Count, ce) << "id " << i;
    }
  }
}

// ---------------------------------------------------------------------------
// modelMap[] <-> algRegistry[] correspondence
//
// cost_model.cc and algorithm_registry.cc each carry a comment demanding the
// other stay in sync, and nothing enforces it. The registry is the user-facing
// half -- ncclCollConfig_t::algSelection resolves names through it -- so a drift
// means a name the user can ask for that the cost model will not price, or a
// kernel the cost model prices that no name can reach.
// ---------------------------------------------------------------------------

TEST_F(CostModelMicrotest, RegistryNamesEveryModelledGeneralIdAndNoOther) {
  for (int id = 0; id < kSymOffset; id++) {
    int algo = -9, proto = -9, sym = -9, ce = -9;
    ASSERT_EQ(ncclSuccess, ncclTuningExpandId(id, &algo, &proto, &sym, &ce));
    const char* name = ncclAlgNameForGeneral(algo, proto);
    const bool modelled = modelMap[id].model != nullptr;

    EXPECT_EQ(modelled, name != nullptr)
        << "tuning id " << id << " (algo " << algo << " proto " << proto << "): modelMap "
        << (modelled ? "prices" : "ignores") << " it but the registry "
        << (name ? "names" : "omits") << " it";

    // The row's mask bit IS the tuning id; that is the whole correspondence.
    if (name != nullptr) EXPECT_NE(0ull, ncclAlgTagMask(name) & (1ull << id)) << name << " -> id " << id;
  }
}

TEST_F(CostModelMicrotest, RegistryNamesEverySymmetricKernelId) {
  for (int id = kSymOffset; id < kCeOffset; id++) {
    int algo = -9, proto = -9, sym = -9, ce = -9;
    ASSERT_EQ(ncclSuccess, ncclTuningExpandId(id, &algo, &proto, &sym, &ce));
    const char* name = ncclAlgNameForSymk(sym);
    ASSERT_NE(nullptr, name) << "symKernelId " << sym << " (tuning id " << id << ") has no registry row";
    EXPECT_NE(0ull, ncclAlgTagMask(name) & (1ull << id)) << name << " -> id " << id;
    // Every symmetric row is priced; modelMap has no null symk entries.
    EXPECT_NE(nullptr, modelMap[id].model) << "tuning id " << id;
  }
}

TEST_F(CostModelMicrotest, RegistryCollMaskMirrorsModelMapEnabledColumns) {
  // algorithm_registry.cc: "collMask mirrors modelMap[].enabled[func]". This is
  // the assertion that sentence is worth.
  for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
    uint64_t valid = ncclAlgValidForFuncMask(static_cast<ncclFunc_t>(f));
    for (int id = 0; id < kCeOffset; id++) {
      bool inRegistry = (valid >> id) & 1;
      bool inModelMap = modelMap[id].enabled[f] != 0;
      EXPECT_EQ(inRegistry, inModelMap)
          << "func " << ncclFuncStr[f] << " tuning id " << id << ": registry says "
          << (inRegistry ? "valid" : "invalid") << ", modelMap says " << (inModelMap ? "enabled" : "disabled");
    }
  }
}

TEST_F(CostModelMicrotest, CopyEngineIdsAreDeliberatelyOutsideTheRegistry) {
  // algorithm_registry.h scopes the registry to [0,39) and cost_model.cc's
  // "TODO: allow NCCL_ALGO=CE" is the other half of the same gap: CE methods are
  // priced but cannot be named. Pinned so that adding CE rows forces this test to
  // be revisited alongside the TODO rather than leaving the two halves to drift.
  for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
    uint64_t valid = ncclAlgValidForFuncMask(static_cast<ncclFunc_t>(f));
    EXPECT_EQ(0ull, valid & NCCL_TUNING_MASK_CE) << "func " << ncclFuncStr[f];
  }
  EXPECT_EQ(0ull, ncclAlgAllBits() & NCCL_TUNING_MASK_CE);

  // ...yet they are modelled, which is why the gap is a gap and not dead code.
  for (int id = kCeOffset; id < NCCL_TUNING_COUNT; id++) {
    EXPECT_NE(nullptr, modelMap[id].model) << "tuning id " << id;
  }
}

TEST_F(CostModelMicrotest, TagPrefixMatchStaysInsideItsFamily) {
  // ncclAlgTagMask documents a name-prefix match: "RING" must span the three RING_*
  // rows without reaching SYMK_RailRing_LsaSTMC, whose name merely contains "Ring".
  uint64_t ring = ncclAlgTagMask("RING");
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    EXPECT_NE(0ull, ring & (1ull << (NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + p))) << "proto " << p;
  }
  EXPECT_EQ(0ull, ring & NCCL_TUNING_MASK_SYM_KERNELS);

  // "NVLS" spans NVLS_SIMPLE and NVLSTREE_SIMPLE, also by prefix.
  uint64_t nvls = ncclAlgTagMask("NVLS");
  EXPECT_NE(0ull, nvls & (1ull << (NCCL_ALGO_NVLS * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE)));
  EXPECT_NE(0ull, nvls & (1ull << (NCCL_ALGO_NVLS_TREE * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE)));

  EXPECT_EQ(0ull, ncclAlgTagMask("NOT_AN_ALGORITHM"));
  EXPECT_EQ(0ull, ncclAlgTagMask(""));
  EXPECT_EQ(0ull, ncclAlgTagMask(nullptr));
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

TEST_F(CostModelMicrotest, EveryIdDispatchesToItsFamilyModel) {
  EnableEverything();
  for (int id = 0; id < NCCL_TUNING_COUNT; id++) {
    ncclTuningResult_t result = Sim(id);
    if (modelMap[id].model == nullptr) {
      EXPECT_EQ(ModelKind::None, g_lastModel) << "tuning id " << id << " has no model but something ran";
      continue;
    }
    EXPECT_EQ(ExpectedKind(id), g_lastModel) << "tuning id " << id << " routed to the wrong family";
    EXPECT_EQ(1, result.valid) << "tuning id " << id;
    EXPECT_FLOAT_EQ(g_modelTimeUs, result.timeUs) << "tuning id " << id;
  }
}

TEST_F(CostModelMicrotest, UnimplementedCombinationsAreIgnoredNotFree) {
  // The ten {nullptr,...} rows are combinations that do not exist (CollNet and
  // NVLS under LL/LL128, PAT under LL/LL128). ncclTuningSelectBestTuning takes an
  // argmin over timeUs, so leaving these at a default 0 would make an unbuildable
  // kernel beat every real one.
  EnableEverything();
  int nullRows = 0;
  for (int id = 0; id < kSymOffset; id++) {
    if (modelMap[id].model != nullptr) continue;
    nullRows++;
    ncclTuningResult_t result = Sim(id);
    EXPECT_EQ(0, result.valid) << "tuning id " << id;
    EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, result.timeUs) << "tuning id " << id;
  }
  // CollNetDirect/Chain, NVLS/NVLSTree under LL and LL128, plus PAT under both.
  EXPECT_EQ(10, nullRows);
}

TEST_F(CostModelMicrotest, OutOfRangeIdIsRejectedRatherThanIndexed) {
  EnableEverything();
  for (int id : {-1, NCCL_TUNING_COUNT, NCCL_TUNING_COUNT + 1}) {
    ncclTuningResult_t result = Sim(id);
    EXPECT_EQ(ncclInvalidArgument, lastRet_) << "id " << id;
    EXPECT_EQ(0, result.valid) << "id " << id;
    EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, result.timeUs) << "id " << id;
  }
}

TEST_F(CostModelMicrotest, DisabledIdNeverReachesItsModel) {
  EnableEverything();
  const int id = kRingSimple;
  ASSERT_EQ(1, Sim(id).valid);

  comm_->tuningContext.enabled[id][ncclFuncAllReduce] = 0;
  ncclTuningResult_t result = Sim(id);
  EXPECT_EQ(ModelKind::None, g_lastModel);
  EXPECT_EQ(0, result.valid);
  EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, result.timeUs);

  // Per-function, not global: another collective still prices it.
  EXPECT_EQ(1, Sim(id, ncclFuncAllGather).valid);
}

TEST_F(CostModelMicrotest, NonPositiveModelTimeIsTreatedAsUnavailable) {
  // A model that cannot run the input reports it by returning a non-positive
  // time; without this the argmin would pick the candidate that opted out. The
  // test is the SIGN boundary, not the sentinel: NCCL_TUNING_IGNORE is itself
  // -1.0, so `timeUs <= 0.0` cannot distinguish it from a genuine negative.
  EnableEverything();
  for (float t : {0.0f, -1.0f}) {
    g_modelTimeUs = t;
    ncclTuningResult_t result = Sim(kRingSimple);
    EXPECT_EQ(ModelKind::Ring, g_lastModel) << "time " << t;
    EXPECT_EQ(0, result.valid) << "time " << t;
    EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, result.timeUs) << "time " << t;
  }

  // The accept side of the same boundary: the smallest positive time still counts.
  g_modelTimeUs = std::numeric_limits<float>::min();
  ncclTuningResult_t accepted = Sim(kRingSimple);
  EXPECT_EQ(1, accepted.valid);
  EXPECT_FLOAT_EQ(g_modelTimeUs, accepted.timeUs);
}

TEST_F(CostModelMicrotest, ModelErrorInvalidatesTheCandidateAndPropagates) {
  EnableEverything();
  g_modelRet = ncclInternalError;
  ncclTuningResult_t result = Sim(kRingSimple);
  EXPECT_EQ(ncclInternalError, lastRet_);
  EXPECT_EQ(0, result.valid);
  EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, result.timeUs);
}

// ---------------------------------------------------------------------------
// Init / finalize
// ---------------------------------------------------------------------------

TEST_F(CostModelMicrotest, InitSeedsEnabledColumnsFromModelMapAndRunsEachInitOnce) {
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));

  int expectedInits = 0;
  for (int id = 0; id < NCCL_TUNING_COUNT; id++) {
    const int expect = modelMap[id].init != nullptr ? 1 : 0;
    expectedInits += expect;
    EXPECT_EQ(expect, g_initCalls[id]) << "tuning id " << id;
  }
  // g_initCalls only counts ids inside the space, so the total is the one place an
  // init call on an out-of-range id would show up.
  EXPECT_EQ(expectedInits, g_initCallTotal);

  // enabled[][] starts as a copy of modelMap[].enabled, so it must match in BOTH
  // directions on this comm: checking only the zeros would let init drop an
  // enabled candidate silently.
  for (int id = 0; id < NCCL_TUNING_COUNT; id++) {
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
      EXPECT_EQ(modelMap[id].enabled[f] != 0, comm_->tuningContext.enabled[id][f] != 0)
          << "tuning id " << id << " func " << f;
    }
  }
}

TEST_F(CostModelMicrotest, InitDisablesLl128ByDefaultButHonoursAnExplicitRequest) {
  // protoEnable == 2 means "defaulted"; only then may isLL128Enabled() veto. With
  // LL128_C2C off, that veto caps inter-node reach at PXB, so a PXN graph falls
  // outside it. Intra stays at PATH_NVB so the separate intra gate passes and the
  // C2C branch is what actually decides.
  g_loadParam = [](const char* env, int64_t deftVal) {
    return strcmp(env, "LL128_C2C") == 0 ? int64_t{0} : deftVal;
  };
  SetGraphPaths(comm_.get(), PATH_PXN, PATH_NVB);
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));
  const int ll128 = NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_LL128;
  EXPECT_EQ(0, comm_->tuningContext.enabled[ll128][ncclFuncAllReduce]);

  SetMicroEnv("NCCL_PROTO", "LL128");
  comm_ = MakeComm();
  SetGraphPaths(comm_.get(), PATH_PXN, PATH_NVB);
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));
  EXPECT_EQ(1, comm_->tuningContext.enabled[ll128][ncclFuncAllReduce]);
}

TEST_F(CostModelMicrotest, AlgoEnvNarrowsSelectionToTheNamedAlgorithm) {
  SetMicroEnv("NCCL_ALGO", "ring");
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));

  EXPECT_EQ(1, comm_->tuningContext.forced[ncclFuncAllReduce]);
  const int treeSimple = NCCL_ALGO_TREE * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE;
  EXPECT_EQ(1, comm_->tuningContext.enabled[kRingSimple][ncclFuncAllReduce]);
  EXPECT_EQ(0, comm_->tuningContext.enabled[treeSimple][ncclFuncAllReduce]);

  // Naming an algorithm zeroes the symmetric-kernel enables too, so a general
  // NCCL_ALGO does not silently leave device-API kernels in the running.
  for (int id = kSymOffset; id < kCeOffset; id++) {
    EXPECT_EQ(0, comm_->tuningContext.enabled[id][ncclFuncAllReduce]) << "tuning id " << id;
  }
}

TEST_F(CostModelMicrotest, SymKernelEnvNarrowsSelectionToTheNamedKernel) {
  SetMicroEnv("NCCL_SYM_KERNEL", "AllReduce_AGxLL_R");
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));

  const int chosen = kSymOffset + ncclSymkKernelId_AllReduce_AGxLL_R;
  EXPECT_EQ(1, comm_->tuningContext.enabled[chosen][ncclFuncAllReduce]);
  const int other = kSymOffset + ncclSymkKernelId_AllReduce_RSxLD_AGxST;
  EXPECT_EQ(0, comm_->tuningContext.enabled[other][ncclFuncAllReduce]);
  // General kernels are excluded as well -- the two enables share one forced flag.
  EXPECT_EQ(0, comm_->tuningContext.enabled[kRingSimple][ncclFuncAllReduce]);
}

TEST_F(CostModelMicrotest, NoModelMapRowCarriesAFinalizeYet) {
  // ncclTuningCostModelFinalize's loop body only runs for rows with a finalize,
  // and there are none, so the function is unobservable today. Pin that, so adding
  // a finalize forces this test to grow a fake and start asserting the call.
  // Note the return value is not asserted: cost_model.cc's `exit:` returns the
  // ncclSuccess literal rather than ret, and `fail:` falls into it, so the call
  // cannot report an error. That swallowed error is a separate follow-up.
  for (int id = 0; id < NCCL_TUNING_COUNT; id++) {
    EXPECT_EQ(nullptr, modelMap[id].finalize) << "tuning id " << id << " gained a finalize; extend this test";
  }
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));
  (void)ncclTuningCostModelFinalize(comm_.get());
}

}  // namespace
