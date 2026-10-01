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
#include <memory>

#include "comm.h"
#include "sym_kernels.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"

// src/init.cc:113 defines this and nothing on the micro link line does. parseList()
// takes it as its prefix table, so NCCL_ALGO="allreduce:tree" cannot parse without
// it. Mirrors production exactly; RcclAssertSourceLine in CMakeLists.txt pins the
// two together.
const char* ncclFuncStr[NCCL_NUM_FUNCTIONS + 4] = {"Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce",
                                                   "AlltoAllPivot", "AlltoAllGda", "AlltoAllvGda",
                                                   "SendRecv"};

// src/sym_kernels.cc:383 defines this. sym_kernels_fakes.cc omits it on the theory that the real
// generated sym_kernels_host.cc (which this binary does compile) supplies it -- it does not; only
// the index/table symbols live there. cost_model.cc uses it solely in a TRACE, so mirror
// production's lookup rather than pulling in sym_kernels_index_fakes.cc, whose
// ncclSymkGetKernelIndex/ncclSymkKernelList* would collide with the generated file.
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

#define DEFINE_MODEL_FAKE(Name, Kind)                                                            \
  ncclResult_t ncclTuning##Name##ModelInit(struct ncclComm*, int id, int[NCCL_NUM_FUNCTIONS]) {   \
    return RecordInit(id);                                                                       \
  }                                                                                              \
  ncclResult_t ncclTuning##Name##ModelSim(struct ncclTuningInput_t* const,                       \
                                          struct ncclTuningResult_t* const tuning) {             \
    return RecordSim(ModelKind::Kind, tuning);                                                    \
  }

DEFINE_MODEL_FAKE(Tree, Tree)
DEFINE_MODEL_FAKE(Ring, Ring)
DEFINE_MODEL_FAKE(Collnet, Collnet)
DEFINE_MODEL_FAKE(Nvls, Nvls)
DEFINE_MODEL_FAKE(Pat, Pat)
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

class CostModelMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetEnvFakes();
    ResetModelFakes();
    comm_ = std::make_unique<ncclComm>();
    // LL128's default-enable rule reads these; a matched pair above Hopper keeps
    // isLL128Enabled() from disabling LL128 for reasons unrelated to the test.
    comm_->minCompCap = comm_->maxCompCap = 90;
    comm_->nRanks = 8;
  }

  void TearDown() override {
    ResetModelFakes();
    ResetEnvFakes();
    ResetNcclFakes();
  }

  // Drive one id through the dispatcher with everything enabled.
  ncclTuningResult_t Sim(int id, ncclFunc_t func = ncclFuncAllReduce) {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.func = func;
    ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
    result.id = id;
    result.valid = 1;
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

TEST_F(CostModelMicrotest, ModelMapHasExactlyOneEntryPerTuningId) {
  // getModelEntry() bounds-checks against NCCL_TUNING_COUNT and indexes
  // modelMap[] with the result, so a table shorter than the id space reads out
  // of bounds instead of failing. Adding an ncclSymkKernelId_* without a
  // modelMap row is exactly that mistake.
  EXPECT_EQ(static_cast<size_t>(NCCL_TUNING_COUNT), sizeof(modelMap) / sizeof(modelMap[0]));

  EXPECT_EQ(NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS, kSymOffset);
  EXPECT_EQ(kSymOffset + ncclSymkKernelId_Count, kCeOffset);
  EXPECT_EQ(kCeOffset + ncclCeMethodId_Count, NCCL_TUNING_COUNT);
}

TEST_F(CostModelMicrotest, FamilyMasksPartitionTheIdSpace) {
  EXPECT_EQ(NCCL_TUNING_MASK_ALL,
            NCCL_TUNING_MASK_GENERAL_KERNELS | NCCL_TUNING_MASK_SYM_KERNELS | NCCL_TUNING_MASK_CE);

  EXPECT_EQ(0ul, NCCL_TUNING_MASK_GENERAL_KERNELS & NCCL_TUNING_MASK_SYM_KERNELS);
  EXPECT_EQ(0ul, NCCL_TUNING_MASK_GENERAL_KERNELS & NCCL_TUNING_MASK_CE);
  EXPECT_EQ(0ul, NCCL_TUNING_MASK_SYM_KERNELS & NCCL_TUNING_MASK_CE);

  EXPECT_EQ(NCCL_NUM_ALGORITHMS * NCCL_NUM_PROTOCOLS, __builtin_popcountll(NCCL_TUNING_MASK_GENERAL_KERNELS));
  EXPECT_EQ(ncclSymkKernelId_Count, __builtin_popcountll(NCCL_TUNING_MASK_SYM_KERNELS));
  EXPECT_EQ(ncclCeMethodId_Count, __builtin_popcountll(NCCL_TUNING_MASK_CE));

  // Callers pass these straight to ncclTuningCompute as tuningMask, so each
  // family's bits must sit where ncclTuningExpandId decodes that family.
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
  const int id = NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE;
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
  // time; without this the argmin would pick the candidate that opted out.
  EnableEverything();
  const int id = NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE;
  for (float t : {0.0f, -1.0f, static_cast<float>(NCCL_TUNING_IGNORE)}) {
    g_modelTimeUs = t;
    ncclTuningResult_t result = Sim(id);
    EXPECT_EQ(ModelKind::Ring, g_lastModel) << "time " << t;
    EXPECT_EQ(0, result.valid) << "time " << t;
    EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, result.timeUs) << "time " << t;
  }
}

TEST_F(CostModelMicrotest, ModelErrorInvalidatesTheCandidateAndPropagates) {
  EnableEverything();
  g_modelRet = ncclInternalError;
  ncclTuningResult_t result = Sim(NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE);
  EXPECT_EQ(ncclInternalError, lastRet_);
  EXPECT_EQ(0, result.valid);
  EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, result.timeUs);
}

// ---------------------------------------------------------------------------
// Init / finalize
// ---------------------------------------------------------------------------

TEST_F(CostModelMicrotest, InitSeedsEnabledColumnsFromModelMapAndRunsEachInitOnce) {
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));

  for (int id = 0; id < NCCL_TUNING_COUNT; id++) {
    EXPECT_EQ(modelMap[id].init != nullptr ? 1 : 0, g_initCalls[id]) << "tuning id " << id;
  }

  // enabled[][] starts as a copy of modelMap[].enabled and is only narrowed
  // afterwards, so it can never name a function the table did not.
  for (int id = 0; id < NCCL_TUNING_COUNT; id++) {
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
      if (modelMap[id].enabled[f] == 0) {
        EXPECT_EQ(0, comm_->tuningContext.enabled[id][f]) << "tuning id " << id << " func " << f;
      }
    }
  }
}

TEST_F(CostModelMicrotest, InitDisablesLl128ByDefaultButHonoursAnExplicitRequest) {
  // protoEnable == 2 means "defaulted"; only then may isLL128Enabled() veto.
  // PATH_SYS inter-node traffic is outside LL128's supported envelope.
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    comm_->graphs[a].typeInter = PATH_SYS;
    comm_->graphs[a].typeIntra = PATH_SYS;
  }
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));
  const int ll128 = NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_LL128;
  EXPECT_EQ(0, comm_->tuningContext.enabled[ll128][ncclFuncAllReduce]);

  SetMicroEnv("NCCL_PROTO", "LL128");
  comm_ = std::make_unique<ncclComm>();
  comm_->minCompCap = comm_->maxCompCap = 90;
  comm_->nRanks = 8;
  for (int a = 0; a < NCCL_NUM_ALGORITHMS; a++) {
    comm_->graphs[a].typeInter = PATH_SYS;
    comm_->graphs[a].typeIntra = PATH_SYS;
  }
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));
  EXPECT_EQ(1, comm_->tuningContext.enabled[ll128][ncclFuncAllReduce]);
}

TEST_F(CostModelMicrotest, AlgoEnvNarrowsSelectionToTheNamedAlgorithm) {
  SetMicroEnv("NCCL_ALGO", "ring");
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));

  EXPECT_EQ(1, comm_->tuningContext.forced[ncclFuncAllReduce]);
  const int ringSimple = NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE;
  const int treeSimple = NCCL_ALGO_TREE * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE;
  EXPECT_EQ(1, comm_->tuningContext.enabled[ringSimple][ncclFuncAllReduce]);
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
  EXPECT_EQ(0, comm_->tuningContext.enabled[NCCL_ALGO_RING * NCCL_NUM_PROTOCOLS + NCCL_PROTO_SIMPLE]
                                           [ncclFuncAllReduce]);
}

TEST_F(CostModelMicrotest, FinalizeVisitsEveryIdWithoutTouchingTheIdSpaceBounds) {
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));
  // No modelMap row carries a finalize today; the loop must still walk the whole
  // space and succeed, so adding one cannot silently be skipped.
  for (int id = 0; id < NCCL_TUNING_COUNT; id++) {
    EXPECT_EQ(nullptr, modelMap[id].finalize) << "tuning id " << id << " gained a finalize; extend this test";
  }
  EXPECT_EQ(ncclSuccess, ncclTuningCostModelFinalize(comm_.get()));
}

}  // namespace
