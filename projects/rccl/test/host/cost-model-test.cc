/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/tuning/cost_model.cc: the unified cost model's
// model registry and dispatch (AICOMRCCL-2007).
//
// The unit is modelMap[], which flattens three kernel families into one id space:
//   [0, 21)                  general algo*NCCL_NUM_PROTOCOLS + proto
//   [21, 39)                 ncclSymkKernelId_*   (device-API symmetric kernels)
//   [39, NCCL_TUNING_COUNT)  ncclCeMethodId_*     (copy engine)
// An off-by-one in that layout mis-prices a whole family rather than failing to
// build, so these tests pin the layout and the dispatch to it.
//
// Neighbours are out of scope: ncclTuningExpandId's decode/reject is
// tuning-general-test.cc, ncclTuningCompute's selection is AICOMRCCL-2448.

#include <gtest/gtest.h>

#include <cstring>
#include <limits>
#include <memory>

#include "comm.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "sym_kernels.h"

// src/init.cc defines this; nothing on the micro link line does, and parseList() needs it as
// its prefix table. RcclAssertSourceLine pins the first line, the static_assert the count.
const char* ncclFuncStr[NCCL_NUM_FUNCTIONS + 4] = {"Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce",
                                                   "AlltoAllPivot", "AlltoAllGda", "AlltoAllvGda",
                                                   "SendRecv"};
static_assert(NCCL_NUM_FUNCTIONS == 5, "ncclFuncStr mirrors src/init.cc; add the new function name");

// src/sym_kernels.cc defines this; the generated sym_kernels_host.cc carries only
// ncclSymkGetKernelIndex and the kernel-list tables. Mirrored rather than pulled from
// sym_kernels_index_fakes.cc, which would collide with that generated file.
const char* ncclSymkKernelIdToString(int kernelId) {
  if (kernelId < 0 || kernelId >= ncclSymkKernelId_Count) return "Unknown";
  return ncclSymKernelStr[kernelId];
}

namespace {

// Model-function seams. modelMap[] points at per-family functions whose real TUs
// (ring.cc, tree.cc, ...) are off this link line, so defining them here both links
// the TU and makes dispatch observable.

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
#define DEFINE_MODEL_FAKE(Name) \
  ncclResult_t ncclTuning##Name##ModelInit(struct ncclComm*, int id, int[NCCL_NUM_FUNCTIONS]) { \
    return RecordInit(id); \
  } \
  ncclResult_t ncclTuning##Name##ModelSim(struct ncclTuningInput_t* const, \
                                          struct ncclTuningResult_t* const tuning) { \
    return RecordSim(ModelKind::Name, tuning); \
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
// After cost_model.cc: algorithm_registry.cc defines unguarded ALGBIT/F_* macros.
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

  // A matched compcap pair above Hopper keeps isLL128Enabled() from vetoing for an
  // unrelated reason.
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
    // Production's only caller decodes into result before dispatching (tuning.cc) and never
    // passes an out-of-range id, so ignore that reject.
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

  // The family an id must route to, derived from the id space, not from modelMap[].
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

// getModelEntry() bounds-checks against NCCL_TUNING_COUNT then indexes modelMap[], so a short
// table reads out of bounds. static, not EXPECT: it must stop the build, not 17 other cases.
static_assert(sizeof(modelMap) / sizeof(modelMap[0]) == static_cast<size_t>(NCCL_TUNING_COUNT),
              "modelMap[] must carry exactly one row per tuning id");

// The offsets and the masks derived from them are macro algebra over the same two counts
// (tuning.h), so these are build-time facts.
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
  // Each family's bits must sit where ncclTuningExpandId decodes that family. This is the half
  // that can go red: tuning_general.cc respells both boundaries instead of reusing the macros.
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

// modelMap[] <-> algRegistry[] correspondence. Each file's comment demands the other stay in
// sync and nothing enforces it. The registry is the user-facing half (algSelection resolves
// names through it), so a drift means a name that cannot be priced, or a price with no name.

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
  // algorithm_registry.cc: "collMask mirrors modelMap[].enabled[func]".
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
  // The registry stops before the CE ids, and cost_model.cc's "TODO: allow NCCL_ALGO=CE" is the
  // other half of that gap. Pinned so adding CE rows forces this test and the TODO together.
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
  // Prefix match: "RING" spans the three RING_* rows without reaching SYMK_RailRing_LsaSTMC.
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
  // The ten {nullptr,...} rows are combinations that do not exist. The selector is an argmin
  // over timeUs, so a default 0 would make an unbuildable kernel beat every real one.
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
  // A model opts out by returning a non-positive time. The boundary is the SIGN, not the
  // sentinel: NCCL_TUNING_IGNORE is -1.0, so `timeUs <= 0.0` cannot tell the two apart.
  EnableEverything();
  for (float t : {0.0f, -1.0f}) {
    g_modelTimeUs = t;
    ncclTuningResult_t result = Sim(kRingSimple);
    EXPECT_EQ(ModelKind::Ring, g_lastModel) << "time " << t;
    EXPECT_EQ(0, result.valid) << "time " << t;
    EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, result.timeUs) << "time " << t;
  }

  // The accept side of the same boundary: the float adjacent to zero still counts.
  g_modelTimeUs = std::numeric_limits<float>::denorm_min();
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
  // g_initCalls drops an out-of-range id, so the total is the only place one would show up.
  EXPECT_EQ(expectedInits, g_initCallTotal);

  // enabled[][] starts as a copy of modelMap[].enabled, so with no env set and LL128 eligible
  // it must match BOTH ways; checking only the zeros would miss a dropped candidate.
  for (int id = 0; id < NCCL_TUNING_COUNT; id++) {
    for (int f = 0; f < NCCL_NUM_FUNCTIONS; f++) {
      EXPECT_EQ(modelMap[id].enabled[f] != 0, comm_->tuningContext.enabled[id][f] != 0)
          << "tuning id " << id << " func " << f;
    }
  }
}

TEST_F(CostModelMicrotest, InitDisablesLl128ByDefaultButHonoursAnExplicitRequest) {
  // protoEnable == 2 means "defaulted"; only then may isLL128Enabled() veto. With LL128_C2C
  // off that veto caps inter-node reach at PXB, so a PXN graph falls outside it; intra stays
  // at PATH_NVB so the C2C branch is what decides.
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

  // Naming an algorithm zeroes the symmetric-kernel enables too.
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
  // No modelMap row carries a finalize, so the function is unobservable today; pin that so
  // adding one forces this test to grow a fake. The return is deliberately not asserted:
  // cost_model.cc's `exit:` returns the literal ncclSuccess and `fail:` falls into it.
  for (int id = 0; id < NCCL_TUNING_COUNT; id++) {
    EXPECT_EQ(nullptr, modelMap[id].finalize) << "tuning id " << id << " gained a finalize; extend this test";
  }
  ASSERT_EQ(ncclSuccess, ncclTuningCostModelInit(comm_.get()));
  (void)ncclTuningCostModelFinalize(comm_.get());
}

}  // namespace
