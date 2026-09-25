/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/tuning/tuning.cc.

#include <gtest/gtest.h>

#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "alloc.h"
#include "comm.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/param_redirect.h"
#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "transport.h"
#include "tuner.h"
#include "tuning.h"

namespace {

using CommHook = std::function<ncclResult_t(struct ncclComm*)>;
using ExpandIdHook = std::function<ncclResult_t(int, int*, int*, int*, int*)>;
using ComputeTuningHook =
    std::function<ncclResult_t(int, struct ncclTuningInput_t*, struct ncclTuningResult_t*)>;
using GetChannelsHook =
    std::function<ncclResult_t(struct ncclTuningInput_t*, struct ncclTuningResult_t*)>;
using NvlsQueryHook = std::function<ncclResult_t(struct ncclComm*, ncclFunc_t, int*)>;
using ParamHook = std::function<int64_t(const char*, int64_t)>;

CommHook g_tuningTunerPluginLoad;
CommHook g_tuningTunerPluginUnload;
CommHook g_tuningCostModelInit;
CommHook g_tuningCostModelFinalize;
CommHook g_tuningSetThreadThresholds;
ExpandIdHook g_tuningExpandId;
ComputeTuningHook g_tuningCostModelSimModel;
GetChannelsHook g_tuningGetChannels;
NvlsQueryHook g_tuningNvlsRegResourcesQuery;
std::function<int()> g_tuningSymkLLKernelMask;

using PluginInitHook = std::function<ncclResult_t(void**, uint64_t, size_t, size_t, ncclDebugLogger_t,
                                                  ncclNvlDomainInfo_v6_t*, ncclTunerConstants_v6_t*)>;
using PluginGetCollInfoHook =
    std::function<ncclResult_t(void*, ncclFunc_t, size_t, int, float**, int, int, int, int*)>;
using PluginFinalizeHook = std::function<ncclResult_t(void*)>;

PluginInitHook g_tuningPluginInit;
PluginGetCollInfoHook g_tuningPluginGetCollInfo;
PluginFinalizeHook g_tuningPluginFinalize;

int g_tuningCallocCalls = 0;
int g_tuningCallocFailAt = -1;
int g_tuningFreeCalls = 0;
std::vector<size_t> g_tuningCallocElements;

ncclResult_t DefaultExpandId(int id, int* algo, int* proto, int* symKernelId, int* ceMethodId) {
  if (id < 0 || id >= NCCL_TUNING_COUNT) return ncclInvalidUsage;
  if (algo) *algo = NCCL_ALGO_UNDEF;
  if (proto) *proto = NCCL_PROTO_UNDEF;
  if (symKernelId) *symKernelId = ncclSymkKernelId_Count;
  if (ceMethodId) *ceMethodId = ncclCeMethodId_Count;
  if (id < NCCL_TUNING_SYM_KERNEL_ID_OFFSET) {
    if (algo) *algo = id / NCCL_NUM_PROTOCOLS;
    if (proto) *proto = id % NCCL_NUM_PROTOCOLS;
  } else if (id < NCCL_TUNING_CE_METHOD_ID_OFFSET) {
    if (symKernelId) *symKernelId = id - NCCL_TUNING_SYM_KERNEL_ID_OFFSET;
  } else {
    if (ceMethodId) *ceMethodId = id - NCCL_TUNING_CE_METHOD_ID_OFFSET;
  }
  return ncclSuccess;
}

void ResetTuningHooks() {
  g_tuningTunerPluginLoad = [](struct ncclComm*) { return ncclSuccess; };
  g_tuningTunerPluginUnload = [](struct ncclComm*) { return ncclSuccess; };
  g_tuningCostModelInit = [](struct ncclComm*) { return ncclSuccess; };
  g_tuningCostModelFinalize = [](struct ncclComm*) { return ncclSuccess; };
  g_tuningSetThreadThresholds = [](struct ncclComm*) { return ncclSuccess; };
  g_tuningExpandId = DefaultExpandId;
  g_tuningCostModelSimModel = [](int, struct ncclTuningInput_t*, struct ncclTuningResult_t* result) {
    result->valid = 0;
    return ncclSuccess;
  };
  g_tuningGetChannels = [](struct ncclTuningInput_t*, struct ncclTuningResult_t*) { return ncclSuccess; };
  g_tuningNvlsRegResourcesQuery = [](struct ncclComm*, ncclFunc_t, int*) { return ncclSystemError; };
  g_tuningSymkLLKernelMask = [] { return 0; };
  g_tuningPluginInit = [](void**, uint64_t, size_t, size_t, ncclDebugLogger_t, ncclNvlDomainInfo_v6_t*,
                          ncclTunerConstants_v6_t*) { return ncclSuccess; };
  g_tuningPluginGetCollInfo = [](void*, ncclFunc_t, size_t, int, float**, int, int, int, int*) {
    return ncclSuccess;
  };
  g_tuningPluginFinalize = [](void*) { return ncclSuccess; };
  g_tuningCallocCalls = 0;
  g_tuningCallocFailAt = -1;
  g_tuningFreeCalls = 0;
  g_tuningCallocElements.clear();
}

}  // namespace

ncclResult_t TuningTunerPluginLoad(struct ncclComm* comm) { return g_tuningTunerPluginLoad(comm); }
ncclResult_t TuningTunerPluginUnload(struct ncclComm* comm) { return g_tuningTunerPluginUnload(comm); }
ncclResult_t TuningCostModelInit(struct ncclComm* comm) { return g_tuningCostModelInit(comm); }
ncclResult_t TuningCostModelFinalize(struct ncclComm* comm) { return g_tuningCostModelFinalize(comm); }
ncclResult_t TuningSetThreadThresholds(struct ncclComm* comm) { return g_tuningSetThreadThresholds(comm); }
ncclResult_t TuningExpandId(int id, int* algo, int* proto, int* symKernelId, int* ceMethodId) {
  return g_tuningExpandId(id, algo, proto, symKernelId, ceMethodId);
}
ncclResult_t TuningCostModelSimModel(int id, struct ncclTuningInput_t* input,
                                     struct ncclTuningResult_t* result) {
  return g_tuningCostModelSimModel(id, input, result);
}
ncclResult_t TuningGetChannels(struct ncclTuningInput_t* input, struct ncclTuningResult_t* result) {
  return g_tuningGetChannels(input, result);
}
ncclResult_t TuningNvlsRegResourcesQuery(struct ncclComm* comm, ncclFunc_t func, int* recChannels) {
  return g_tuningNvlsRegResourcesQuery(comm, func, recChannels);
}
int TuningSymkLLKernelMask() { return g_tuningSymkLLKernelMask(); }
const char* TuningSymkKernelIdToString(int) { return "micro-sym-kernel"; }

const char* TuningFuncStr[NCCL_NUM_FUNCTIONS + 4] = {
    "Broadcast", "Reduce", "AllGather", "ReduceScatter", "AllReduce", "AlltoAllPivot", "AlltoAllGda",
    "AlltoAllvGda", "SendRecv"};
static_assert(NCCL_NUM_FUNCTIONS == 5, "TuningFuncStr mirrors src/init.cc; add the new function name");

ncclResult_t TuningPluginInit(void** context, uint64_t commId, size_t nRanks, size_t nNodes,
                              ncclDebugLogger_t logger, ncclNvlDomainInfo_v6_t* nvlDomainInfo,
                              ncclTunerConstants_v6_t* constants) {
  return g_tuningPluginInit(context, commId, nRanks, nNodes, logger, nvlDomainInfo, constants);
}
ncclResult_t TuningPluginGetCollInfo(void* context, ncclFunc_t func, size_t nBytes, int numPipeOps,
                                     float** table, int numAlgo, int numProto, int regBuff, int* nChannels) {
  return g_tuningPluginGetCollInfo(context, func, nBytes, numPipeOps, table, numAlgo, numProto, regBuff, nChannels);
}
ncclResult_t TuningPluginFinalize(void* context) { return g_tuningPluginFinalize(context); }

template <typename T>
ncclResult_t TuningCalloc(const char* file, int line, const char* function, T** pointer, size_t elements) {
  const int call = g_tuningCallocCalls++;
  g_tuningCallocElements.push_back(elements);
  if (call == g_tuningCallocFailAt) return ncclSystemError;
  return ncclCallocDebug(pointer, elements, file, line, function, true);
}

void TuningFree(void* pointer) {
  if (pointer != nullptr) ++g_tuningFreeCalls;
  std::free(pointer);
}

// Redirect tuning.cc dependencies to the host trampolines above. Keep all declaring headers before this block.
#undef ncclCalloc
#define ncclCalloc(...) TuningCalloc(__FILE__, __LINE__, __func__, __VA_ARGS__)
#define free(pointer) TuningFree(pointer)

#define ncclTunerPluginLoad TuningTunerPluginLoad
#define ncclTunerPluginUnload TuningTunerPluginUnload
#define ncclTuningCostModelInit TuningCostModelInit
#define ncclTuningCostModelFinalize TuningCostModelFinalize
#define ncclTuningCostModelSimModel TuningCostModelSimModel
#define ncclTuningSetThreadThresholds TuningSetThreadThresholds
#define ncclTuningExpandId TuningExpandId
#define ncclTuningGetChannels TuningGetChannels
#define ncclNvlsRegResourcesQuery TuningNvlsRegResourcesQuery
#define ncclSymkLLKernelMask TuningSymkLLKernelMask
#define ncclSymkKernelIdToString TuningSymkKernelIdToString
#define ncclFuncStr TuningFuncStr
#define ncclTuningInit TuningMicroInit
#define ncclTuningFinalize TuningMicroFinalize
#define ncclTuningCompute TuningMicroCompute

#include TUNING_CC_PATH

#undef ncclTuningCompute
#undef ncclTuningFinalize
#undef ncclTuningInit
#undef ncclFuncStr
#undef ncclSymkKernelIdToString
#undef ncclSymkLLKernelMask
#undef ncclNvlsRegResourcesQuery
#undef ncclTuningGetChannels
#undef ncclTuningExpandId
#undef ncclTuningSetThreadThresholds
#undef ncclTuningCostModelSimModel
#undef ncclTuningCostModelFinalize
#undef ncclTuningCostModelInit
#undef ncclTunerPluginUnload
#undef ncclTunerPluginLoad
#undef free
#undef ncclCalloc

namespace {

constexpr int GeneralId(int algo, int proto) { return algo * NCCL_NUM_PROTOCOLS + proto; }
constexpr uint64_t TuningBit(int id) { return uint64_t{1} << id; }
constexpr int symFirst = NCCL_TUNING_SYM_KERNEL_ID_OFFSET;
constexpr int symSecond = NCCL_TUNING_SYM_KERNEL_ID_OFFSET + 1;

ComputeTuningHook ModelSelecting(int selectedId, float timeUs = 1.0f) {
  return [=](int id, ncclTuningInput_t*, ncclTuningResult_t* tuning) {
    tuning->valid = id == selectedId;
    tuning->timeUs = timeUs;
    return ncclSuccess;
  };
}

GetChannelsHook ChannelsReturning(int nChannels = 0, int nWarps = 0) {
  return [=](ncclTuningInput_t*, ncclTuningResult_t* tuning) {
    tuning->nChannels = nChannels;
    tuning->nWarps = nWarps;
    return ncclSuccess;
  };
}

ParamHook TuningParams(std::vector<std::pair<std::string, int64_t>> overrides) {
  return [overrides = std::move(overrides)](const char* name, int64_t defaultValue) {
    for (const auto& [param, value] : overrides) {
      if (param == name) return value;
    }
    return defaultValue;
  };
}

ncclTuningResult_t EmptyResult() {
  ncclTuningResult_t result = NCCL_TUNING_RESULT_INIT;
  return result;
}

class TuningMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ResetNcclFakes();
    ResetEnvFakes();
    ResetTuningHooks();
    comm_ = std::make_unique<ncclComm>();
    comm_->rank = 1;
    comm_->nRanks = 2;
    comm_->nNodes = 1;
    comm_->intraRanks = 1;
    for (int algo = 0; algo < NCCL_NUM_ALGORITHMS; ++algo) {
      for (int proto = 0; proto < NCCL_NUM_PROTOCOLS; ++proto) {
        comm_->tuningContext.maxThreads[algo][proto] = 256;
      }
    }
    plugin_ = {};
    plugin_.name = "micro-tuner";
    plugin_.init = TuningPluginInit;
    plugin_.getCollInfo = TuningPluginGetCollInfo;
    plugin_.finalize = TuningPluginFinalize;
  }

  void TearDown() override {
    ResetTuningHooks();
    ResetEnvFakes();
    ResetNcclFakes();
  }

  ncclTuningInput_t MakeInput() const {
    ncclTuningInput_t input{};
    input.comm = comm_.get();
    input.func = ncclFuncAllReduce;
    input.datatype = ncclFloat32;
    input.nBytes = 4096;
    input.numPipeOps = 3;
    input.count = 1024;
    input.countMax = 2048;
    input.nWorks = 1;
    input.winRegType = ncclSymSendNonregRecvNonreg;
    return input;
  }

  ncclTuningInput_t MakeNvlsEligibleInput(ncclFunc_t func) const {
    ncclTuningInput_t input = MakeInput();
    input.func = func;
    input.CTAPolicy = NCCL_CTA_POLICY_EFFICIENCY;
    input.regBuff = 1;
    input.nvlsSupport = 1;
    input.tuningMask = TuningBit(GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL)) |
                       TuningBit(GeneralId(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE));
    return input;
  }

  std::unique_ptr<ncclComm> comm_;
  ncclTuner_t plugin_{};
};

TEST_F(TuningMicrotest, ResultListPushesAtFrontAndFreesEveryNode) {
  ncclTuningResultList_t list{};
  ncclTuningResult_t first = EmptyResult();
  first.id = 3;
  first.timeUs = 7.0f;
  first.algo = NCCL_ALGO_TREE;
  first.nChannels = 11;
  ncclTuningResult_t second = EmptyResult();
  second.id = 9;
  second.timeUs = 2.0f;
  second.proto = NCCL_PROTO_SIMPLE;
  second.forced = 1;

  ASSERT_EQ(ncclSuccess, ncclTuningResultListPushFront(&list, first));
  ASSERT_EQ(ncclSuccess, ncclTuningResultListPushFront(&list, second));
  ASSERT_NE(nullptr, list.head);
  EXPECT_EQ(9, list.head->result.id);
  EXPECT_EQ(NCCL_PROTO_SIMPLE, list.head->result.proto);
  EXPECT_EQ(1, list.head->result.forced);
  ASSERT_NE(nullptr, list.head->next);
  EXPECT_EQ(3, list.head->next->result.id);
  EXPECT_EQ(NCCL_ALGO_TREE, list.head->next->result.algo);
  EXPECT_EQ(11, list.head->next->result.nChannels);
  EXPECT_EQ(nullptr, list.head->next->next);
  EXPECT_EQ((std::vector<size_t>{1, 1}), g_tuningCallocElements);

  ncclTuningResultListFree(&list);
  EXPECT_EQ(nullptr, list.head);
  EXPECT_EQ(2, g_tuningFreeCalls);
  ncclTuningResultListFree(&list);
  EXPECT_EQ(2, g_tuningFreeCalls);
}

TEST_F(TuningMicrotest, ResultListAllocationFailureLeavesListUntouched) {
  ncclTuningResultList_t list{};
  g_tuningCallocFailAt = 0;

  EXPECT_EQ(ncclSystemError, ncclTuningResultListPushFront(&list, EmptyResult()));
  EXPECT_EQ(1, g_tuningCallocCalls);
  EXPECT_EQ((std::vector<size_t>{1}), g_tuningCallocElements);
  EXPECT_EQ(nullptr, list.head);
}

TEST_F(TuningMicrotest, InitWithoutPluginRunsCostModelThenThresholdSetup) {
  std::vector<std::string> order;
  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_TUNING);
  ScopedHook load(g_tuningTunerPluginLoad, [&](ncclComm* comm) {
    EXPECT_EQ(comm_.get(), comm);
    order.push_back("load");
    return ncclSuccess;
  });
  ScopedHook cost(g_tuningCostModelInit, [&](ncclComm* comm) {
    EXPECT_EQ(comm_.get(), comm);
    order.push_back("cost");
    return ncclSuccess;
  });
  ScopedHook thresholds(g_tuningSetThreadThresholds, [&](ncclComm* comm) {
    EXPECT_EQ(comm_.get(), comm);
    order.push_back("thresholds");
    return ncclSuccess;
  });

  ncclResult_t result = ncclInternalError;
  const std::string log = RcclUnitTesting::CaptureLog([&] { result = TuningMicroInit(comm_.get()); });
  EXPECT_EQ(ncclSuccess, result);
  EXPECT_EQ((std::vector<std::string>{"load", "cost", "thresholds"}), order);
  EXPECT_EQ(1, load.calls);
  EXPECT_EQ(1, cost.calls);
  EXPECT_EQ(1, thresholds.calls);
  EXPECT_FALSE(RcclUnitTesting::LogHas(log, "Algorithm   |"));
}

TEST_F(TuningMicrotest, InitPassesPluginInputsBeforeCostModelAndPrintsRankZeroTables) {
  std::vector<std::string> order;
  int contextSentinel = 0;
  comm_->rank = 0;
  comm_->commHash = 0x1122334455667788ull;
  comm_->nRanks = 16;
  comm_->nNodes = 4;
  comm_->tuningContext.generalLatencies[ncclFuncAllReduce][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 12.5f;
  comm_->tuningContext.generalBandwidths[ncclFuncAllReduce][NCCL_ALGO_PAT][NCCL_PROTO_SIMPLE] = 34.5f;
  RcclUnitTesting::ScopedDebugLogging debugLogging(NCCL_LOG_INFO, NCCL_TUNING);

  ScopedHook load(g_tuningTunerPluginLoad, [&](ncclComm* comm) {
    order.push_back("load");
    comm->tuner = &plugin_;
    return ncclSuccess;
  });
  ScopedHook pluginInit(
      g_tuningPluginInit,
      [&](void** context, uint64_t commId, size_t nRanks, size_t nNodes, ncclDebugLogger_t logger,
          ncclNvlDomainInfo_v6_t* nvlDomainInfo, ncclTunerConstants_v6_t* constants) {
        order.push_back("plugin-init");
        EXPECT_EQ(comm_->commHash, commId);
        EXPECT_EQ(16u, nRanks);
        EXPECT_EQ(4u, nNodes);
        EXPECT_NE(nullptr, logger);
        EXPECT_EQ(&comm_->nvlDomainInfo, nvlDomainInfo);
        EXPECT_EQ(&comm_->tuningContext.tuningConstants, constants);
        *context = &contextSentinel;
        return ncclSuccess;
      });
  ScopedHook cost(g_tuningCostModelInit, [&](ncclComm*) {
    order.push_back("cost");
    return ncclSuccess;
  });
  ScopedHook thresholds(g_tuningSetThreadThresholds, [&](ncclComm*) {
    order.push_back("thresholds");
    return ncclSuccess;
  });

  ncclResult_t result = ncclInternalError;
  const std::string log = RcclUnitTesting::CaptureLog([&] { result = TuningMicroInit(comm_.get()); });
  EXPECT_EQ(ncclSuccess, result);
  EXPECT_EQ(&contextSentinel, comm_->tunerContext);
  EXPECT_EQ((std::vector<std::string>{"load", "plugin-init", "cost", "thresholds"}), order);
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "Algorithm   |")) << log;
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "PAT")) << log;
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "Simple")) << log;
  EXPECT_TRUE(RcclUnitTesting::LogHas(log, "12.5/  34.5")) << log;
}

TEST_F(TuningMicrotest, InitStopsAtTheFailingStage) {
  {
    ScopedHook load(g_tuningTunerPluginLoad, [](ncclComm*) { return ncclSystemError; });
    ScopedHook cost(g_tuningCostModelInit, [](ncclComm*) { return ncclSuccess; });
    EXPECT_EQ(ncclSystemError, TuningMicroInit(comm_.get()));
    EXPECT_EQ(1, load.calls);
    EXPECT_EQ(0, cost.calls);
  }
  {
    ScopedHook load(g_tuningTunerPluginLoad, [&](ncclComm* comm) {
      comm->tuner = &plugin_;
      return ncclSuccess;
    });
    ScopedHook pluginInit(g_tuningPluginInit,
                          [](void**, uint64_t, size_t, size_t, ncclDebugLogger_t, ncclNvlDomainInfo_v6_t*,
                             ncclTunerConstants_v6_t*) { return ncclInvalidUsage; });
    ScopedHook cost(g_tuningCostModelInit, [](ncclComm*) { return ncclSuccess; });
    EXPECT_EQ(ncclInvalidUsage, TuningMicroInit(comm_.get()));
    EXPECT_EQ(1, pluginInit.calls);
    EXPECT_EQ(0, cost.calls);
    comm_->tuner = nullptr;
  }
  {
    ScopedHook cost(g_tuningCostModelInit, [](ncclComm*) { return ncclInternalError; });
    ScopedHook thresholds(g_tuningSetThreadThresholds, [](ncclComm*) { return ncclSuccess; });
    EXPECT_EQ(ncclInternalError, TuningMicroInit(comm_.get()));
    EXPECT_EQ(1, cost.calls);
    EXPECT_EQ(0, thresholds.calls);
  }
}

TEST_F(TuningMicrotest, FinalizeOrdersCostPluginAndUnloadAndStopsOnErrors) {
  std::vector<std::string> order;
  int contextSentinel = 0;
  comm_->tunerContext = &contextSentinel;

  {
    ScopedHook cost(g_tuningCostModelFinalize, [&](ncclComm*) {
      order.push_back("cost");
      return ncclSuccess;
    });
    EXPECT_EQ(ncclSuccess, TuningMicroFinalize(comm_.get()));
    EXPECT_EQ((std::vector<std::string>{"cost"}), order);
  }

  order.clear();
  comm_->tuner = &plugin_;
  {
    ScopedHook cost(g_tuningCostModelFinalize, [&](ncclComm*) {
      order.push_back("cost");
      return ncclSuccess;
    });
    ScopedHook pluginFinalize(g_tuningPluginFinalize, [&](void* context) {
      EXPECT_EQ(&contextSentinel, context);
      order.push_back("plugin-finalize");
      return ncclSuccess;
    });
    ScopedHook unload(g_tuningTunerPluginUnload, [&](ncclComm* comm) {
      EXPECT_EQ(comm_.get(), comm);
      order.push_back("unload");
      return ncclSuccess;
    });
    EXPECT_EQ(ncclSuccess, TuningMicroFinalize(comm_.get()));
    EXPECT_EQ((std::vector<std::string>{"cost", "plugin-finalize", "unload"}), order);
  }

  {
    ScopedHook cost(g_tuningCostModelFinalize, [](ncclComm*) { return ncclSystemError; });
    ScopedHook pluginFinalize(g_tuningPluginFinalize, [](void*) { return ncclSuccess; });
    EXPECT_EQ(ncclSystemError, TuningMicroFinalize(comm_.get()));
    EXPECT_EQ(0, pluginFinalize.calls);
  }
  {
    ScopedHook pluginFinalize(g_tuningPluginFinalize, [](void*) { return ncclInvalidUsage; });
    ScopedHook unload(g_tuningTunerPluginUnload, [](ncclComm*) { return ncclSuccess; });
    EXPECT_EQ(ncclInvalidUsage, TuningMicroFinalize(comm_.get()));
    EXPECT_EQ(0, unload.calls);
  }
  {
    ScopedHook unload(g_tuningTunerPluginUnload, [](ncclComm*) { return ncclInternalError; });
    EXPECT_EQ(ncclInternalError, TuningMicroFinalize(comm_.get()));
    EXPECT_EQ(1, unload.calls);
  }
}

TEST_F(TuningMicrotest, ComputeAllTuningsVisitsOnlyMaskedIdsAndKeepsOnlyValidResults) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  const int ringSimple = GeneralId(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(treeLl) | TuningBit(ringSimple) | TuningBit(symFirst);
  ncclTuningResultList_t list{};
  std::vector<int> expanded;
  std::vector<int> simulated;

  ScopedHook expand(g_tuningExpandId, [&](int id, int* algo, int* proto, int* symKernelId, int* ceMethodId) {
    expanded.push_back(id);
    return DefaultExpandId(id, algo, proto, symKernelId, ceMethodId);
  });
  ScopedHook model(g_tuningCostModelSimModel, [&](int id, ncclTuningInput_t*, ncclTuningResult_t* result) {
    simulated.push_back(id);
    if (id == ringSimple) result->valid = 0;
    result->timeUs = static_cast<float>(100 - id);
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, ncclTuningComputeAllTunings(&input, &list));
  EXPECT_EQ((std::vector<int>{treeLl, ringSimple, symFirst}), expanded);
  EXPECT_EQ(expanded, simulated);
  ASSERT_NE(nullptr, list.head);
  EXPECT_EQ(symFirst, list.head->result.id);
  ASSERT_NE(nullptr, list.head->next);
  EXPECT_EQ(treeLl, list.head->next->result.id);
  EXPECT_EQ(nullptr, list.head->next->next);
  ncclTuningResultListFree(&list);
}

TEST_F(TuningMicrotest, ComputeAllTuningsPropagatesDecodeModelAndAllocationFailures) {
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(0);
  ncclTuningResultList_t list{};

  {
    ScopedHook expand(g_tuningExpandId,
                      [](int, int*, int*, int*, int*) { return ncclInvalidUsage; });
    ScopedHook model(g_tuningCostModelSimModel,
                     [](int, ncclTuningInput_t*, ncclTuningResult_t*) { return ncclSuccess; });
    EXPECT_EQ(ncclInvalidUsage, ncclTuningComputeAllTunings(&input, &list));
    EXPECT_EQ(1, expand.calls);
    EXPECT_EQ(0, model.calls);
  }
  {
    ScopedHook model(g_tuningCostModelSimModel,
                     [](int, ncclTuningInput_t*, ncclTuningResult_t*) { return ncclRemoteError; });
    EXPECT_EQ(ncclRemoteError, ncclTuningComputeAllTunings(&input, &list));
    EXPECT_EQ(1, model.calls);
  }
  {
    ScopedHook model(g_tuningCostModelSimModel, [](int, ncclTuningInput_t*, ncclTuningResult_t* result) {
      result->valid = 1;
      return ncclSuccess;
    });
    g_tuningCallocCalls = 0;
    g_tuningCallocFailAt = 0;
    EXPECT_EQ(ncclSystemError, ncclTuningComputeAllTunings(&input, &list));
    EXPECT_EQ(1, g_tuningCallocCalls);
    EXPECT_EQ(nullptr, list.head);
  }

  ncclTuningResult_t result = EmptyResult();
  ScopedHook model(g_tuningCostModelSimModel,
                   [](int id, ncclTuningInput_t*, ncclTuningResult_t* result) {
                     EXPECT_EQ(17, id);
                     result->timeUs = 12.5f;
                     return ncclInvalidArgument;
                   });
  EXPECT_EQ(ncclInvalidArgument, ncclTuningComputeTuning(17, &input, &result));
  EXPECT_FLOAT_EQ(12.5f, result.timeUs);
}

TEST_F(TuningMicrotest, SelectBestHandlesEmptyMinimumAndStableTies) {
  ncclTuningResultList_t empty{};
  ncclTuningResult_t best = EmptyResult();
  ASSERT_EQ(ncclSuccess, ncclTuningSelectBestTuning(&empty, &best));
  EXPECT_EQ(FLT_MAX, best.timeUs);
  EXPECT_EQ(NCCL_ALGO_UNDEF, best.algo);

  ncclTuningResultListNode second{};
  second.result = EmptyResult();
  second.result.id = 2;
  second.result.timeUs = 5.0f;
  ncclTuningResultListNode first{};
  first.next = &second;
  first.result = EmptyResult();
  first.result.id = 1;
  first.result.timeUs = 5.0f;
  ncclTuningResultList_t list{&first};

  ASSERT_EQ(ncclSuccess, ncclTuningSelectBestTuning(&list, &best));
  EXPECT_EQ(1, best.id);
  second.result.timeUs = 4.0f;
  ASSERT_EQ(ncclSuccess, ncclTuningSelectBestTuning(&list, &best));
  EXPECT_EQ(2, best.id);
}

TEST_F(TuningMicrotest, ComputeSingleRankUsesRingSimpleDefaultsBeforeChannelSelection) {
  comm_->nRanks = 1;
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = NCCL_TUNING_MASK_ALL;
  ncclTuningResult_t result = EmptyResult();
  result.forced = 77;
  ScopedHook channels(g_tuningGetChannels, [](ncclTuningInput_t*, ncclTuningResult_t* tuning) {
    EXPECT_EQ(NCCL_ALGO_RING, tuning->algo);
    EXPECT_EQ(NCCL_PROTO_SIMPLE, tuning->proto);
    EXPECT_EQ(ncclSymkKernelId_Count, tuning->symKernelId);
    EXPECT_EQ(ncclCeMethodId_Count, tuning->ceMethodId);
    EXPECT_EQ(0, tuning->nChannels);
    EXPECT_EQ(0, tuning->maxChannels);
    EXPECT_EQ(0, tuning->nWarps);
    EXPECT_EQ(0, tuning->forced);
    tuning->nChannels = 3;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(1, channels.calls);
  EXPECT_EQ(NCCL_ALGO_RING, result.algo);
  EXPECT_EQ(NCCL_PROTO_SIMPLE, result.proto);
  EXPECT_EQ(3, result.nChannels);
  EXPECT_EQ(0, result.forced);
}

TEST_F(TuningMicrotest, ChannelSelectionFailureLeavesCallerResultUntouched) {
  comm_->nRanks = 1;
  ncclTuningInput_t input = MakeInput();
  ncclTuningResult_t result = EmptyResult();
  result.id = 123;
  result.algo = NCCL_ALGO_PAT;
  ScopedHook channels(g_tuningGetChannels,
                      [](ncclTuningInput_t*, ncclTuningResult_t*) { return ncclRemoteError; });

  EXPECT_EQ(ncclRemoteError, TuningMicroCompute(&input, &result));
  EXPECT_EQ(1, channels.calls);
  EXPECT_EQ(123, result.id);
  EXPECT_EQ(NCCL_ALGO_PAT, result.algo);
}

TEST_F(TuningMicrotest, ComputeAllTuningsFailureLeavesCallerResultUntouched) {
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(0);
  ncclTuningResult_t result = EmptyResult();
  result.id = 123;
  result.algo = NCCL_ALGO_PAT;
  ScopedHook model(g_tuningCostModelSimModel,
                   [](int, ncclTuningInput_t*, ncclTuningResult_t*) { return ncclRemoteError; });
  ScopedHook channels(g_tuningGetChannels, ChannelsReturning());

  EXPECT_EQ(ncclRemoteError, TuningMicroCompute(&input, &result));
  EXPECT_EQ(1, model.calls);
  EXPECT_EQ(0, channels.calls);
  EXPECT_EQ(123, result.id);
  EXPECT_EQ(NCCL_ALGO_PAT, result.algo);
}

TEST_F(TuningMicrotest, ComputeSelectsLowestGeneralCandidateThenGetsChannels) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  const int ringSimple = GeneralId(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(treeLl) | TuningBit(ringSimple);
  ncclTuningResult_t result = EmptyResult();

  ScopedHook model(g_tuningCostModelSimModel, [&](int id, ncclTuningInput_t*, ncclTuningResult_t* tuning) {
    tuning->valid = 1;
    tuning->timeUs = id == treeLl ? 10.0f : 3.0f;
    return ncclSuccess;
  });
  ScopedHook channels(g_tuningGetChannels, [&](ncclTuningInput_t* passedInput, ncclTuningResult_t* tuning) {
    EXPECT_EQ(&input, passedInput);
    EXPECT_EQ(ringSimple, tuning->id);
    EXPECT_EQ(NCCL_ALGO_RING, tuning->algo);
    EXPECT_EQ(NCCL_PROTO_SIMPLE, tuning->proto);
    tuning->nChannels = 6;
    tuning->nWarps = 7;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(ringSimple, result.id);
  EXPECT_FLOAT_EQ(3.0f, result.timeUs);
  EXPECT_EQ(6, result.nChannels);
  EXPECT_EQ(7, result.nWarps);
  EXPECT_EQ(2, g_tuningFreeCalls);
}

TEST_F(TuningMicrotest, PluginReceivesGeneralTableAndOverridesTimeAndMaxChannels) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  const int ringSimple = GeneralId(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  int contextSentinel = 0;
  comm_->tuner = &plugin_;
  comm_->tunerContext = &contextSentinel;
  ncclTuningInput_t input = MakeInput();
  input.func = ncclFuncBroadcast;
  input.nBytes = 123456;
  input.numPipeOps = 7;
  input.regBuff = 1;
  input.tuningMask = TuningBit(treeLl) | TuningBit(ringSimple) | TuningBit(symFirst);
  ncclTuningResult_t result = EmptyResult();

  ScopedHook model(g_tuningCostModelSimModel, [&](int id, ncclTuningInput_t*, ncclTuningResult_t* tuning) {
    tuning->valid = 1;
    tuning->timeUs = id == treeLl ? 10.0f : id == ringSimple ? 20.0f : 50.0f;
    return ncclSuccess;
  });
  ScopedHook plugin(g_tuningPluginGetCollInfo,
                    [&](void* context, ncclFunc_t func, size_t nBytes, int numPipeOps, float** rawTable,
                        int numAlgo, int numProto, int regBuff, int* nChannels) {
                      EXPECT_EQ(&contextSentinel, context);
                      EXPECT_EQ(ncclFuncBroadcast, func);
                      EXPECT_EQ(123456u, nBytes);
                      EXPECT_EQ(7, numPipeOps);
                      EXPECT_EQ(NCCL_NUM_ALGORITHMS, numAlgo);
                      EXPECT_EQ(NCCL_NUM_PROTOCOLS, numProto);
                      EXPECT_EQ(1, regBuff);
                      auto* table = reinterpret_cast<float (*)[NCCL_NUM_PROTOCOLS]>(rawTable);
                      EXPECT_FLOAT_EQ(10.0f, table[NCCL_ALGO_TREE][NCCL_PROTO_LL]);
                      EXPECT_FLOAT_EQ(20.0f, table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE]);
                      EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, table[NCCL_ALGO_PAT][NCCL_PROTO_LL128]);
                      table[NCCL_ALGO_TREE][NCCL_PROTO_LL] = 30.0f;
                      table[NCCL_ALGO_RING][NCCL_PROTO_SIMPLE] = 5.0f;
                      *nChannels = 7;
                      return ncclSuccess;
                    });
  ScopedHook channels(g_tuningGetChannels, [](ncclTuningInput_t*, ncclTuningResult_t* tuning) {
    EXPECT_EQ(7, tuning->maxChannels);
    tuning->nChannels = 7;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(1, plugin.calls);
  EXPECT_EQ(ringSimple, result.id);
  EXPECT_FLOAT_EQ(5.0f, result.timeUs);
  EXPECT_EQ(7, result.maxChannels);
  EXPECT_EQ(7, result.nChannels);
}

TEST_F(TuningMicrotest, PluginFailureStopsBeforeChannelSelection) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  comm_->tuner = &plugin_;
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(treeLl);
  ncclTuningResult_t result = EmptyResult();
  ScopedHook model(g_tuningCostModelSimModel, ModelSelecting(treeLl));
  ScopedHook plugin(g_tuningPluginGetCollInfo,
                    [](void*, ncclFunc_t, size_t, int, float**, int, int, int, int*) {
                      return ncclRemoteError;
                    });
  ScopedHook channels(g_tuningGetChannels, ChannelsReturning());

  EXPECT_EQ(ncclRemoteError, TuningMicroCompute(&input, &result));
  EXPECT_EQ(1, plugin.calls);
  EXPECT_EQ(0, channels.calls);
  EXPECT_EQ(1, g_tuningFreeCalls);
}

TEST_F(TuningMicrotest, PluginZeroChannelCapIsForwardedToChannelSelection) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  comm_->tuner = &plugin_;
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(treeLl);
  ncclTuningResult_t result = EmptyResult();
  ScopedHook model(g_tuningCostModelSimModel, ModelSelecting(treeLl));
  ScopedHook plugin(g_tuningPluginGetCollInfo,
                    [](void*, ncclFunc_t, size_t, int, float**, int, int, int, int* nChannels) {
                      *nChannels = 0;
                      return ncclSuccess;
                    });
  ScopedHook channels(g_tuningGetChannels, [](ncclTuningInput_t*, ncclTuningResult_t* tuning) {
    EXPECT_EQ(0, tuning->maxChannels);
    tuning->nChannels = 8;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(treeLl, result.id);
  EXPECT_EQ(1, channels.calls);
  EXPECT_EQ(8, result.nChannels);
  EXPECT_EQ(0, result.maxChannels);
}

TEST_F(TuningMicrotest, PartialGeneralCoordinatesAreNeverIndexedOrChanneled) {
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(0);
  comm_->tuner = &plugin_;

  for (const bool missingAlgo : {false, true}) {
    ncclTuningResult_t result = EmptyResult();
    ScopedHook expand(g_tuningExpandId, [&](int, int* algo, int* proto, int* symKernelId, int* ceMethodId) {
      *algo = missingAlgo ? NCCL_ALGO_UNDEF : NCCL_ALGO_TREE;
      *proto = missingAlgo ? NCCL_PROTO_LL : NCCL_PROTO_UNDEF;
      *symKernelId = ncclSymkKernelId_Count;
      *ceMethodId = ncclCeMethodId_Count;
      return ncclSuccess;
    });
    ScopedHook model(g_tuningCostModelSimModel, ModelSelecting(0));
    ScopedHook plugin(g_tuningPluginGetCollInfo,
                      [](void*, ncclFunc_t, size_t, int, float** rawTable, int, int, int, int*) {
                        auto* table = reinterpret_cast<float (*)[NCCL_NUM_PROTOCOLS]>(rawTable);
                        for (int algo = 0; algo < NCCL_NUM_ALGORITHMS; ++algo) {
                          for (int proto = 0; proto < NCCL_NUM_PROTOCOLS; ++proto) {
                            EXPECT_FLOAT_EQ(NCCL_TUNING_IGNORE, table[algo][proto]);
                          }
                        }
                        return ncclSuccess;
                      });
    ScopedHook channels(g_tuningGetChannels, ChannelsReturning());
    SetMicroEnvAbsent("NCCL_ALGO");
    SetMicroEnvAbsent("NCCL_PROTO");
    SetMicroEnvAbsent("NCCL_SYM_KERNEL");

    ncclResult_t status = ncclSuccess;
    const std::string log = RcclUnitTesting::CaptureLog([&] { status = TuningMicroCompute(&input, &result); });
    EXPECT_EQ(ncclInternalError, status);
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, "No algorithm/protocol nor symKernelId available")) << log;
    EXPECT_EQ(1, plugin.calls);
    EXPECT_EQ(0, channels.calls);
    EXPECT_EQ(missingAlgo ? NCCL_ALGO_UNDEF : NCCL_ALGO_TREE, result.algo);
    EXPECT_EQ(missingAlgo ? NCCL_PROTO_LL : NCCL_PROTO_UNDEF, result.proto);
    EXPECT_EQ(NCCL_TUNING_ENTRY_INIT_VALUE, result.maxChannels);
  }
}

TEST_F(TuningMicrotest, CopyEngineCandidateReturnsWithoutGeneralChannelSelection) {
  const int ce = NCCL_TUNING_CE_METHOD_ID_OFFSET;
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(ce);
  ncclTuningResult_t result = EmptyResult();
  ScopedHook model(g_tuningCostModelSimModel, ModelSelecting(ce));
  ScopedHook channels(g_tuningGetChannels, ChannelsReturning());

  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(ce, result.id);
  EXPECT_EQ(NCCL_ALGO_UNDEF, result.algo);
  EXPECT_EQ(NCCL_PROTO_UNDEF, result.proto);
  EXPECT_EQ(ncclSymkKernelId_Count, result.symKernelId);
  EXPECT_EQ(0, result.ceMethodId);
  EXPECT_EQ(0, channels.calls);
}

TEST_F(TuningMicrotest, EfficiencyPolicyOverridesWithEligibleNvlsResources) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  ncclTuningInput_t input = MakeNvlsEligibleInput(ncclFuncAllGather);
  comm_->tuningContext.maxThreads[NCCL_ALGO_NVLS][NCCL_PROTO_SIMPLE] = 320;
  SetMicroEnvAbsent("NCCL_ALGO");
  SetMicroEnvAbsent("NCCL_PROTO");
  ncclTuningResult_t result = EmptyResult();

  ScopedHook model(g_tuningCostModelSimModel, ModelSelecting(treeLl));
  ScopedHook channels(g_tuningGetChannels, ChannelsReturning(8, 2));
  ncclFunc_t expectedFunc = ncclFuncAllGather;
  ScopedHook query(g_tuningNvlsRegResourcesQuery, [&](ncclComm* comm, ncclFunc_t func, int* recChannels) {
    EXPECT_EQ(comm_.get(), comm);
    EXPECT_EQ(expectedFunc, func);
    *recChannels = 4;
    return ncclSuccess;
  });

  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(1, query.calls);
  EXPECT_EQ(NCCL_ALGO_NVLS, result.algo);
  EXPECT_EQ(NCCL_PROTO_SIMPLE, result.proto);
  EXPECT_EQ(4, result.nChannels);
  EXPECT_EQ(4, result.maxChannels);
  EXPECT_EQ(10, result.nWarps);

  expectedFunc = ncclFuncReduceScatter;
  input.func = expectedFunc;
  result = EmptyResult();
  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(2, query.calls);
  EXPECT_EQ(NCCL_ALGO_NVLS, result.algo);

  expectedFunc = ncclFuncAllGather;
  input.func = expectedFunc;
  comm_->nNodes = 2;
  input.collNetSupport = 1;
  result = EmptyResult();
  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(3, query.calls);
  EXPECT_EQ(NCCL_ALGO_NVLS, result.algo);

  ScopedHook exact(g_tuningNvlsRegResourcesQuery, [](ncclComm*, ncclFunc_t, int* recChannels) {
    *recChannels = 8;
    return ncclSuccess;
  });
  result = EmptyResult();
  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(NCCL_ALGO_NVLS, result.algo);
  EXPECT_EQ(8, result.maxChannels);
  EXPECT_EQ(10, result.nWarps);

  ScopedHook tooMany(g_tuningNvlsRegResourcesQuery, [](ncclComm*, ncclFunc_t, int* recChannels) {
    *recChannels = 9;
    return ncclSuccess;
  });
  result = EmptyResult();
  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(NCCL_ALGO_TREE, result.algo);
  EXPECT_EQ(8, result.nChannels);
}

TEST_F(TuningMicrotest, EfficiencyPolicyGuardsPreventIneligibleNvlsOverride) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  ncclTuningInput_t input = MakeInput();
  auto resetEligible = [&] {
    comm_->tuner = nullptr;
    comm_->nNodes = 1;
    comm_->MNNVL = 0;
    input = MakeNvlsEligibleInput(ncclFuncReduceScatter);
    SetMicroEnvAbsent("NCCL_ALGO");
    SetMicroEnvAbsent("NCCL_PROTO");
  };
  resetEligible();

  ScopedHook model(g_tuningCostModelSimModel, ModelSelecting(treeLl));
  ScopedHook channels(g_tuningGetChannels, ChannelsReturning(8));
  ScopedHook query(g_tuningNvlsRegResourcesQuery, [](ncclComm*, ncclFunc_t, int* recChannels) {
    *recChannels = 4;
    return ncclSuccess;
  });
  ScopedHook plugin(g_tuningPluginGetCollInfo,
                    [](void*, ncclFunc_t, size_t, int, float**, int, int, int, int*) { return ncclSuccess; });

  auto expectBlocked = [&] {
    const int callsBefore = query.calls;
    ncclTuningResult_t result = EmptyResult();
    ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
    EXPECT_EQ(callsBefore, query.calls);
    EXPECT_EQ(NCCL_ALGO_TREE, result.algo);
  };

  input.CTAPolicy = 0;
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  input.CTAPolicy = NCCL_CTA_POLICY_ZERO;
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  SetMicroEnv("NCCL_ALGO", "Ring");
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  SetMicroEnv("NCCL_PROTO", "Simple");
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  comm_->MNNVL = 1;
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  input.tuningMask = TuningBit(treeLl);
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  input.regBuff = 0;
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  input.func = ncclFuncAllReduce;
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  input.nvlsSupport = 0;
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  comm_->nNodes = 2;
  input.collNetSupport = 1;
  input.nvlsSupport = 0;
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  comm_->nNodes = 2;
  input.collNetSupport = 0;
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
  resetEligible();
  comm_->tuner = &plugin_;
  ASSERT_NO_FATAL_FAILURE(expectBlocked());
}

TEST_F(TuningMicrotest, EfficiencyPolicyPropagatesNvlsQueryFailure) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  ncclTuningInput_t input = MakeNvlsEligibleInput(ncclFuncAllGather);
  SetMicroEnvAbsent("NCCL_ALGO");
  SetMicroEnvAbsent("NCCL_PROTO");
  ncclTuningResult_t result = EmptyResult();
  ScopedHook model(g_tuningCostModelSimModel, ModelSelecting(treeLl));
  ScopedHook channels(g_tuningGetChannels, ChannelsReturning(8));
  ScopedHook query(g_tuningNvlsRegResourcesQuery,
                   [](ncclComm*, ncclFunc_t, int*) { return ncclRemoteError; });

  EXPECT_EQ(ncclRemoteError, TuningMicroCompute(&input, &result));
  EXPECT_EQ(1, query.calls);
  EXPECT_EQ(NCCL_ALGO_UNDEF, result.algo);
}

TEST_F(TuningMicrotest, SymmetricFallbackUsesGeneralTuningWhenExecutionRequiresIt) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(symSecond);
  input.winRegType = ncclSymSendNonregRecvNonreg;
  comm_->intraRanks = 2;
  ncclTuningResult_t result = EmptyResult();
  bool sawGeneralMask = false;

  ScopedHook model(g_tuningCostModelSimModel, [&](int id, ncclTuningInput_t* modelInput,
                                                  ncclTuningResult_t* tuning) {
    if (modelInput->tuningMask == NCCL_TUNING_MASK_GENERAL_KERNELS) {
      sawGeneralMask = true;
      tuning->valid = id == treeLl;
      tuning->timeUs = 2.0f;
    } else {
      tuning->valid = id == symSecond;
      tuning->timeUs = 1.0f;
    }
    return ncclSuccess;
  });
  ScopedHook llMask(g_tuningSymkLLKernelMask, [] { return 1 << 1; });
  ScopedHook params(
      g_loadParam, TuningParams({{"SINGLE_PROC_MEM_REG_ENABLE", 0}, {"SYM_NOWIN_ENABLE", 1}}));
  ScopedHook channels(g_tuningGetChannels, ChannelsReturning(5));

  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_TRUE(sawGeneralMask);
  EXPECT_EQ(treeLl, result.id);
  EXPECT_EQ(NCCL_ALGO_TREE, result.algo);
  EXPECT_EQ(NCCL_PROTO_LL, result.proto);
  EXPECT_EQ(5, result.nChannels);
}

TEST_F(TuningMicrotest, SymmetricRegisteredAndNonLlCasesDoNotFallback) {
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(symSecond);
  ncclTuningResult_t result = EmptyResult();
  int generalCalls = 0;

  ScopedHook model(g_tuningCostModelSimModel, [&](int id, ncclTuningInput_t* modelInput,
                                                  ncclTuningResult_t* tuning) {
    if (modelInput->tuningMask == NCCL_TUNING_MASK_GENERAL_KERNELS) ++generalCalls;
    tuning->valid = id == symSecond;
    tuning->timeUs = 1.0f;
    return ncclSuccess;
  });

  input.winRegType = ncclSymSendRegRecvReg;
  {
    ScopedHook llMask(g_tuningSymkLLKernelMask, [] { return 1 << 1; });
    ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
    EXPECT_EQ(1, result.symKernelId);
  }

  input.winRegType = ncclSymSendNonregRecvNonreg;
  result = EmptyResult();
  {
    ScopedHook llMask(g_tuningSymkLLKernelMask, [] { return 0; });
    ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
    EXPECT_EQ(1, result.symKernelId);
  }

  EXPECT_EQ(0, generalCalls);
}

TEST_F(TuningMicrotest, SymmetricHalfRegisteredBuffersDoNotForceFallback) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(symSecond);
  int generalPasses = 0;

  ScopedHook model(g_tuningCostModelSimModel, [&](int id, ncclTuningInput_t* modelInput,
                                                  ncclTuningResult_t* tuning) {
    if (modelInput->tuningMask == NCCL_TUNING_MASK_GENERAL_KERNELS) {
      if (id == treeLl) ++generalPasses;
      tuning->valid = id == treeLl;
      tuning->timeUs = 2.0f;
    } else {
      tuning->valid = id == symSecond;
      tuning->timeUs = 1.0f;
    }
    return ncclSuccess;
  });
  ScopedHook llMask(g_tuningSymkLLKernelMask, [] { return 1 << 1; });
  ScopedHook channels(g_tuningGetChannels, ChannelsReturning());

  comm_->intraRanks = 2;
  input.winRegType = ncclSymSendNonregRecvReg;
  {
    ScopedHook params(
        g_loadParam, TuningParams({{"SINGLE_PROC_MEM_REG_ENABLE", 0}, {"SYM_NOWIN_ENABLE", 1}}));
    ncclTuningResult_t result = EmptyResult();
    ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
    EXPECT_EQ(1, result.symKernelId);
    EXPECT_EQ(NCCL_ALGO_UNDEF, result.algo);
    EXPECT_EQ(1, generalPasses);
  }

  comm_->intraRanks = 1;
  input.winRegType = ncclSymSendRegRecvNonreg;
  generalPasses = 0;
  {
    ScopedHook params(g_loadParam, TuningParams({{"SYM_NOWIN_ENABLE", 0}}));
    ncclTuningResult_t result = EmptyResult();
    ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
    EXPECT_EQ(1, result.symKernelId);
    EXPECT_EQ(NCCL_ALGO_UNDEF, result.algo);
    EXPECT_EQ(1, generalPasses);
  }
}

TEST_F(TuningMicrotest, SymmetricNoWindowPreferenceUsesGeneralProtocolAsTiebreaker) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  const int treeSimple = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE);
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(symSecond);
  input.winRegType = ncclSymSendNonregRecvNonreg;
  int generalProto = NCCL_PROTO_LL;
  int generalPasses = 0;
  ncclTuningResult_t result = EmptyResult();

  ScopedHook model(g_tuningCostModelSimModel, [&](int id, ncclTuningInput_t* modelInput,
                                                  ncclTuningResult_t* tuning) {
    if (modelInput->tuningMask == NCCL_TUNING_MASK_GENERAL_KERNELS) {
      if (id == treeLl) ++generalPasses;
      const int selected = generalProto == NCCL_PROTO_LL ? treeLl : treeSimple;
      tuning->valid = id == selected;
      tuning->timeUs = 2.0f;
    } else {
      tuning->valid = id == symSecond;
      tuning->timeUs = 1.0f;
    }
    return ncclSuccess;
  });
  ScopedHook llMask(g_tuningSymkLLKernelMask, [] { return 1 << 1; });
  {
    ScopedHook params(g_loadParam, TuningParams({{"SYM_NOWIN_ENABLE", 1}}));

    ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
    EXPECT_EQ(1, result.symKernelId);
    EXPECT_EQ(NCCL_ALGO_UNDEF, result.algo);
    EXPECT_EQ(1, generalPasses);

    generalProto = NCCL_PROTO_SIMPLE;
    generalPasses = 0;
    result = EmptyResult();
    ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
    EXPECT_EQ(treeSimple, result.id);
    EXPECT_EQ(NCCL_ALGO_TREE, result.algo);
    EXPECT_EQ(NCCL_PROTO_SIMPLE, result.proto);
    EXPECT_EQ(1, generalPasses);
  }

  generalProto = NCCL_PROTO_LL;
  generalPasses = 0;
  result = EmptyResult();
  {
    ScopedHook params(g_loadParam, TuningParams({{"SYM_NOWIN_ENABLE", 0}}));
    ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
    EXPECT_EQ(treeLl, result.id);
    EXPECT_EQ(NCCL_ALGO_TREE, result.algo);
    EXPECT_EQ(NCCL_PROTO_LL, result.proto);
    EXPECT_EQ(1, generalPasses);
  }
}

TEST_F(TuningMicrotest, SingleProcessRegistrationAvoidsImmediateSymmetricFallback) {
  const int treeLl = GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL);
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(symSecond);
  input.winRegType = ncclSymSendNonregRecvNonreg;
  comm_->intraRanks = 2;
  ncclTuningResult_t result = EmptyResult();

  ScopedHook model(g_tuningCostModelSimModel, [&](int id, ncclTuningInput_t* modelInput,
                                                  ncclTuningResult_t* tuning) {
    if (modelInput->tuningMask == NCCL_TUNING_MASK_GENERAL_KERNELS) {
      tuning->valid = id == treeLl;
      tuning->timeUs = 2.0f;
    } else {
      tuning->valid = id == symSecond;
      tuning->timeUs = 1.0f;
    }
    return ncclSuccess;
  });
  ScopedHook llMask(g_tuningSymkLLKernelMask, [] { return 1 << 1; });
  ScopedHook params(
      g_loadParam, TuningParams({{"SINGLE_PROC_MEM_REG_ENABLE", 1}, {"SYM_NOWIN_ENABLE", 1}}));

  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(1, result.symKernelId);
  EXPECT_EQ(NCCL_ALGO_UNDEF, result.algo);
}

TEST_F(TuningMicrotest, MissingSymmetricCandidateFallsBackToGeneralMask) {
  const int ringSimple = GeneralId(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE);
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(symSecond);
  ncclTuningResult_t result = EmptyResult();

  ScopedHook model(g_tuningCostModelSimModel, [&](int id, ncclTuningInput_t* modelInput,
                                                  ncclTuningResult_t* tuning) {
    if (modelInput->tuningMask == NCCL_TUNING_MASK_GENERAL_KERNELS) {
      tuning->valid = id == ringSimple;
      tuning->timeUs = 3.0f;
    } else {
      tuning->valid = 0;
    }
    return ncclSuccess;
  });
  ScopedHook channels(g_tuningGetChannels, ChannelsReturning(6));

  ASSERT_EQ(ncclSuccess, TuningMicroCompute(&input, &result));
  EXPECT_EQ(ringSimple, result.id);
  EXPECT_EQ(NCCL_ALGO_RING, result.algo);
  EXPECT_EQ(NCCL_PROTO_SIMPLE, result.proto);
  EXPECT_EQ(6, result.nChannels);
}

TEST_F(TuningMicrotest, NoValidTuningDistinguishesConfigurationErrors) {
  ncclTuningInput_t input = MakeInput();
  input.tuningMask = TuningBit(GeneralId(NCCL_ALGO_TREE, NCCL_PROTO_LL));
  ncclTuningResult_t result = EmptyResult();
  SetMicroEnvAbsent("NCCL_ALGO");
  SetMicroEnvAbsent("NCCL_PROTO");
  SetMicroEnvAbsent("NCCL_SYM_KERNEL");

  ncclResult_t status = ncclSuccess;
  const std::string noConfigLog =
      RcclUnitTesting::CaptureLog([&] { status = TuningMicroCompute(&input, &result); });
  EXPECT_EQ(ncclInternalError, status);
  EXPECT_FALSE(RcclUnitTesting::LogHas(noConfigLog, "was set to")) << noConfigLog;
  EXPECT_EQ(NCCL_ALGO_UNDEF, result.algo);
  EXPECT_EQ(NCCL_PROTO_UNDEF, result.proto);
  EXPECT_EQ(ncclSymkKernelId_Count, result.symKernelId);
  EXPECT_EQ(ncclCeMethodId_Count, result.ceMethodId);

  const std::pair<const char*, const char*> configured[] = {
      {"NCCL_ALGO", "Ring"}, {"NCCL_PROTO", "Simple"}, {"NCCL_SYM_KERNEL", "AllGather_LL"}};
  for (const auto& [name, value] : configured) {
    SetMicroEnvAbsent("NCCL_ALGO");
    SetMicroEnvAbsent("NCCL_PROTO");
    SetMicroEnvAbsent("NCCL_SYM_KERNEL");
    SetMicroEnv(name, value);
    result = EmptyResult();
    status = ncclSuccess;
    const std::string log = RcclUnitTesting::CaptureLog([&] { status = TuningMicroCompute(&input, &result); });
    EXPECT_EQ(ncclInvalidUsage, status) << name;
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, name)) << log;
    EXPECT_TRUE(RcclUnitTesting::LogHas(log, value)) << log;
  }
}

}  // namespace
