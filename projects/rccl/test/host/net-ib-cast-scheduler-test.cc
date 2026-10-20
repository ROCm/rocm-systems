/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net_ib_cast/scheduler.cc (via NET_IB_CAST_SCHEDULER_CC_PATH): the
// multi-QP scheduler, its env parsing and the test introspection API.

#include <gtest/gtest.h>

#include <unistd.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "../common/LogCapture.hpp"
#include "ScopedHook.h"
#include "common_cast.h"
#include "fakes/env_fakes.h"
#include "fakes/libc_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/net_ib_cast_common_fakes.h"
#include "fakes/net_ib_cast_connect_fakes.h"
#include "fakes/transport_stubs.h"
#include "ibvsymbols.h"
#include "p2p_resiliency_cast.h"

#include "fakes/param_redirect.h"

// The RTT bookkeeping reads CLOCK_MONOTONIC; route it through a hook so a sample's age is exact.
static std::function<int(clockid_t, struct timespec*)> g_clockGettime = ::clock_gettime;

#include "fakes/libc_seam.h"
#undef send  // ncclIbRequest::send, not the socket call
#define clock_gettime(clk, ts) g_clockGettime((clk), (ts))
#include NET_IB_CAST_SCHEDULER_CC_PATH
#undef clock_gettime
#include "fakes/libc_seam_undef.h"

extern struct ncclIbvSymbols ibvSymbols;

namespace {

using RcclUnitTesting::CaptureLog;

bool Has(const std::string& log, const char* needle) { return log.find(needle) != std::string::npos; }

// Each comm is ~1 MB; keep it off the stack. Value-initialised, like the calloc production uses.
template <typename Comm>
std::unique_ptr<Comm> NewComm() {
  return std::unique_ptr<Comm>(new Comm());
}

class NetIbCastSchedulerMicrotest : public ::testing::Test {
 protected:
  void TearDown() override {
    ResetEnvFakes();
    ResetLibcFakes();
    ResetTransportStubs();
    ResetNetIbCastConnectFakes();
    ResetNetIbCastCommonFakes();
    castGlobalQpSchedParms = {};
    stagedSchedParms = {ncclNumFuncs};
    IbCastQpSchedProxyPrevCollType = ncclNumFuncs;
    IbCastQpSchedProxyPrevMsgSz = 0;
    if (IbCastQpSchedLogStream) fclose(IbCastQpSchedLogStream);
    IbCastQpSchedLogStream = nullptr;
    ncclDebugMask = 0;
    ncclDebugLevelMask = 0;
  }

  // Unset params read as their compile-time default, as production does.
  int64_t Param(const char* env, int64_t deft) {
    auto it = params_.find(env);
    return it == params_.end() ? deft : it->second;
  }

  static void LogInfo() {
    ncclDebugMask = NCCL_ALL;
    ncclDebugLevelMask = 1u << NCCL_LOG_INFO;
  }

  ncclIbQpSchedParms Init(ncclResult_t expect = ncclSuccess) {
    ncclIbQpSchedParms parms{};
    EXPECT_EQ(IbCastQpSchedInitParms(&parms), expect);
    return parms;
  }

  std::map<std::string, int64_t> params_;
  ScopedHook<int64_t(const char*, int64_t)> loadParam_{g_loadParam,
                                                       [this](const char* env, int64_t d) { return Param(env, d); }};
};

TEST_F(NetIbCastSchedulerMicrotest, UseQpSched_ExplicitParamOverridesNetAndAinic) {
  SetMicroEnv("NCCL_NET", "ib-cast");
  params_["RCCL_IB_QP_SCHED_ENABLE"] = 0;
  EXPECT_FALSE(rcclUseIbCastQpSched());
  g_rcclUseAinicValue = true;
  params_["RCCL_IB_QP_SCHED_ENABLE"] = 1;
  EXPECT_TRUE(rcclUseIbCastQpSched());
}

TEST_F(NetIbCastSchedulerMicrotest, UseQpSched_ForcedOnWithByOrder_WarnsButStaysOn) {
  ScopedHook byOrder(g_ibCastByOrderRequested, [] { return true; });
  params_["RCCL_IB_QP_SCHED_ENABLE"] = 1;
  bool on = false;
  const std::string log = CaptureLog([&] { on = rcclUseIbCastQpSched(); });
  EXPECT_TRUE(on);
  EXPECT_TRUE(Has(log, "unsupported combination")) << log;

  params_["RCCL_IB_QP_SCHED_ENABLE"] = 0;
  EXPECT_TRUE(CaptureLog([&] { on = rcclUseIbCastQpSched(); }).empty());
  EXPECT_FALSE(on);
}

TEST_F(NetIbCastSchedulerMicrotest, UseQpSched_ByOrderByDefault_DisablesWithAWarning) {
  ScopedHook byOrder(g_ibCastByOrderRequested, [] { return true; });
  SetMicroEnv("NCCL_NET", "ib-cast");
  bool on = true;
  const std::string log = CaptureLog([&] { on = rcclUseIbCastQpSched(); });
  EXPECT_FALSE(on);
  EXPECT_TRUE(Has(log, "disabling the QP scheduler, split-data and WRR")) << log;
}

TEST_F(NetIbCastSchedulerMicrotest, UseQpSched_NetIbCastForcesItOnAinicCaseInsensitively) {
  g_rcclUseAinicValue = true;
  SetMicroEnv("NCCL_NET", "IB-Cast");
  EXPECT_TRUE(rcclUseIbCastQpSched());
  SetMicroEnv("NCCL_NET", "IB");
  EXPECT_FALSE(rcclUseIbCastQpSched());
  SetMicroEnvAbsent("NCCL_NET");
  EXPECT_FALSE(rcclUseIbCastQpSched());
  g_rcclUseAinicValue = false;
  EXPECT_TRUE(rcclUseIbCastQpSched());
}

TEST_F(NetIbCastSchedulerMicrotest, InitParms_Disabled_KeepsDefaultsAndIgnoresTheKnobs) {
  params_["RCCL_IB_QP_SCHED_ENABLE"] = 0;
  params_["RCCL_IB_QP_SCHED_UPDATE_INTERVAL"] = 100;
  params_["RCCL_IB_QP_SCHED_SPLIT_DATA_MIN"] = 4096;
  SetMicroEnv("RCCL_IB_QP_SCHED_WEIGHT", "0.5");
  const ncclIbQpSchedParms p = Init();
  EXPECT_FALSE(p.enable);
  EXPECT_TRUE(p.doWrr);
  EXPECT_EQ(p.updateInterval, QP_SCHED_UPDATE_DEF);
  EXPECT_EQ(p.splitDataMin, (uint32_t)QP_SCHED_SPLIT_DATA_MIN_DEF);
  EXPECT_EQ(p.weightNew, (double)QP_SCHED_WEIGHT_DEF);
  EXPECT_FALSE(p.logEnable);
}

TEST_F(NetIbCastSchedulerMicrotest, InitParms_Enabled_Defaults) {
  const ncclIbQpSchedParms p = Init();
  EXPECT_TRUE(p.enable);
  EXPECT_TRUE(p.wrrEnable);
  EXPECT_TRUE(p.doWrr);
  EXPECT_FALSE(p.splitData);
  EXPECT_EQ(p.resetInterval, QP_SCHED_RESET_DEF);
  EXPECT_EQ(p.updateInterval, QP_SCHED_UPDATE_DEF);
  EXPECT_EQ(p.weightNew, (double)QP_SCHED_WEIGHT_DEF);
  EXPECT_EQ(p.splitDataMin, (uint32_t)QP_SCHED_SPLIT_DATA_MIN_DEF);
  EXPECT_EQ(p.logInterval, QP_SCHED_LOG_DEF);
  EXPECT_FALSE(p.logEnable);
  EXPECT_EQ(IbCastQpSchedLogStream, nullptr);
}

TEST_F(NetIbCastSchedulerMicrotest, InitParms_Intervals_AreConvertedToNanoseconds) {
  params_["RCCL_IB_QP_SCHED_RESET_INTERVAL"] = 5;
  params_["RCCL_IB_QP_SCHED_UPDATE_INTERVAL"] = 100;
  ncclIbQpSchedParms p = Init();
  EXPECT_EQ(p.resetInterval, 5 * NSEC_PER_MSEC);
  EXPECT_EQ(p.updateInterval, 100 * NSEC_PER_USEC);

  params_["RCCL_IB_QP_SCHED_RESET_INTERVAL"] = 0;
  EXPECT_EQ(Init().resetInterval, (uint64_t)QP_SCHED_RESET_NEVER);
}

TEST_F(NetIbCastSchedulerMicrotest, InitParms_UpdateInterval_HonoursOnlyItsRange) {
  const int64_t maxUs = QP_SCHED_UPDATE_MAX / NSEC_PER_USEC;
  params_["RCCL_IB_QP_SCHED_UPDATE_INTERVAL"] = 1;
  EXPECT_EQ(Init().updateInterval, QP_SCHED_UPDATE_MIN);
  params_["RCCL_IB_QP_SCHED_UPDATE_INTERVAL"] = maxUs;
  EXPECT_EQ(Init().updateInterval, QP_SCHED_UPDATE_MAX);
  params_["RCCL_IB_QP_SCHED_UPDATE_INTERVAL"] = maxUs + 1;
  EXPECT_EQ(Init().updateInterval, QP_SCHED_UPDATE_DEF);
  params_["RCCL_IB_QP_SCHED_UPDATE_INTERVAL"] = 0;
  EXPECT_EQ(Init().updateInterval, QP_SCHED_UPDATE_DEF);
}

TEST_F(NetIbCastSchedulerMicrotest, InitParms_Weight_RcclNameWinsAndRangeIsInclusive) {
  SetMicroEnv("NCCL_IB_QP_SCHED_WEIGHT", "0.25");
  EXPECT_EQ(Init().weightNew, 0.25);
  SetMicroEnv("RCCL_IB_QP_SCHED_WEIGHT", "0.75");
  EXPECT_EQ(Init().weightNew, 0.75);
  SetMicroEnv("RCCL_IB_QP_SCHED_WEIGHT", "1.0");
  EXPECT_EQ(Init().weightNew, QP_SCHED_WEIGHT_MAX);
  SetMicroEnv("RCCL_IB_QP_SCHED_WEIGHT", "1.5");
  EXPECT_EQ(Init().weightNew, (double)QP_SCHED_WEIGHT_DEF);
  SetMicroEnv("RCCL_IB_QP_SCHED_WEIGHT", "-0.5");
  EXPECT_EQ(Init().weightNew, (double)QP_SCHED_WEIGHT_DEF);
}

TEST_F(NetIbCastSchedulerMicrotest, InitParms_SplitDataMin_OnlyPositiveValuesApply) {
  params_["RCCL_IB_QP_SCHED_SPLIT_DATA_MIN"] = 4096;
  EXPECT_EQ(Init().splitDataMin, 4096u);
  params_["RCCL_IB_QP_SCHED_SPLIT_DATA_MIN"] = 0;
  EXPECT_EQ(Init().splitDataMin, (uint32_t)QP_SCHED_SPLIT_DATA_MIN_DEF);
}

TEST_F(NetIbCastSchedulerMicrotest, InitParms_SplitData_KeepsWrrWhenTheKnobIsOn) {
  params_["IB_SPLIT_DATA_ON_QPS"] = 1;
  const ncclIbQpSchedParms p = Init();
  EXPECT_TRUE(p.splitData);
  EXPECT_TRUE(p.doWrr);
}

// Pins AICOMRCCL-3077; a fix flips this pin.
TEST_F(NetIbCastSchedulerMicrotest, InitParms_KnownBug_WrrEnableZeroCannotTurnWrrOff) {
  params_["IB_SPLIT_DATA_ON_QPS"] = 1;
  params_["RCCL_IB_QP_SCHED_WRR_ENABLE"] = 0;
  const ncclIbQpSchedParms p = Init();
  EXPECT_TRUE(p.splitData);
  EXPECT_TRUE(p.wrrEnable);
  EXPECT_TRUE(p.doWrr);
}

class NetIbCastSchedulerLogMicrotest : public NetIbCastSchedulerMicrotest {
 protected:
  void SetUp() override {
    char tmpl[] = "/tmp/rccl-net-ib-cast-sched-XXXXXX";
    ASSERT_NE(mkdtemp(tmpl), nullptr);
    dir_ = tmpl;
  }
  void TearDown() override {
    NetIbCastSchedulerMicrotest::TearDown();
    std::filesystem::remove_all(dir_);
  }
  std::string LogFile() const { return dir_ + "/cast_log_nodeA_" + std::to_string(getpid()); }

  std::string dir_;
  ScopedHook<int(char*, size_t)> hostname_{g_gethostname, [](char* name, size_t len) {
                                             snprintf(name, len, "nodeA");
                                             return 0;
                                           }};
};

TEST_F(NetIbCastSchedulerLogMicrotest, InitParms_LogPath_OpensAPerProcessFile) {
  SetMicroEnv("RCCL_IB_QP_SCHED_LOG_PATH", dir_.c_str());
  params_["RCCL_IB_QP_SCHED_LOG_INTERVAL"] = 200;
  const ncclIbQpSchedParms p = Init();
  EXPECT_TRUE(p.logEnable);
  EXPECT_EQ(p.logInterval, 200 * NSEC_PER_USEC);
  ASSERT_NE(IbCastQpSchedLogStream, nullptr);
  EXPECT_TRUE(std::filesystem::exists(LogFile())) << LogFile();
}

TEST_F(NetIbCastSchedulerLogMicrotest, InitParms_LogPathAlias_AndOutOfRangeInterval) {
  SetMicroEnv("NCCL_IB_QP_SCHED_LOG_PATH", dir_.c_str());
  params_["RCCL_IB_QP_SCHED_LOG_INTERVAL"] = 0;
  const ncclIbQpSchedParms p = Init();
  EXPECT_TRUE(p.logEnable);
  EXPECT_EQ(p.logInterval, QP_SCHED_LOG_DEF);
  EXPECT_TRUE(std::filesystem::exists(LogFile()));
}

TEST_F(NetIbCastSchedulerLogMicrotest, InitParms_LogPathUnwritable_DisablesLogging) {
  const std::string missing = dir_ + "/missing";
  SetMicroEnv("RCCL_IB_QP_SCHED_LOG_PATH", missing.c_str());
  ncclIbQpSchedParms p{};
  const std::string log = CaptureLog([&] { p = Init(); });
  EXPECT_FALSE(p.logEnable);
  EXPECT_TRUE(p.enable);
  EXPECT_EQ(IbCastQpSchedLogStream, nullptr);
  EXPECT_TRUE(Has(log, "fopen failed")) << log;
}

TEST_F(NetIbCastSchedulerLogMicrotest, InitParms_FileNameAllocationFails_ReturnsInternalError) {
  SetMicroEnv("RCCL_IB_QP_SCHED_LOG_PATH", dir_.c_str());
  ScopedHook calloc(g_calloc, [](size_t, size_t) -> void* { return nullptr; });
  ncclIbQpSchedParms p{};
  const std::string log = CaptureLog([&] { p = Init(ncclInternalError); });
  EXPECT_EQ(calloc.calls, 1);
  EXPECT_FALSE(p.logEnable);
  EXPECT_TRUE(Has(log, "calloc failed")) << log;
}

TEST_F(NetIbCastSchedulerMicrotest, LogSched_WritesOneLinePerQp) {
  char* buf = nullptr;
  size_t len = 0;
  IbCastQpSchedLogStream = open_memstream(&buf, &len);
  ASSERT_NE(IbCastQpSchedLogStream, nullptr);
  auto comm = NewComm<ncclIbSendComm>();
  comm->base.nqps = 2;
  comm->base.vProps.ndevs = 1;
  comm->devs[0].base.ibDevN = 3;
  snprintf(IbCastDevs[3].devName, sizeof(IbCastDevs[3].devName), "mlx5_3");
  comm->base.qpTxSched[1] = {0.5, 0.25, 0.75};
  comm->base.qpTxStats[1].rtt = 1500.0;
  IbCastLogSched(comm.get());
  fflush(IbCastQpSchedLogStream);
  const std::string out(buf, len);
  char head[64];
  snprintf(head, sizeof(head), "comm %p: num qp's 2 ndevs 1\n", (void*)comm.get());
  EXPECT_EQ(out.rfind(head, 0), 0u) << out;
  EXPECT_TRUE(Has(out, "  qp[01]: dev[0]=mlx5_3 (ibDev 3): minW=0.2500 curW=0.5000 maxW=0.7500 lat=1.50us\n")) << out;
  fclose(IbCastQpSchedLogStream);
  IbCastQpSchedLogStream = nullptr;
  free(buf);
  IbCastDevs[3].devName[0] = '\0';
}

TEST_F(NetIbCastSchedulerMicrotest, UpdateSchedParmsTry_ByOrder_LeavesTheCommAlone) {
  auto comm = NewComm<ncclIbSendComm>();
  comm->base.recvMatchingScheme = BY_ORDER;
  castGlobalQpSchedParms.splitDataMin = 123;
  IbCastUpdateSchedParmsTry(&comm->base, 1, 64);
  EXPECT_FALSE(comm->base.schedParmsInit);
  EXPECT_EQ(comm->base.schedParms.splitDataMin, 0u);
}

TEST_F(NetIbCastSchedulerMicrotest, UpdateSchedParmsTry_FirstCall_CopiesTheGlobalParmsOnce) {
  auto comm = NewComm<ncclIbSendComm>();
  castGlobalQpSchedParms.splitDataMin = 123;
  IbCastUpdateSchedParmsTry(&comm->base, 1, 64);
  EXPECT_TRUE(comm->base.schedParmsInit);
  EXPECT_EQ(comm->base.schedParms.splitDataMin, 123u);
  castGlobalQpSchedParms.splitDataMin = 456;
  IbCastUpdateSchedParmsTry(&comm->base, 1, 64);
  EXPECT_EQ(comm->base.schedParms.splitDataMin, 123u);
}

TEST_F(NetIbCastSchedulerMicrotest, UpdateSchedParmsTry_NewEpoch_AdoptsTheStagedParmsOnce) {
  auto comm = NewComm<ncclIbSendComm>();
  comm->base.resetRttDone = true;
  stagedSchedParms.prodEpoch = 2;
  stagedSchedParms.collType = ncclFuncAllReduce;
  stagedSchedParms.msgSz = 1 << 20;
  stagedSchedParms.parms.splitDataMin = 777;
  stagedSchedParms.parms.resetRtt = true;
  LogInfo();
  std::string log = CaptureLog([&] { IbCastUpdateSchedParmsTry(&comm->base, 2, 4096); });
  EXPECT_EQ(comm->base.schedParms.splitDataMin, 777u);
  EXPECT_EQ(comm->base.stagedParmsConEpoch, 2u);
  EXPECT_FALSE(comm->base.resetRttDone);
  EXPECT_EQ(IbCastQpSchedProxyPrevCollType, ncclFuncAllReduce);
  EXPECT_EQ(IbCastQpSchedProxyPrevMsgSz, (size_t)1 << 20);
  EXPECT_TRUE(Has(log, "CollType = All_Reduce  MsgSz = 1048576  ChunkSz = 4096  NumChunks = 2")) << log;

  stagedSchedParms.parms.splitDataMin = 888;
  comm->base.resetRttDone = true;
  log = CaptureLog([&] { IbCastUpdateSchedParmsTry(&comm->base, 2, 4096); });
  EXPECT_EQ(comm->base.schedParms.splitDataMin, 777u);
  EXPECT_TRUE(comm->base.resetRttDone);
  EXPECT_TRUE(log.empty()) << log;
}

// A second comm adopting the same epoch sees no coll/size change, so it adopts silently.
TEST_F(NetIbCastSchedulerMicrotest, UpdateSchedParmsTry_UnchangedCollAndSize_AdoptsWithoutLogging) {
  auto a = NewComm<ncclIbSendComm>();
  auto b = NewComm<ncclIbRecvComm>();
  stagedSchedParms.prodEpoch = 1;
  stagedSchedParms.collType = ncclFuncBroadcast;
  stagedSchedParms.msgSz = 8;
  stagedSchedParms.parms.splitDataMin = 5;
  LogInfo();
  EXPECT_FALSE(CaptureLog([&] { IbCastUpdateSchedParmsTry(&a->base, 1, 8); }).empty());
  EXPECT_TRUE(CaptureLog([&] { IbCastUpdateSchedParmsTry(&b->base, 1, 8); }).empty());
  EXPECT_EQ(b->base.schedParms.splitDataMin, 5u);
  EXPECT_EQ(b->base.stagedParmsConEpoch, 1u);
}

struct CollName {
  ncclFunc_t coll;
  const char* text;
};

class NetIbCastSchedulerCollNameMicrotest : public NetIbCastSchedulerMicrotest,
                                            public ::testing::WithParamInterface<CollName> {};

TEST_P(NetIbCastSchedulerCollNameMicrotest, UpdateSchedParmsTry_LogsTheCollectiveName) {
  auto comm = NewComm<ncclIbSendComm>();
  stagedSchedParms.prodEpoch = 1;
  stagedSchedParms.collType = GetParam().coll;
  stagedSchedParms.msgSz = 64;
  LogInfo();
  const std::string log = CaptureLog([&] { IbCastUpdateSchedParmsTry(&comm->base, 1, 64); });
  EXPECT_TRUE(Has(log, (std::string("CollType = ") + GetParam().text + "  MsgSz = 64").c_str())) << log;
}

INSTANTIATE_TEST_SUITE_P(Colls, NetIbCastSchedulerCollNameMicrotest,
                         ::testing::Values(CollName{ncclFuncBroadcast, "Broadcast"}, CollName{ncclFuncReduce, "Reduce"},
                                           CollName{ncclFuncAllGather, "All_Gather"},
                                           CollName{ncclFuncReduceScatter, "Reduce_Scatter"},
                                           CollName{ncclFuncAllReduce, "All_Reduce"},
                                           CollName{ncclFuncSendRecv, "5"}));

class NetIbCastSchedulerTxMicrotest : public NetIbCastSchedulerMicrotest {
 protected:
  void SetUp() override {
    comm_ = NewComm<ncclIbSendComm>();
    comm_->base.nqps = 2;
  }
  void Rtts(double a, double b, uint64_t minSampleOfA) {
    comm_->base.qpTxStats[0].rtt = a;
    comm_->base.qpTxStats[0].minRttSample = minSampleOfA;
    comm_->base.qpTxStats[1].rtt = b;
    comm_->base.qpTxStats[1].minRttSample = 1;
  }
  ncclIbNetCommBase& base() { return comm_->base; }
  std::unique_ptr<ncclIbSendComm> comm_;
};

// weight_i is proportional to 1 / (rtt_i - minRtt + minRttSample), so with samples of 100/300 and a floor of
// 50 the weights are 1/50 : 1/250, i.e. 5/6 and 1/6, and the tokens round to 83 and 17.
TEST_F(NetIbCastSchedulerTxMicrotest, UpdateTx_WeightsAreInverseToAdjustedRtt) {
  Rtts(100.0, 300.0, 50);
  base().rrQpTxSched.activeTokens.qpTokens[0] = 90;
  base().rrQpTxSched.activeTokens.qpTokens[1] = 5;
  IbCastQpSchedUpdateTx(&base());
  EXPECT_TRUE(base().qpTxSchedInit);
  EXPECT_DOUBLE_EQ(base().qpTxSched[0].weight, 5.0 / 6.0);
  EXPECT_DOUBLE_EQ(base().qpTxSched[1].weight, 1.0 / 6.0);
  EXPECT_EQ(base().rrQpTxSched.initTokens.qpTokens[0], 83);
  EXPECT_EQ(base().rrQpTxSched.initTokens.qpTokens[1], 17);
  EXPECT_EQ(base().rrQpTxSched.initTokens.totTokens, NCCL_IB_TARGET_TOT_TOKENS);
  EXPECT_EQ(base().rrQpTxSched.activeTokens.qpTokens[0], 83);
  EXPECT_EQ(base().rrQpTxSched.activeTokens.qpTokens[1], 5);
  EXPECT_EQ(base().rrQpTxSched.activeTokens.totTokens, 88);
}

TEST_F(NetIbCastSchedulerTxMicrotest, UpdateTx_TinyWeight_StillGetsOneToken) {
  Rtts(1.0, 1e9, 1);
  IbCastQpSchedUpdateTx(&base());
  EXPECT_EQ(base().rrQpTxSched.initTokens.qpTokens[0], NCCL_IB_TARGET_TOT_TOKENS);
  EXPECT_EQ(base().rrQpTxSched.initTokens.qpTokens[1], 1);
  EXPECT_EQ(base().rrQpTxSched.initTokens.totTokens, NCCL_IB_TARGET_TOT_TOKENS + 1);
}

TEST_F(NetIbCastSchedulerTxMicrotest, UpdateTx_UnmeasuredQpOrZeroFloor_LeavesTheSchedulerUnarmed) {
  Rtts(100.0, 0.0, 50);
  IbCastQpSchedUpdateTx(&base());
  EXPECT_FALSE(base().qpTxSchedInit);
  EXPECT_EQ(base().rrQpTxSched.initTokens.totTokens, 0);

  Rtts(100.0, 100.0, 0);
  IbCastQpSchedUpdateTx(&base());
  EXPECT_FALSE(base().qpTxSchedInit);
}

TEST_F(NetIbCastSchedulerTxMicrotest, UpdateTx_NoQps_ArmsAnEmptyTable) {
  base().nqps = 0;
  IbCastQpSchedUpdateTx(&base());
  EXPECT_TRUE(base().qpTxSchedInit);
  EXPECT_EQ(base().rrQpTxSched.initTokens.totTokens, 0);
}

TEST_F(NetIbCastSchedulerTxMicrotest, UpdateTx_LogEnabled_TracksMinAndMaxWeights) {
  base().schedParms.logEnable = true;
  Rtts(100.0, 300.0, 50);
  IbCastQpSchedUpdateTx(&base());
  EXPECT_DOUBLE_EQ(base().qpTxSched[0].minWeight, 5.0 / 6.0);
  EXPECT_DOUBLE_EQ(base().qpTxSched[0].maxWeight, 5.0 / 6.0);
  Rtts(300.0, 100.0, 50);
  base().qpTxStats[1].minRttSample = 50;
  IbCastQpSchedUpdateTx(&base());
  EXPECT_DOUBLE_EQ(base().qpTxSched[0].minWeight, 1.0 / 6.0);
  EXPECT_DOUBLE_EQ(base().qpTxSched[0].maxWeight, 5.0 / 6.0);
  EXPECT_DOUBLE_EQ(base().qpTxSched[1].minWeight, 1.0 / 6.0);
  EXPECT_DOUBLE_EQ(base().qpTxSched[1].maxWeight, 5.0 / 6.0);
}

class NetIbCastSchedulerTxStatsMicrotest : public NetIbCastSchedulerTxMicrotest {
 protected:
  // 125 bytes at 1000 Mb/s take 1000 ns on the wire; the device is reached through qps[].devIndex and vProps.devs[].
  static constexpr uint64_t kWireNs = 1000;
  void SetUp() override {
    NetIbCastSchedulerTxMicrotest::SetUp();
    base().qps[1].devIndex = 1;
    base().vProps.devs[1] = 5;
    IbCastDevs[5].speed = 1000;
    remap_.qpIndex = 1;
    remap_.tx.bytes = 125;
    remap_.tx.startTimeNs = 1000;
  }
  void TearDown() override {
    IbCastDevs[5].speed = 0;
    NetIbCastSchedulerTxMicrotest::TearDown();
  }
  void SampleAt(uint64_t nowNs) {
    ScopedHook clock(g_clockGettime, [nowNs](clockid_t clk, timespec* ts) {
      EXPECT_EQ(clk, CLOCK_MONOTONIC);
      ts->tv_sec = nowNs / NSEC_PER_SEC;
      ts->tv_nsec = nowNs % NSEC_PER_SEC;
      return 0;
    });
    IbCastQpSchedUpdateTxStats(&remap_, &base());
  }
  ncclIbQpTxStats& stats() { return base().qpTxStats[1]; }
  ncclIbRemapWrId remap_{};
};

TEST_F(NetIbCastSchedulerTxStatsMicrotest, UpdateTxStats_FirstSample_IsRoundTripMinusWireTime) {
  SampleAt(remap_.tx.startTimeNs + 10000);
  EXPECT_EQ(stats().numMeasurements, 1u);
  EXPECT_EQ(stats().totRtt, 10000 - kWireNs);
  EXPECT_EQ(stats().minRttSample, 10000 - kWireNs);
  EXPECT_DOUBLE_EQ(stats().rtt, 10000.0 - kWireNs);
  EXPECT_EQ(base().qpTxStats[0].numMeasurements, 0u);
}

TEST_F(NetIbCastSchedulerTxStatsMicrotest, UpdateTxStats_LaterSamples_BlendByWeightNew) {
  remap_.parms.weightNew = 0.25;
  SampleAt(remap_.tx.startTimeNs + 9000);
  SampleAt(remap_.tx.startTimeNs + 5000);
  EXPECT_EQ(stats().numMeasurements, 2u);
  EXPECT_EQ(stats().minRttSample, 4000u);
  EXPECT_DOUBLE_EQ(stats().rtt, 8000.0 * 0.75 + 4000.0 * 0.25);
}

TEST_F(NetIbCastSchedulerTxStatsMicrotest, UpdateTxStats_NoWeight_AveragesEverySample) {
  SampleAt(remap_.tx.startTimeNs + 9000);
  SampleAt(remap_.tx.startTimeNs + 13000);
  EXPECT_DOUBLE_EQ(stats().rtt, (8000.0 + 12000.0) / 2);
  EXPECT_EQ(stats().minRttSample, 8000u);
}

TEST_F(NetIbCastSchedulerTxStatsMicrotest, UpdateTxStats_UntimedSampleOrClockFailure_IsDropped) {
  remap_.tx.startTimeNs = 0;
  SampleAt(5000);
  EXPECT_EQ(stats().numMeasurements, 0u);

  remap_.tx.startTimeNs = 1000;
  ScopedHook clock(g_clockGettime, [](clockid_t, timespec*) { return -1; });
  IbCastQpSchedUpdateTxStats(&remap_, &base());
  EXPECT_EQ(stats().numMeasurements, 0u);
}

TEST_F(NetIbCastSchedulerTxStatsMicrotest, UpdateTxStats_ResetRtt_ClearsHistoryOncePerRequest) {
  base().qpTxStats[0].totRtt = 77;
  base().qpTxStats[0].numMeasurements = 3;
  stats().totRtt = 99;
  stats().numMeasurements = 4;
  stats().rtt = 50.0;
  remap_.parms.resetRtt = true;
  remap_.parms.resetInterval = QP_SCHED_RESET_NEVER;
  SampleAt(remap_.tx.startTimeNs + 3000);
  EXPECT_TRUE(base().resetRttDone);
  EXPECT_EQ(base().qpTxStats[0].numMeasurements, 0u);
  EXPECT_EQ(base().qpTxStats[0].totRtt, 0u);
  EXPECT_EQ(stats().numMeasurements, 1u);
  EXPECT_DOUBLE_EQ(stats().rtt, 2000.0);
  EXPECT_EQ(base().nextQpTxStatsResetNs, remap_.tx.startTimeNs + 3000);

  SampleAt(remap_.tx.startTimeNs + 5000);
  EXPECT_EQ(stats().numMeasurements, 2u);
}

TEST_F(NetIbCastSchedulerTxStatsMicrotest, UpdateTxStats_ResetInterval_ClearsHistoryWhenItElapses) {
  remap_.parms.resetInterval = 100000;
  base().nextQpTxStatsResetNs = 50000;
  stats().numMeasurements = 4;
  stats().totRtt = 99;
  SampleAt(20000);
  EXPECT_EQ(stats().numMeasurements, 5u);
  SampleAt(50000);
  EXPECT_EQ(stats().numMeasurements, 1u);
  EXPECT_EQ(stats().totRtt, 49000 - kWireNs);
  EXPECT_EQ(base().nextQpTxStatsResetNs, 150000u);
}

class NetIbCastSchedulerNqpsMicrotest : public NetIbCastSchedulerTxMicrotest {
 protected:
  void SetUp() override {
    NetIbCastSchedulerTxMicrotest::SetUp();
    base().nqps = 3;
    base().qpIndex = 2;
    req_.base = &base();
    req_.nreqs = 1;
    req_.desc.parms.enable = true;
  }
  int Pick(int* start, bool* wrr) { return IbCastQpSchedGetEffectiveTxNqps(&req_, start, wrr); }
  ncclIbRequest req_{};
};

TEST_F(NetIbCastSchedulerNqpsMicrotest, EffectiveNqps_SchedulerOff_RotatesByRequestId) {
  req_.desc.parms.enable = false;
  req_.id = 7;
  int start = -1;
  bool wrr = true;
  EXPECT_EQ(Pick(&start, &wrr), 3);
  EXPECT_EQ(start, 7 % 3);
  EXPECT_FALSE(wrr);
}

TEST_F(NetIbCastSchedulerNqpsMicrotest, EffectiveNqps_SingleQpComm_UsesIt) {
  base().nqps = 1;
  base().qpIndex = 0;
  int start = -1;
  bool wrr = true;
  EXPECT_EQ(Pick(&start, &wrr), 1);
  EXPECT_EQ(start, 0);
  EXPECT_FALSE(wrr);
}

TEST_F(NetIbCastSchedulerNqpsMicrotest, EffectiveNqps_SplitData_SplitsOnlyFromTheMinimum) {
  req_.desc.parms.splitData = true;
  req_.desc.parms.splitDataMin = 1000;
  req_.send.size = 1500;
  req_.nreqs = 2;
  int start = -1;
  bool wrr = true;
  EXPECT_EQ(Pick(&start, &wrr), 3);
  EXPECT_EQ(start, 2);
  EXPECT_FALSE(wrr);

  req_.send.size = 1499;
  EXPECT_EQ(Pick(&start, &wrr), 1);
  EXPECT_EQ(start, 2);
  EXPECT_FALSE(wrr);
}

// Tokens {2,0,1}: the cursor moves on after every pick and skips the empty qp1, so qp0 and qp2 alternate; the
// table refills from initTokens once it runs dry (after the third pick).
TEST_F(NetIbCastSchedulerNqpsMicrotest, EffectiveNqps_Wrr_SpendsTokensRoundRobin) {
  req_.desc.parms.doWrr = true;
  const int tokens[] = {2, 0, 1};
  ASSERT_EQ(ncclIbCastSetTokens(comm_.get(), tokens, 3), ncclSuccess);
  std::vector<int> picks;
  for (int i = 0; i < 5; i++) {
    int start = -1;
    bool wrr = false;
    ASSERT_EQ(Pick(&start, &wrr), 1);
    EXPECT_TRUE(wrr);
    picks.push_back(start);
  }
  EXPECT_EQ(picks, (std::vector<int>{0, 2, 0, 2, 0}));
}

TEST_F(NetIbCastSchedulerNqpsMicrotest, EffectiveNqps_WrrBeforeFirstUpdate_StaysOnTheCommQp) {
  req_.desc.parms.doWrr = true;
  int start = -1;
  bool wrr = true;
  EXPECT_EQ(Pick(&start, &wrr), 1);
  EXPECT_EQ(start, 2);
  EXPECT_FALSE(wrr);
}

TEST_F(NetIbCastSchedulerTxMicrotest, Remap_HandsOutFreeSlotsUntilExhausted) {
  ncclIbRemapWrId* first = nullptr;
  ASSERT_EQ(IbCastQpSchedGetRemap(&base(), 0xab, 3, &first), ncclSuccess);
  EXPECT_EQ(first, &base().remapWrId[0]);
  EXPECT_EQ(first->origWrId, 0xabu);
  EXPECT_EQ(first->qpIndex, 3);
  EXPECT_EQ(first->state, NCCL_NET_IB_REMAP_USED);
  for (int i = 1; i < NET_IB_MAX_REQUESTS; i++) {
    ncclIbRemapWrId* r = nullptr;
    ASSERT_EQ(IbCastQpSchedGetRemap(&base(), i, 0, &r), ncclSuccess);
  }
  ncclIbRemapWrId* none = first;
  ncclResult_t res = ncclSuccess;
  const std::string log = CaptureLog([&] { res = IbCastQpSchedGetRemap(&base(), 1, 0, &none); });
  EXPECT_EQ(res, ncclInternalError);
  EXPECT_EQ(none, nullptr);
  EXPECT_TRUE(Has(log, "unable to allocate remap buffer")) << log;

  ASSERT_EQ(IbCastQpSchedFreeRemap(&base().remapWrId[5]), ncclSuccess);
  ASSERT_EQ(IbCastQpSchedGetRemap(&base(), 9, 1, &none), ncclSuccess);
  EXPECT_EQ(none, &base().remapWrId[5]);
}

TEST_F(NetIbCastSchedulerTxMicrotest, GetSchedState_CopiesTheSchedulerOut) {
  ncclIbCastSchedState out{};
  EXPECT_EQ(ncclIbCastGetSchedState(nullptr, &out), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetSchedState(comm_.get(), nullptr), ncclInvalidArgument);
  const int tokens[] = {4, 6};
  ASSERT_EQ(ncclIbCastSetTokens(comm_.get(), tokens, 2), ncclSuccess);
  base().rrQpTxSched.activeTokens.qpTokens[1] = 1;
  base().rrQpTxSched.activeTokens.totTokens = 5;
  base().rrQpTxSched.qpIndex = 1;
  ASSERT_EQ(ncclIbCastSetSchedParms(comm_.get(), true, true, true, 2048), ncclSuccess);
  ASSERT_EQ(ncclIbCastGetSchedState(comm_.get(), &out), ncclSuccess);
  EXPECT_EQ(out.nqps, 2);
  EXPECT_TRUE(out.schedInit);
  EXPECT_EQ(out.qpIndex, 1);
  EXPECT_EQ(out.initTotTokens, 10);
  EXPECT_EQ(out.activeTotTokens, 5);
  EXPECT_EQ(out.initQpTokens[0], 4);
  EXPECT_EQ(out.initQpTokens[1], 6);
  EXPECT_EQ(out.activeQpTokens[1], 1);
  EXPECT_TRUE(out.schedEnable && out.doWrr && out.splitData);
  EXPECT_EQ(out.splitDataMin, 2048u);

  ASSERT_EQ(ncclIbCastSetSchedParms(comm_.get(), false, false, false, 1), ncclSuccess);
  ASSERT_EQ(ncclIbCastGetSchedState(comm_.get(), &out), ncclSuccess);
  EXPECT_FALSE(out.schedEnable || out.doWrr || out.splitData);
  EXPECT_EQ(ncclIbCastSetSchedParms(nullptr, true, true, true, 1), ncclInvalidArgument);
}

TEST_F(NetIbCastSchedulerTxMicrotest, GetSchedState_ClampsTheCopyToTheArray) {
  base().nqps = NCCL_IB_MAX_QPS + 5;
  base().rrQpTxSched.initTokens.qpTokens[NCCL_IB_MAX_QPS - 1] = 42;
  auto out = std::make_unique<ncclIbCastSchedState>();
  ASSERT_EQ(ncclIbCastGetSchedState(comm_.get(), out.get()), ncclSuccess);
  EXPECT_EQ(out->nqps, NCCL_IB_MAX_QPS + 5);
  EXPECT_EQ(out->initQpTokens[NCCL_IB_MAX_QPS - 1], 42);
}

TEST_F(NetIbCastSchedulerTxMicrotest, SetTokens_ValidatesAndArmsTheTable) {
  const int tokens[] = {3, 1};
  EXPECT_EQ(ncclIbCastSetTokens(nullptr, tokens, 2), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastSetTokens(comm_.get(), nullptr, 2), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastSetTokens(comm_.get(), tokens, 0), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastSetTokens(comm_.get(), tokens, NCCL_IB_MAX_QPS + 1), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastSetTokens(comm_.get(), tokens, 1), ncclInvalidArgument);
  EXPECT_FALSE(base().qpTxSchedInit);

  base().rrQpTxSched.initTokens.qpTokens[2] = 9;
  base().rrQpTxSched.qpIndex = 1;
  ASSERT_EQ(ncclIbCastSetTokens(comm_.get(), tokens, 2), ncclSuccess);
  EXPECT_TRUE(base().qpTxSchedInit);
  EXPECT_EQ(base().rrQpTxSched.initTokens.totTokens, 4);
  EXPECT_EQ(base().rrQpTxSched.initTokens.qpTokens[2], 0);
  EXPECT_EQ(base().rrQpTxSched.activeTokens.qpTokens[0], 3);
  EXPECT_EQ(base().rrQpTxSched.activeTokens.totTokens, 4);
  EXPECT_EQ(base().rrQpTxSched.qpIndex, 0);

  base().nqps = 0;
  EXPECT_EQ(ncclIbCastSetTokens(comm_.get(), tokens, 1), ncclSuccess);
}

TEST_F(NetIbCastSchedulerTxMicrotest, GetOptRecvCompletion_ReportsTheFlag) {
  int out = -1;
  EXPECT_EQ(ncclIbCastGetOptRecvCompletion(nullptr, &out), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetOptRecvCompletion(comm_.get(), nullptr), ncclInvalidArgument);
  ASSERT_EQ(ncclIbCastGetOptRecvCompletion(comm_.get(), &out), ncclSuccess);
  EXPECT_EQ(out, 0);
  base().optRecvCompletion = true;
  ASSERT_EQ(ncclIbCastGetOptRecvCompletion(comm_.get(), &out), ncclSuccess);
  EXPECT_EQ(out, 1);
}

class NetIbCastSchedulerGrhMicrotest : public NetIbCastSchedulerTxMicrotest {
 protected:
  void TearDown() override {
    ibvSymbols.ibv_internal_query_qp = savedQueryQp_;
    NetIbCastSchedulerTxMicrotest::TearDown();
  }
  decltype(ibvSymbols.ibv_internal_query_qp) savedQueryQp_ = ibvSymbols.ibv_internal_query_qp;
  ibv_qp qps_[2]{};
};

// The second QP's query fails, so only the first reports the driver's is_global; a QP with no ibv_qp reports zeros.
TEST_F(NetIbCastSchedulerGrhMicrotest, GetGrhState_ReadsIsGlobalBackFromTheDriver) {
  ncclIbCastGrhState* null = nullptr;
  EXPECT_EQ(ncclIbCastGetGrhState(nullptr, null), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetGrhState(comm_.get(), nullptr), ncclInvalidArgument);
  base().nqps = 3;
  qps_[0].qp_num = 11;
  qps_[1].qp_num = 12;
  base().qps[0].qp = &qps_[0];
  base().qps[0].rtrAttr.linkLayer = IBV_LINK_LAYER_ETHERNET;
  base().qps[1].qp = &qps_[1];
  base().qps[1].rtrAttr.linkLayer = IBV_LINK_LAYER_INFINIBAND;
  base().qps[2].rtrAttr.linkLayer = IBV_LINK_LAYER_ETHERNET;
  for (int i = 0; i < 3; i++) base().activeQps[i] = &base().qps[i];
  ibvSymbols.ibv_internal_query_qp = [](ibv_qp* qp, ibv_qp_attr* attr, int mask, ibv_qp_init_attr*) {
    EXPECT_EQ(mask, IBV_QP_AV);
    if (qp->qp_num == 12) return EIO;
    attr->ah_attr.is_global = 1;
    return 0;
  };
  auto out = std::make_unique<ncclIbCastGrhState>();
  out->queryOk[2] = true;
  ASSERT_EQ(ncclIbCastGetGrhState(comm_.get(), out.get()), ncclSuccess);
  EXPECT_EQ(out->nqps, 3);
  EXPECT_EQ(out->linkLayer[0], IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(out->qpNum[0], 11u);
  EXPECT_EQ(out->isGlobal[0], 1);
  EXPECT_TRUE(out->queryOk[0]);
  EXPECT_EQ(out->linkLayer[1], IBV_LINK_LAYER_INFINIBAND);
  EXPECT_EQ(out->qpNum[1], 12u);
  EXPECT_FALSE(out->queryOk[1]);
  EXPECT_EQ(out->isGlobal[1], 0);
  EXPECT_EQ(out->linkLayer[2], IBV_LINK_LAYER_ETHERNET);
  EXPECT_EQ(out->qpNum[2], 0u);
  EXPECT_FALSE(out->queryOk[2]);

  base().activeQps[0] = nullptr;
  ASSERT_EQ(ncclIbCastGetGrhState(comm_.get(), out.get()), ncclSuccess);
  EXPECT_EQ(out->linkLayer[0], 0);
  EXPECT_EQ(out->qpNum[0], 0u);
}

TEST_F(NetIbCastSchedulerGrhMicrotest, GetGrhState_ClampsNqps) {
  auto out = std::make_unique<ncclIbCastGrhState>();
  base().nqps = -4;
  ASSERT_EQ(ncclIbCastGetGrhState(comm_.get(), out.get()), ncclSuccess);
  EXPECT_EQ(out->nqps, 0);
  base().nqps = NCCL_IB_MAX_QPS + 3;
  for (int i = 0; i < NCCL_IB_MAX_QPS; i++) base().activeQps[i] = nullptr;
  ASSERT_EQ(ncclIbCastGetGrhState(comm_.get(), out.get()), ncclSuccess);
  EXPECT_EQ(out->nqps, NCCL_IB_MAX_QPS);
}

TEST_F(NetIbCastSchedulerTxMicrotest, ResiliencyInspect_NeedsAContextAndCopiesItsState) {
  ncclIbCastResiliencyState out{};
  int reposts = -1;
  EXPECT_EQ(ncclIbCastGetResiliencyState(nullptr, &out), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetResiliencyState(comm_.get(), nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetResiliencyState(comm_.get(), &out), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetRepostCount(nullptr, &reposts), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetRepostCount(comm_.get(), nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclIbCastGetRepostCount(comm_.get(), &reposts), ncclInvalidArgument);

  auto res = std::make_unique<ncclIbResiliency>();
  res->recoveryEnabled = true;
  res->inProgress = true;
  res->outstandingRequests = 3;
  res->outstandingRecovery = 1;
  res->repostCount = 6;
  res->ndevs = 2;
  res->devs[1].state = ncclIbResiliencyDevStateRecovered;
  res->devs[1].recoveryCount = 4;
  base().resiliency = res.get();
  ASSERT_EQ(ncclIbCastGetResiliencyState(comm_.get(), &out), ncclSuccess);
  EXPECT_TRUE(out.recoveryEnabled && out.inProgress);
  EXPECT_EQ(out.outstandingRequests, 3);
  EXPECT_EQ(out.outstandingRecovery, 1);
  EXPECT_EQ(out.ndevs, 2);
  EXPECT_EQ(out.devState[0], ncclIbResiliencyDevStateOk);
  EXPECT_EQ(out.devState[1], ncclIbResiliencyDevStateRecovered);
  EXPECT_EQ(out.recoveryCount[1], 4);
  ASSERT_EQ(ncclIbCastGetRepostCount(comm_.get(), &reposts), ncclSuccess);
  EXPECT_EQ(reposts, 6);

  res->ndevs = NCCL_IB_MAX_DEVS_PER_NIC + 2;
  ASSERT_EQ(ncclIbCastGetResiliencyState(comm_.get(), &out), ncclSuccess);
  EXPECT_EQ(out.ndevs, NCCL_IB_MAX_DEVS_PER_NIC + 2);
  base().resiliency = nullptr;
}

}  // namespace
