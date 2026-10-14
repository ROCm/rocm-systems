/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net.cc, driven through a fake ncclNet_t.
// This TU #includes the hipified net.cc (NET_CC_PATH) so its file-static helpers
// are reachable. No GPU, no librccl.so, no network.

#include <gtest/gtest.h>

#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <string>

#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/rocmwrap_fakes.h"
#include "fakes/transport_p2p_fakes.h"

// Routes every NCCL_PARAM / RCCL_PARAM in net.cc through g_loadParam on each call.
#include "fakes/param_redirect.h"

#include NET_CC_PATH

// init.cc data symbols. allocTracker is updated by the alloc.h calloc templates; a
// non-null ncclGdrCopy selects net.cc's GDRCopy arms.
struct allocationTracker allocTracker[MAX_ALLOC_TRACK_NGPU] = {};
gdr_t ncclGdrCopy = nullptr;

namespace {

// ncclNet_t has no context argument on most slots, so the trampolines reach the
// fake through the instance the fixture installs.
struct FakeNet;
FakeNet* g_activeNet = nullptr;

struct FakeNet {
  int setNetAttrCalls = 0;
  void* lastAttrCtx = nullptr;
  ncclNetAttr_t lastAttr{};

  ncclResult_t setNetAttr(void* ctx, ncclNetAttr_t* attr) {
    ++setNetAttrCalls;
    lastAttrCtx = ctx;
    lastAttr = *attr;
    return ncclSuccess;
  }

  ncclNet_t vtable() {
    ncclNet_t v{};
    v.name = "FakeNet";
    v.setNetAttr = &FakeNet::TrampSetNetAttr;
    return v;
  }

 private:
  static ncclResult_t TrampSetNetAttr(void* ctx, ncclNetAttr_t* attr) { return g_activeNet->setNetAttr(ctx, attr); }
};

// Param values for the redirected NCCL_PARAM / RCCL_PARAM bodies; unset names keep their defaults.
std::function<int64_t(const char*, int64_t)> ParamOverride(std::initializer_list<std::pair<const char*, int64_t>> kv) {
  std::vector<std::pair<std::string, int64_t>> table(kv.begin(), kv.end());
  return [table](const char* env, int64_t deft) {
    for (const auto& [name, value] : table)
      if (name == env) return value;
    return deft;
  };
}

class NetMicrotestBase : public ::testing::Test {
 protected:
  FakeNet net_;
  ncclNet_t vtable_{};
  ncclProxyState proxyState_{};

  void SetUp() override {
    g_activeNet = &net_;
    vtable_ = net_.vtable();
    proxyState_.ncclNet = &vtable_;
    proxyState_.netContext = &net_;
    std::call_once(rcclDetectIBNicsOnce, [] {});  // rcclUseAinic() must never scan the host's sysfs
  }

  void TearDown() override {
    g_activeNet = nullptr;
    rcclPrimaryNicInfo = {rcclIBNicTypeUnknown, 0, 0};
    ncclGdrCopy = nullptr;
    ResetRocmWrapFakes();
    ResetNcclFakes();
    ResetHipFakes();
    ResetTransportP2pFakes();
    ResetEnvFakes();
  }
};

// ---------------------------------------------------------------------------
// IB NIC detection (rcclDetectIBNics, readIBNicRate, rcclUseAinic)
// ---------------------------------------------------------------------------

class NetIbNicDetectMicrotest : public NetMicrotestBase {
 protected:
  std::filesystem::path root_;
  const char* savedPath_ = sysfsIBPath;
  std::string rootStr_;

  void SetUp() override {
    NetMicrotestBase::SetUp();
    char tmpl[] = "/tmp/net-ib-sysfs-XXXXXX";
    ASSERT_NE(mkdtemp(tmpl), nullptr);
    root_ = tmpl;
    rootStr_ = root_.string();
    sysfsIBPath = rootStr_.c_str();
    rcclPrimaryNicInfo = {rcclIBNicTypeUnknown, 0, 0};
  }

  void TearDown() override {
    sysfsIBPath = savedPath_;
    std::filesystem::remove_all(root_);
    NetMicrotestBase::TearDown();
  }

  static void WriteFile(const std::filesystem::path& p, const std::string& text) {
    std::filesystem::create_directories(p.parent_path());
    std::ofstream(p) << text << "\n";
  }

  // One /sys/class/infiniband/<name> entry with the given PCI ids and per-port rate strings.
  void AddDevice(const std::string& name, const std::string& vendor, const std::string& device,
                 std::initializer_list<std::string> portRates = {}) {
    WriteFile(root_ / name / "device" / "vendor", vendor);
    WriteFile(root_ / name / "device" / "device", device);
    int port = 1;
    for (const auto& rate : portRates) WriteFile(root_ / name / "ports" / std::to_string(port++) / "rate", rate);
  }
};

TEST_F(NetIbNicDetectMicrotest, ReadIbNicRate_SeveralPorts_ReturnsHighestParsableRate) {
  AddDevice("mlx5_0", "0x15b3", "0x1021", {"100 Gb/sec (4X EDR)", "garbage", "400 Gb/sec (4X NDR)"});

  EXPECT_EQ(readIBNicRate("mlx5_0"), 400);
}

TEST_F(NetIbNicDetectMicrotest, ReadIbNicRate_NoPortsDir_ReturnsZero) {
  AddDevice("mlx5_0", "0x15b3", "0x1021");

  EXPECT_EQ(readIBNicRate("mlx5_0"), 0);
}

TEST_F(NetIbNicDetectMicrotest, Detect_NetEnvRocmIb_ForcesAinicWithoutScanning) {
  SetMicroEnv("NCCL_NET", "rocm-ib");
  sysfsIBPath = "/nonexistent/net-ib-sysfs";

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeAINIC);
}

TEST_F(NetIbNicDetectMicrotest, Detect_NoSysfsDir_ReportsDefault) {
  sysfsIBPath = "/nonexistent/net-ib-sysfs";

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeDefault);
}

TEST_F(NetIbNicDetectMicrotest, Detect_MlxMajority_ReportsMlxCountAndRate) {
  AddDevice("mlx5_0", "0x15b3", "0x1021", {"200 Gb/sec (4X HDR)"});
  AddDevice("mlx5_1", "0x15b3", "0x1021", {"200 Gb/sec (4X HDR)"});
  AddDevice("ionic_0", "0x1dd8", "0x1002", {"400 Gb/sec"});

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeMLX);
  EXPECT_EQ(rcclPrimaryNicInfo.count, 2);
  EXPECT_EQ(rcclPrimaryNicInfo.rate, 200);
}

TEST_F(NetIbNicDetectMicrotest, Detect_AinicDeviceIdPrefix_ReportsAinic) {
  AddDevice("ionic_0", "0x1dd8", "0x1002", {"400 Gb/sec"});

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeAINIC);
  EXPECT_EQ(rcclPrimaryNicInfo.rate, 400);
}

TEST_F(NetIbNicDetectMicrotest, Detect_Bnxt2Device_ReportsBnxt2) {
  AddDevice("bnxt_re0", "0x14e4", "0x1760");

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeBNXT2);
}

TEST_F(NetIbNicDetectMicrotest, Detect_UnknownVendors_ReportsDefaultWithCount) {
  AddDevice("rxe0", "0x8086", "0x1592");
  AddDevice("rxe1", "0x8086", "0x1592");

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeDefault);
  EXPECT_EQ(rcclPrimaryNicInfo.count, 2);
}

TEST_F(NetIbNicDetectMicrotest, Detect_MissingPciIds_IgnoresDevice) {
  AddDevice("mlx5_0", "0x15b3", "0x1021");
  std::filesystem::create_directories(root_ / "broken0" / "device");

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeMLX);
  EXPECT_EQ(rcclPrimaryNicInfo.count, 1);
}

TEST_F(NetIbNicDetectMicrotest, Detect_AinicRoceDisabled_CountsAinicAsDefault) {
  ScopedHook param(g_loadParam, ParamOverride({{"RCCL_AINIC_ROCE", 0}}));
  AddDevice("ionic_0", "0x1dd8", "0x1002");
  AddDevice("ionic_1", "0x1dd8", "0x1002");
  AddDevice("mlx5_0", "0x15b3", "0x1021");

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeDefault);
  EXPECT_EQ(rcclPrimaryNicInfo.count, 2);
}

TEST_F(NetIbNicDetectMicrotest, Detect_AinicRoceEnforced_CountsOtherVendorsAsDefault) {
  ScopedHook param(g_loadParam, ParamOverride({{"RCCL_AINIC_ROCE", 1}}));
  AddDevice("mlx5_0", "0x15b3", "0x1021");
  AddDevice("mlx5_1", "0x15b3", "0x1021");
  AddDevice("ionic_0", "0x1dd8", "0x1002");

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeDefault);
  EXPECT_EQ(rcclPrimaryNicInfo.count, 2);
}

TEST_F(NetIbNicDetectMicrotest, Detect_IbHcaList_CountsOnlyListedDevices) {
  SetMicroEnv("NCCL_IB_HCA", "=ionic_0");
  AddDevice("mlx5_0", "0x15b3", "0x1021");
  AddDevice("mlx5_1", "0x15b3", "0x1021");
  AddDevice("ionic_0", "0x1dd8", "0x1002");

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeAINIC);
  EXPECT_EQ(rcclPrimaryNicInfo.count, 1);
}

TEST_F(NetIbNicDetectMicrotest, Detect_IbHcaExclusion_SkipsListedDevices) {
  SetMicroEnv("NCCL_IB_HCA", "^mlx5");
  AddDevice("mlx5_0", "0x15b3", "0x1021");
  AddDevice("mlx5_1", "0x15b3", "0x1021");
  AddDevice("ionic_0", "0x1dd8", "0x1002");

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeAINIC);
  EXPECT_EQ(rcclPrimaryNicInfo.count, 1);
}

TEST_F(NetIbNicDetectMicrotest, PrimaryNicQueries_DetectionDone_ReturnCachedInfo) {
  rcclPrimaryNicInfo = {rcclIBNicTypeAINIC, 400, 2};
  EXPECT_TRUE(rcclUseAinic());
  EXPECT_EQ(rcclPrimaryNic().count, 2);

  rcclPrimaryNicInfo = {rcclIBNicTypeMLX, 200, 1};
  EXPECT_FALSE(rcclUseAinic());
}

// ---------------------------------------------------------------------------
// Network attribute hints (populateCommNetAttrs, setNetAttrs, setXferNetAttrs)
// ---------------------------------------------------------------------------

class NetAttrsMicrotest : public NetMicrotestBase {
 protected:
  ncclComm comm_{};
  ncclConnector conn_{};
  ncclProxyArgs args_{};
};

TEST_F(NetAttrsMicrotest, PopulateCommAttrs_P2pConnector_CapsPeersByP2pChannels) {
  conn_.p2pOnly = 1;
  comm_.nRanks = 64;
  comm_.p2pnChannels = 4;
  comm_.p2pnChannelsPerPeer = 2;
  ncclNetAttr_t attr;

  populateCommNetAttrs(&comm_, &conn_, &attr);

  EXPECT_EQ(attr.sendCommAttr.maxConcurrentPeers, 4 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
  EXPECT_EQ(attr.recvCommAttr.maxConcurrentPeers, 4 * NCCL_MAX_DEV_WORK_P2P_PER_BATCH);
  EXPECT_EQ(attr.sendCommAttr.maxFlowsPerPeer, 2);
  EXPECT_EQ(attr.recvCommAttr.maxFlowsPerPeer, 2);
  EXPECT_EQ(attr.sendCommAttr.minConcurrentPeers, 1);
  EXPECT_EQ(attr.recvCommAttr.minFlowsPerPeer, 1);
  EXPECT_EQ(attr.op, BIT(ncclFuncSend) | BIT(ncclFuncRecv) | BIT(ncclFuncAlltoAll) | BIT(ncclFuncScatter) |
                       BIT(ncclFuncGather));
  EXPECT_EQ(attr.algo, (uint32_t)NCCL_NET_ATTR_UNDEF);
}

TEST_F(NetAttrsMicrotest, PopulateCommAttrs_FewRanks_CapsPeersByRankCount) {
  conn_.p2pOnly = 1;
  comm_.nRanks = 3;
  comm_.p2pnChannels = 4;
  ncclNetAttr_t attr;

  populateCommNetAttrs(&comm_, &conn_, &attr);

  EXPECT_EQ(attr.sendCommAttr.maxConcurrentPeers, 3);
}

TEST_F(NetAttrsMicrotest, PopulateCommAttrs_CollectiveConnector_UsesTreeFanoutAndChannels) {
  comm_.nRanks = 64;
  comm_.nChannels = 8;
  ncclNetAttr_t attr;

  populateCommNetAttrs(&comm_, &conn_, &attr);

  EXPECT_EQ(attr.sendCommAttr.maxConcurrentPeers, (NCCL_MAX_TREE_ARITY - 1) * 2);
  EXPECT_EQ(attr.recvCommAttr.maxConcurrentPeers, (NCCL_MAX_TREE_ARITY - 1) * 2);
  EXPECT_EQ(attr.sendCommAttr.maxFlowsPerPeer, 8);
  EXPECT_EQ(attr.recvCommAttr.maxFlowsPerPeer, 8);
  EXPECT_EQ(attr.op, (uint32_t)NCCL_NET_ATTR_UNDEF);
}

TEST_F(NetAttrsMicrotest, PopulateCommAttrs_CollectiveFewRanks_CapsPeersByRankCount) {
  comm_.nRanks = 2;
  ncclNetAttr_t attr;

  populateCommNetAttrs(&comm_, &conn_, &attr);

  EXPECT_EQ(attr.recvCommAttr.maxConcurrentPeers, 2);
}

TEST_F(NetAttrsMicrotest, SetNetAttrs_PluginSupportsHints_ForwardsAndCaches) {
  ncclNetAttr_t attr = NCCL_NET_ATTR_INIT;
  attr.op = BIT(ncclFuncAllReduce);

  setNetAttrs(&proxyState_, &attr);

  EXPECT_EQ(net_.setNetAttrCalls, 1);
  EXPECT_EQ(net_.lastAttrCtx, &net_);
  EXPECT_EQ(net_.lastAttr.op, BIT(ncclFuncAllReduce));
  EXPECT_EQ(proxyState_.netAttr.op, BIT(ncclFuncAllReduce));
}

TEST_F(NetAttrsMicrotest, SetNetAttrs_PluginWithoutHints_LeavesCacheUntouched) {
  vtable_.setNetAttr = nullptr;
  proxyState_.netAttr.op = 7;
  ncclNetAttr_t attr = NCCL_NET_ATTR_INIT;

  setNetAttrs(&proxyState_, &attr);

  EXPECT_EQ(proxyState_.netAttr.op, 7u);
}

TEST_F(NetAttrsMicrotest, SetXferAttrs_SendCollective_PinsSendPeersFlowsAndAlgo) {
  proxyState_.netAttr = NCCL_NET_ATTR_INIT;
  args_.nPeers = 2;
  args_.nChannels = 4;
  args_.collAPI = ncclFuncAllReduce;
  args_.algorithm = NCCL_ALGO_RING;
  args_.protocol = NCCL_PROTO_SIMPLE;

  setXferNetAttrs(&proxyState_, &args_, /*send=*/1);

  ASSERT_EQ(net_.setNetAttrCalls, 1);
  const ncclNetAttr_t& a = net_.lastAttr;
  EXPECT_EQ(a.sendCommAttr.minConcurrentPeers, 2);
  EXPECT_EQ(a.sendCommAttr.maxConcurrentPeers, 2);
  EXPECT_EQ(a.sendCommAttr.minFlowsPerPeer, 4);
  EXPECT_EQ(a.sendCommAttr.maxFlowsPerPeer, 4);
  EXPECT_EQ(a.recvCommAttr.maxConcurrentPeers, NCCL_NET_ATTR_UNDEF);
  EXPECT_EQ(a.op, BIT(ncclFuncAllReduce));
  EXPECT_EQ(a.algo, BIT(NCCL_ALGO_RING));
  EXPECT_EQ(a.proto, BIT(NCCL_PROTO_SIMPLE));
}

TEST_F(NetAttrsMicrotest, SetXferAttrs_RecvP2p_PinsRecvPeersAndLeavesAlgoUnset) {
  proxyState_.netAttr = NCCL_NET_ATTR_INIT;
  args_.nPeers = 3;
  args_.nChannels = 1;
  args_.collAPI = ncclFuncRecv;

  setXferNetAttrs(&proxyState_, &args_, /*send=*/0);

  ASSERT_EQ(net_.setNetAttrCalls, 1);
  const ncclNetAttr_t& a = net_.lastAttr;
  EXPECT_EQ(a.recvCommAttr.maxConcurrentPeers, 3);
  EXPECT_EQ(a.recvCommAttr.minFlowsPerPeer, 1);
  EXPECT_EQ(a.sendCommAttr.maxConcurrentPeers, NCCL_NET_ATTR_UNDEF);
  EXPECT_EQ(a.op, BIT(ncclFuncRecv));
  EXPECT_EQ(a.algo, (uint32_t)NCCL_NET_ATTR_UNDEF);
}

TEST_F(NetAttrsMicrotest, SetXferAttrs_UnchangedHints_SkipsPlugin) {
  proxyState_.netAttr = NCCL_NET_ATTR_INIT;
  args_.nPeers = 2;
  args_.nChannels = 4;
  args_.collAPI = ncclFuncSend;
  setXferNetAttrs(&proxyState_, &args_, 1);
  ASSERT_EQ(net_.setNetAttrCalls, 1);

  setXferNetAttrs(&proxyState_, &args_, 1);

  EXPECT_EQ(net_.setNetAttrCalls, 1);
}

// ---------------------------------------------------------------------------
// Small helpers (canConnect, ncclTopoFlushTypeStr, ncclNetGetDeviceHandle)
// ---------------------------------------------------------------------------

TEST(NetHelpersMicrotest, CanConnect_AnyPeers_ReportsConnectable) {
  ncclPeerInfo a{}, b{};
  a.hostHash = b.hostHash = 42;
  int ret = 0;

  EXPECT_EQ(canConnect(&ret, nullptr, nullptr, &a, &b), ncclSuccess);
  EXPECT_EQ(ret, 1);
}

TEST(NetHelpersMicrotest, FlushTypeStr_NamesEachFlushType) {
  EXPECT_STREQ(ncclTopoFlushTypeStr(ncclTopoFlushNone), "None");
  EXPECT_STREQ(ncclTopoFlushTypeStr(ncclTopoFlushAlways), "Always");
  EXPECT_STREQ(ncclTopoFlushTypeStr(ncclTopoFlushC2c), "C2c");
  EXPECT_STREQ(ncclTopoFlushTypeStr((ncclTopoFlushType)99), "Unknown");
}

TEST(NetHelpersMicrotest, GetDeviceHandle_UnpackRecv_AllocatesTypedHandleOnce) {
  ncclNetDeviceHandle_t* handle = nullptr;

  ASSERT_EQ(ncclNetGetDeviceHandle(NCCL_NET_DEVICE_UNPACK, NCCL_NET_DEVICE_UNPACK_VERSION, true, &handle), ncclSuccess);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->netDeviceType, NCCL_NET_DEVICE_UNPACK);
  EXPECT_EQ(handle->netDeviceVersion, NCCL_NET_DEVICE_UNPACK_VERSION);

  ncclNetDeviceHandle_t* first = handle;
  ASSERT_EQ(ncclNetGetDeviceHandle(NCCL_NET_DEVICE_UNPACK, NCCL_NET_DEVICE_UNPACK_VERSION, true, &handle), ncclSuccess);
  EXPECT_EQ(handle, first);
  free(handle);
}

TEST(NetHelpersMicrotest, GetDeviceHandle_NoOffloadNeeded_ClearsHandle) {
  ncclNetDeviceHandle_t stale{};
  ncclNetDeviceHandle_t* handle = &stale;
  EXPECT_EQ(ncclNetGetDeviceHandle(NCCL_NET_DEVICE_UNPACK, NCCL_NET_DEVICE_UNPACK_VERSION, false, &handle),
            ncclSuccess);
  EXPECT_EQ(handle, nullptr);

  handle = &stale;
  EXPECT_EQ(ncclNetGetDeviceHandle(NCCL_NET_DEVICE_UNPACK, NCCL_NET_DEVICE_UNPACK_VERSION + 1, true, &handle),
            ncclSuccess);
  EXPECT_EQ(handle, nullptr);

  handle = &stale;
  EXPECT_EQ(ncclNetGetDeviceHandle(NCCL_NET_DEVICE_HOST, 0, true, &handle), ncclSuccess);
  EXPECT_EQ(handle, nullptr);
}

}  // namespace
