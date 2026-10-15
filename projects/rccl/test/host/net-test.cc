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
#include <functional>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/rocmwrap_fakes.h"
#include "fakes/topo_net_stubs.h"
#include "fakes/topo_stubs.h"
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
  ncclNetProperties_t props = DefaultProps();
  ncclResult_t propsResult = ncclSuccess;
  int lastPropsDev = -1;

  ncclResult_t listenResult = ncclSuccess;
  int listenCalls = 0;
  int lastListenDev = -1;
  void* lastListenCtx = nullptr;
  char listenComm = 0;  // its address is the listen comm handed out

  int setNetAttrCalls = 0;
  void* lastAttrCtx = nullptr;
  ncclNetAttr_t lastAttr{};

  static ncclNetProperties_t DefaultProps() {
    ncclNetProperties_t p{};
    p.ptrSupport = NCCL_PTR_HOST | NCCL_PTR_CUDA;
    p.maxRecvs = 1;
    p.maxP2pBytes = MAX_NET_SIZE;
    return p;
  }

  ncclResult_t getProperties(int dev, ncclNetProperties_t* out) {
    lastPropsDev = dev;
    *out = props;
    return propsResult;
  }

  ncclResult_t listen(void* ctx, int dev, void* handle, void** comm) {
    ++listenCalls;
    lastListenCtx = ctx;
    lastListenDev = dev;
    if (listenResult != ncclSuccess) return listenResult;
    memset(handle, 0x5A, NCCL_NET_HANDLE_MAXSIZE);
    *comm = &listenComm;
    return ncclSuccess;
  }

  ncclResult_t setNetAttr(void* ctx, ncclNetAttr_t* attr) {
    ++setNetAttrCalls;
    lastAttrCtx = ctx;
    lastAttr = *attr;
    return ncclSuccess;
  }

  ncclNet_t vtable() {
    ncclNet_t v{};
    v.name = "FakeNet";
    v.getProperties = &FakeNet::TrampGetProperties;
    v.listen = &FakeNet::TrampListen;
    v.setNetAttr = &FakeNet::TrampSetNetAttr;
    return v;
  }

 private:
  static ncclResult_t TrampGetProperties(int dev, ncclNetProperties_t* props) {
    return g_activeNet->getProperties(dev, props);
  }
  static ncclResult_t TrampListen(void* ctx, int dev, void* handle, void** comm) {
    return g_activeNet->listen(ctx, dev, handle, comm);
  }
  static ncclResult_t TrampSetNetAttr(void* ctx, ncclNetAttr_t* attr) { return g_activeNet->setNetAttr(ctx, attr); }
};

// Param values for the redirected NCCL_PARAM / RCCL_PARAM bodies; unset names keep their defaults.
std::function<int64_t(const char*, int64_t)> ParamOverride(std::initializer_list<std::pair<const char*, int64_t>> kv) {
  std::vector<std::pair<std::string, int64_t>> table(kv.begin(), kv.end());
  return [table](const char* env, int64_t deft) {
    for (const auto& [name, value] : table) {
      if (name == env) return value;
    }
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
    // A healthy device for alloc.h's calloc/free paths: capture-mode swaps, side streams,
    // memsets and syncs succeed, and every allocation is its own base address.
    g_hipAsyncOpsResult = hipSuccess;
    g_hipDeviceGetPCIBusIdResult = hipSuccess;
    g_hipStreamCreateResult = hipSuccess;
    g_hipStreamSynchronize = [](hipStream_t) { return hipSuccess; };
    g_hipMemGetAddressRange = [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      *base = ptr;
      *size = 0;
      return hipSuccess;
    };
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
    ResetTopoStubs();
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
  AddDevice("mlx5_0", "0x15b3", "0x1021", {"100 Gb/sec (4X EDR)", "400 Gb/sec (4X NDR)", "garbage", "200 Gb/sec"});

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
  WriteFile(root_ / "broken1" / "device" / "vendor", "0x15b3");  // device id missing

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeMLX);
  EXPECT_EQ(rcclPrimaryNicInfo.count, 1);
}

TEST_F(NetIbNicDetectMicrotest, Detect_KnownVendorUnknownDevice_CountsAsDefault) {
  AddDevice("ionic_0", "0x1dd8", "0x2001");
  AddDevice("bnxt_re0", "0x14e4", "0x1751");

  rcclDetectIBNics();

  EXPECT_EQ(rcclPrimaryNicInfo.type, rcclIBNicTypeDefault);
  EXPECT_EQ(rcclPrimaryNicInfo.count, 2);
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

// ---------------------------------------------------------------------------
// Shared p2p buffer pool (sharedNetBuffersInit / Get / Destroy, proxySharedInit)
// ---------------------------------------------------------------------------

class NetSharedBuffersMicrotest : public NetMicrotestBase {
 protected:
  static constexpr int kLocalRanks = 4;
  static constexpr int kChunk = 1 << 12;

  // Every device, host and shareable buffer goes through this ledger, so TearDown
  // releases exactly what production left allocated.
  std::set<void*> live_;
  int hostAllocs_ = 0, devAllocs_ = 0, shareableAllocs_ = 0;
  int hostFrees_ = 0, devFrees_ = 0;

  void* Alloc(size_t n) {
    void* p = calloc(1, n);
    live_.insert(p);
    return p;
  }
  void Release(void* p) {
    if (live_.erase(p)) free(p);
  }

  void SetUp() override {
    NetMicrotestBase::SetUp();
    proxyState_.tpLocalnRanks = kLocalRanks;
    proxyState_.p2pChunkSize = kChunk;
    g_hipHostMalloc = [this](void** p, size_t n, unsigned) {
      ++hostAllocs_;
      *p = Alloc(n);
      return hipSuccess;
    };
    g_hipExtMallocWithFlags = [this](void** p, size_t n, unsigned) {
      ++devAllocs_;
      *p = Alloc(n);
      return hipSuccess;
    };
    g_ncclP2pAllocateShareableBuffer = [this](size_t n, int, ncclIpcDesc* desc, void** p, int, ncclMemManager*,
                                              ncclMemType_t) {
      ++shareableAllocs_;
      *p = Alloc(n);
      memset(desc, 0xAB, sizeof(*desc));
      return ncclSuccess;
    };
    g_hipHostFree = [this](void* p) {
      ++hostFrees_;
      Release(p);
      return hipSuccess;
    };
    g_hipFree = [this](void* p) {
      ++devFrees_;
      Release(p);
      return hipSuccess;
    };
  }

  void TearDown() override {
    if (ncclProxyPeer** peers = proxyState_.progressState.localPeers) {
      for (int r = 0; r < kLocalRanks; r++) free(peers[r]);
      free(peers);
    }
    for (void* p : live_) free(p);
    NetMicrotestBase::TearDown();
  }

  ncclProxySharedP2p& State(int localRank, int type) {
    ncclProxyPeer* peer = proxyState_.progressState.localPeers[localRank];
    return type == 0 ? peer->send : peer->recv;
  }

  ncclResult_t Init(int cuda, int localRank, int type, int sameProcess, int nChannels, char** gpu = nullptr,
                    char** cpu = nullptr, ssize_t* size = nullptr, ncclIpcDesc* ipc = nullptr) {
    return sharedNetBuffersInit(&proxyState_, cuda, localRank, type, sameProcess, nChannels, gpu, cpu, size, ipc);
  }

  ncclResult_t Destroy(int localRank, int type, int sameProcess = 1) {
    ncclProxyConnection conn{};
    conn.sameProcess = sameProcess;
    return sharedNetBuffersDestroy(&proxyState_, localRank, type, &conn);
  }
};

TEST_F(NetSharedBuffersMicrotest, Get_ChannelAndSlot_ReturnsChunkAlignedOffset) {
  ssize_t offset = -1;
  size_t size = 0;

  ASSERT_EQ(sharedBuffersGet(&proxyState_, /*channel=*/2, /*slot=*/5, &offset, &size), ncclSuccess);

  EXPECT_EQ(offset, (ssize_t)kChunk * (2 * NCCL_SHARED_STEPS + 5));
  EXPECT_EQ(size, (size_t)kChunk);
  EXPECT_EQ(sharedBuffersGet(&proxyState_, 0, 1, &offset, nullptr), ncclSuccess);
  EXPECT_EQ(offset, kChunk);
}

TEST_F(NetSharedBuffersMicrotest, Init_HostBufferAcrossProcesses_Rejected) {
  EXPECT_EQ(Init(/*cuda=*/0, 0, 0, /*sameProcess=*/0, 1), ncclInternalError);
}

TEST_F(NetSharedBuffersMicrotest, Init_LocalRankOutOfRange_RejectedWithoutAllocating) {
  EXPECT_EQ(Init(0, kLocalRanks, 0, 1, 1), ncclInvalidArgument);
  EXPECT_EQ(Init(0, -1, 0, 1, 1), ncclInvalidArgument);
  EXPECT_EQ(proxyState_.progressState.localPeers, nullptr);
}

TEST_F(NetSharedBuffersMicrotest, Init_HostBufferSameProcess_SizesPoolAndSharesPointer) {
  char *gpu = nullptr, *cpu = nullptr;
  ssize_t size = 0;

  ASSERT_EQ(Init(0, 1, 0, 1, /*nChannels=*/2, &gpu, &cpu, &size), ncclSuccess);

  EXPECT_EQ(size, (ssize_t)2 * NCCL_SHARED_STEPS * kChunk);
  EXPECT_EQ(hostAllocs_, 1);
  ASSERT_NE(cpu, nullptr);
  EXPECT_EQ(cpu, State(1, 0).hostBuff);
  EXPECT_EQ(gpu, cpu);
  EXPECT_EQ(State(1, 0).refcount, 1);
}

TEST_F(NetSharedBuffersMicrotest, Init_SecondUser_ReusesPoolAndCountsReference) {
  char* first = nullptr;
  ASSERT_EQ(Init(0, 0, 1, 1, 2, nullptr, &first), ncclSuccess);
  char* second = nullptr;
  ssize_t size = 0;

  ASSERT_EQ(Init(0, 0, 1, 1, /*nChannels=*/8, nullptr, &second, &size), ncclSuccess);

  EXPECT_EQ(second, first);
  EXPECT_EQ(hostAllocs_, 1);
  EXPECT_EQ(size, (ssize_t)2 * NCCL_SHARED_STEPS * kChunk);
  EXPECT_EQ(State(0, 1).refcount, 2);
  EXPECT_EQ(State(0, 0).refcount, 0);
}

TEST_F(NetSharedBuffersMicrotest, Init_DeviceBufferCrossProcess_AllocatesShareableAndExportsIpc) {
  char *gpu = reinterpret_cast<char*>(0x1), *cpu = nullptr;
  ncclIpcDesc ipc{};

  ASSERT_EQ(Init(/*cuda=*/1, 2, 0, /*sameProcess=*/0, 1, &gpu, &cpu, nullptr, &ipc), ncclSuccess);

  EXPECT_EQ(shareableAllocs_, 1);
  EXPECT_EQ(cpu, State(2, 0).cudaBuff);
  EXPECT_EQ(gpu, nullptr);
  ncclIpcDesc expected;
  memset(&expected, 0xAB, sizeof(expected));
  EXPECT_EQ(memcmp(&ipc, &expected, sizeof(ipc)), 0);
}

TEST_F(NetSharedBuffersMicrotest, Init_DeviceBufferSameProcess_AllocatesPlainDeviceMemory) {
  char* cpu = nullptr;

  ASSERT_EQ(Init(1, 0, 0, 1, 1, nullptr, &cpu), ncclSuccess);

  EXPECT_EQ(devAllocs_, 1);
  EXPECT_EQ(shareableAllocs_, 0);
  EXPECT_EQ(cpu, State(0, 0).cudaBuff);
}

TEST_F(NetSharedBuffersMicrotest, Init_DeviceBufferWithCuMem_AllocatesShareable) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });

  ASSERT_EQ(Init(1, 0, 0, 1, 1), ncclSuccess);

  EXPECT_EQ(shareableAllocs_, 1);
  EXPECT_EQ(devAllocs_, 0);
}

TEST_F(NetSharedBuffersMicrotest, Init_AllocationFails_ReturnsError) {
  ScopedHook alloc(g_ncclP2pAllocateShareableBuffer,
                   [](size_t, int, ncclIpcDesc*, void**, int, ncclMemManager*, ncclMemType_t) {
                     return ncclSystemError;
                   });

  EXPECT_EQ(Init(1, 0, 0, 0, 1), ncclSystemError);
}

TEST_F(NetSharedBuffersMicrotest, Destroy_InvalidPoolState_Rejected) {
  EXPECT_EQ(Destroy(kLocalRanks, 0), ncclInvalidArgument);
  EXPECT_EQ(Destroy(-1, 0), ncclInvalidArgument);
  EXPECT_EQ(Destroy(0, 0), ncclInternalError);  // no pool yet

  ASSERT_EQ(Init(0, 0, 0, 1, 1), ncclSuccess);
  EXPECT_EQ(Destroy(1, 0), ncclInternalError);  // no peer
  EXPECT_EQ(Destroy(0, 1), ncclInternalError);  // recv side never sized
}

TEST_F(NetSharedBuffersMicrotest, Destroy_OtherUsersRemain_KeepsBuffer) {
  ASSERT_EQ(Init(0, 0, 0, 1, 1), ncclSuccess);
  ASSERT_EQ(Init(0, 0, 0, 1, 1), ncclSuccess);

  ASSERT_EQ(Destroy(0, 0), ncclSuccess);

  EXPECT_EQ(hostFrees_, 0);
  EXPECT_EQ(State(0, 0).refcount, 1);
}

TEST_F(NetSharedBuffersMicrotest, Destroy_LastUserOfLastPeer_FreesBufferPeerAndTable) {
  ASSERT_EQ(Init(0, 0, 0, 1, 1), ncclSuccess);

  ASSERT_EQ(Destroy(0, 0), ncclSuccess);

  EXPECT_EQ(hostFrees_, 1);
  EXPECT_TRUE(live_.empty());
  EXPECT_EQ(proxyState_.progressState.localPeers, nullptr);
}

TEST_F(NetSharedBuffersMicrotest, Destroy_OtherDirectionStillActive_KeepsPeer) {
  ASSERT_EQ(Init(0, 0, 0, 1, 1), ncclSuccess);
  ASSERT_EQ(Init(0, 0, 1, 1, 1), ncclSuccess);

  ASSERT_EQ(Destroy(0, 0), ncclSuccess);

  ASSERT_NE(proxyState_.progressState.localPeers, nullptr);
  ASSERT_NE(proxyState_.progressState.localPeers[0], nullptr);
  EXPECT_EQ(State(0, 1).refcount, 1);
}

TEST_F(NetSharedBuffersMicrotest, Destroy_OtherPeerStillActive_KeepsTable) {
  ASSERT_EQ(Init(0, 0, 0, 1, 1), ncclSuccess);
  ASSERT_EQ(Init(0, 3, 1, 1, 1), ncclSuccess);

  ASSERT_EQ(Destroy(0, 0), ncclSuccess);

  ASSERT_NE(proxyState_.progressState.localPeers, nullptr);
  EXPECT_EQ(proxyState_.progressState.localPeers[0], nullptr);
  EXPECT_NE(proxyState_.progressState.localPeers[3], nullptr);
}

TEST_F(NetSharedBuffersMicrotest, Destroy_DeviceBufferCrossProcess_ReleasesIpcAndMemory) {
  ASSERT_EQ(Init(1, 0, 0, /*sameProcess=*/0, 1), ncclSuccess);
  ScopedHook ipcFree(g_ncclP2pFreeShareableBuffer, [](ncclIpcDesc*) { return ncclSuccess; });

  ASSERT_EQ(Destroy(0, 0, /*sameProcess=*/0), ncclSuccess);

  EXPECT_EQ(ipcFree.calls, 1);
  EXPECT_EQ(devFrees_, 1);
  EXPECT_TRUE(live_.empty());
}

TEST_F(NetSharedBuffersMicrotest, Destroy_DeviceBufferSameProcess_FreesWithoutIpcRelease) {
  ASSERT_EQ(Init(1, 0, 0, 1, 1), ncclSuccess);
  ScopedHook ipcFree(g_ncclP2pFreeShareableBuffer, [](ncclIpcDesc*) { return ncclSuccess; });

  ASSERT_EQ(Destroy(0, 0, /*sameProcess=*/1), ncclSuccess);

  EXPECT_EQ(ipcFree.calls, 0);
  EXPECT_EQ(devFrees_, 1);
  EXPECT_TRUE(live_.empty());
}

TEST_F(NetSharedBuffersMicrotest, Destroy_DeviceBufferWithCuMem_ReleasesIpc) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ASSERT_EQ(Init(1, 0, 0, 1, 1), ncclSuccess);
  ScopedHook ipcFree(g_ncclP2pFreeShareableBuffer, [](ncclIpcDesc*) { return ncclSuccess; });
  ScopedHook retain(g_hipMemRetainAllocationHandle, [](hipMemGenericAllocationHandle_t*, void*) { return hipSuccess; });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });
  ScopedHook unmap(g_hipMemUnmap, [](void*, size_t) { return hipSuccess; });
  ScopedHook addrFree(g_hipMemAddressFree, [this](void* p, size_t) {
    Release(p);
    return hipSuccess;
  });

  ASSERT_EQ(Destroy(0, 0, /*sameProcess=*/1), ncclSuccess);

  EXPECT_EQ(ipcFree.calls, 1);
  EXPECT_TRUE(live_.empty());
}

// proxySharedInit takes host memory under WSL (/dev/dxg present) and device memory otherwise.
TEST_F(NetSharedBuffersMicrotest, ProxySharedInit_ReservesSendPoolForConnectionRank) {
  ncclProxyConnection conn{};
  conn.tpLocalRank = 2;
  conn.sameProcess = 1;

  ASSERT_EQ(proxySharedInit(&conn, &proxyState_, /*nChannels=*/3), ncclSuccess);

  EXPECT_EQ(State(2, 0).refcount, 1);
  EXPECT_EQ(State(2, 0).size, (int64_t)3 * NCCL_SHARED_STEPS * kChunk);
  EXPECT_EQ(State(2, 1).refcount, 0);
  bool wsl = access("/dev/dxg", F_OK) == 0;
  EXPECT_EQ(State(2, 0).hostBuff != nullptr, wsl);
  EXPECT_EQ(State(2, 0).cudaBuff != nullptr, !wsl);
}

// ---------------------------------------------------------------------------
// Proxy-side setup (sendProxySetup, recvProxySetup)
// ---------------------------------------------------------------------------

class NetProxySetupMicrotest : public NetMicrotestBase {
 protected:
  ncclProxyConnection conn_{};
  setupReq req_{};
  ncclNetHandle_t handle_{};
  int done_ = 0;

  void SetUp() override {
    NetMicrotestBase::SetUp();
    proxyState_.tpLocalnRanks = 4;
    req_.tpRank = 5;
    req_.tpLocalRank = 1;
    req_.tpRemoteRank = 9;
    req_.netDev = 2;
  }

  void TearDown() override {
    free(conn_.transportResources);
    NetMicrotestBase::TearDown();
  }

  ncclResult_t SendSetup(int reqSize = sizeof(setupReq), int respSize = 0) {
    return sendProxySetup(&conn_, &proxyState_, &req_, reqSize, nullptr, respSize, &done_);
  }
  ncclResult_t RecvSetup(int reqSize = sizeof(setupReq), int respSize = sizeof(ncclNetHandle_t)) {
    return recvProxySetup(&conn_, &proxyState_, &req_, reqSize, &handle_, respSize, &done_);
  }
  sendNetResources* SendRes() { return static_cast<sendNetResources*>(conn_.transportResources); }
  recvNetResources* RecvRes() { return static_cast<recvNetResources*>(conn_.transportResources); }

  // Runs one side's setup on a fresh connection and reports the DMA-BUF choice.
  void ResetConn() {
    free(conn_.transportResources);
    conn_ = {};
  }

  int UseDmaBuf(bool send) {
    ResetConn();
    EXPECT_EQ(send ? SendSetup() : RecvSetup(), ncclSuccess);
    if (!conn_.transportResources) return -1;
    return send ? SendRes()->useDmaBuf : RecvRes()->useDmaBuf;
  }
};

TEST_F(NetProxySetupMicrotest, SendSetup_ValidRequest_RecordsRequestAndNicLimits) {
  req_.shared = 1;
  req_.useGdr = ncclTopoGdrModePci;
  req_.channelId = 3;
  req_.connIndex = 1;
  req_.isP2p = 1;
  req_.sameDevice = 1;
  uint32_t hdp = 0;
  req_.curr_hdp_reg = &hdp;
  net_.props.maxRecvs = 8;
  net_.props.netDeviceType = NCCL_NET_DEVICE_UNPACK;
  net_.props.netDeviceVersion = 7;
  net_.props.maxP2pBytes = 1 << 20;

  ASSERT_EQ(SendSetup(), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(net_.lastPropsDev, 2);
  EXPECT_EQ(conn_.shared, 1);
  sendNetResources* r = SendRes();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r->tpRank, 5);
  EXPECT_EQ(r->tpLocalRank, 1);
  EXPECT_EQ(r->tpRemoteRank, 9);
  EXPECT_EQ(r->netDev, 2);
  EXPECT_EQ(r->shared, 1);
  EXPECT_EQ(r->useGdr, ncclTopoGdrModePci);
  EXPECT_EQ(r->channelId, 3);
  EXPECT_EQ(r->connIndex, 1);
  EXPECT_EQ(r->curr_hdp_reg, &hdp);
  EXPECT_EQ(r->isP2p, 1);
  EXPECT_EQ(r->sameDevice, 1);
  EXPECT_EQ(r->maxRecvs, 8);
  EXPECT_EQ(r->netDeviceType, NCCL_NET_DEVICE_UNPACK);
  EXPECT_EQ(r->netDeviceVersion, 7);
  EXPECT_EQ(r->maxP2pBytes, (size_t)1 << 20);
}

TEST_F(NetProxySetupMicrotest, RecvSetup_ValidRequest_RecordsRequestAndListens) {
  req_.shared = 1;
  req_.useGdr = ncclTopoGdrModeDefault;
  req_.needFlush = ncclTopoFlushAlways;
  req_.channelId = 3;
  req_.connIndex = 1;
  req_.sameDevice = 1;
  uint32_t hdp = 0;
  req_.curr_hdp_reg = &hdp;
  net_.props.maxRecvs = 8;
  net_.props.netDeviceType = NCCL_NET_DEVICE_UNPACK;
  net_.props.netDeviceVersion = 7;
  net_.props.maxP2pBytes = 1 << 20;

  ASSERT_EQ(RecvSetup(), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(net_.lastPropsDev, 2);
  EXPECT_EQ(conn_.shared, 1);
  recvNetResources* r = RecvRes();
  ASSERT_NE(r, nullptr);
  EXPECT_EQ(r->tpRank, 5);
  EXPECT_EQ(r->tpLocalRank, 1);
  EXPECT_EQ(r->tpRemoteRank, 9);
  EXPECT_EQ(r->netDev, 2);
  EXPECT_EQ(r->shared, 1);
  EXPECT_EQ(r->useGdr, ncclTopoGdrModeDefault);
  EXPECT_EQ(r->needFlush, ncclTopoFlushAlways);
  EXPECT_EQ(r->curr_hdp_reg, &hdp);
  EXPECT_EQ(r->maxP2pBytes, (size_t)1 << 20);
  EXPECT_EQ(r->channelId, 3);
  EXPECT_EQ(r->connIndex, 1);
  EXPECT_EQ(r->sameDevice, 1);
  EXPECT_EQ(r->maxRecvs, 8);
  EXPECT_EQ(r->netDeviceType, NCCL_NET_DEVICE_UNPACK);
  EXPECT_EQ(r->netDeviceVersion, 7);
  EXPECT_EQ(net_.listenCalls, 1);
  EXPECT_EQ(net_.lastListenCtx, &net_);
  EXPECT_EQ(net_.lastListenDev, 2);
  EXPECT_EQ(r->netListenComm, &net_.listenComm);
  EXPECT_EQ(handle_[0], 0x5A);  // the plugin's listen handle is the response
}

TEST_F(NetProxySetupMicrotest, Setup_WrongRequestSize_RejectedWithoutResources) {
  EXPECT_EQ(SendSetup(sizeof(setupReq) - 1), ncclInternalError);
  EXPECT_EQ(RecvSetup(sizeof(setupReq) + 1), ncclInternalError);
  EXPECT_EQ(conn_.transportResources, nullptr);
}

TEST_F(NetProxySetupMicrotest, Setup_LocalRankOutOfRange_RejectedWithoutResources) {
  for (int bad : {4, -1}) {
    req_.tpLocalRank = bad;
    EXPECT_EQ(SendSetup(), ncclInvalidArgument);
    EXPECT_EQ(RecvSetup(), ncclInvalidArgument);
  }
  EXPECT_EQ(conn_.transportResources, nullptr);
}

TEST_F(NetProxySetupMicrotest, Setup_PropertiesFail_ReturnsError) {
  net_.propsResult = ncclSystemError;
  EXPECT_EQ(SendSetup(), ncclSystemError);
  ResetConn();
  EXPECT_EQ(RecvSetup(), ncclSystemError);
  EXPECT_EQ(net_.listenCalls, 0);
}

TEST_F(NetProxySetupMicrotest, Setup_InvalidMaxP2pBytes_Rejected) {
  for (size_t bad : {(size_t)0, (size_t)NCCL_MAX_NET_SIZE_BYTES + 1}) {
    SCOPED_TRACE(bad);
    net_.props.maxP2pBytes = bad;
    ResetConn();
    EXPECT_EQ(SendSetup(), ncclInternalError);
    ResetConn();
    EXPECT_EQ(RecvSetup(), ncclInternalError);
  }
  EXPECT_EQ(net_.listenCalls, 0);
}

TEST_F(NetProxySetupMicrotest, Setup_MaxP2pBytesAtLimit_Accepted) {
  net_.props.maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  EXPECT_EQ(SendSetup(), ncclSuccess);
  ResetConn();
  EXPECT_EQ(RecvSetup(), ncclSuccess);
}

TEST_F(NetProxySetupMicrotest, Setup_WrongResponseSize_Rejected) {
  EXPECT_EQ(SendSetup(sizeof(setupReq), /*respSize=*/4), ncclInternalError);
  EXPECT_EQ(done_, 0);
  ResetConn();
  EXPECT_EQ(RecvSetup(sizeof(setupReq), /*respSize=*/0), ncclInternalError);
  EXPECT_EQ(net_.listenCalls, 0);
}

TEST_F(NetProxySetupMicrotest, RecvSetup_ListenFails_ReturnsError) {
  net_.listenResult = ncclSystemError;

  EXPECT_EQ(RecvSetup(), ncclSystemError);
  EXPECT_EQ(done_, 0);
}

TEST_F(NetProxySetupMicrotest, Setup_GdrWithPeermem_PrefersPeermem) {
  req_.useGdr = ncclTopoGdrModeDefault;
  proxyState_.dmaBufSupport = true;
  net_.props.ptrSupport = NCCL_PTR_CUDA | NCCL_PTR_DMABUF;

  EXPECT_EQ(UseDmaBuf(true), 0);
  EXPECT_EQ(UseDmaBuf(false), 0);
}

TEST_F(NetProxySetupMicrotest, Setup_GdrWithoutPeermem_UsesDmaBuf) {
  req_.useGdr = ncclTopoGdrModeDefault;
  proxyState_.dmaBufSupport = true;
  net_.props.ptrSupport = NCCL_PTR_DMABUF;

  EXPECT_EQ(UseDmaBuf(true), 1);
  EXPECT_EQ(UseDmaBuf(false), 1);
}

TEST_F(NetProxySetupMicrotest, Setup_ForceDmaBuf_UsesDmaBufOverPeermem) {
  ScopedHook param(g_loadParam, ParamOverride({{"RCCL_FORCE_ENABLE_DMABUF", 1}}));
  req_.useGdr = ncclTopoGdrModeDefault;
  proxyState_.dmaBufSupport = true;
  net_.props.ptrSupport = NCCL_PTR_CUDA | NCCL_PTR_DMABUF;

  EXPECT_EQ(UseDmaBuf(true), 1);
  EXPECT_EQ(UseDmaBuf(false), 1);
}

TEST_F(NetProxySetupMicrotest, Setup_CuMem_UsesDmaBufOverPeermem) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  req_.useGdr = ncclTopoGdrModeDefault;
  proxyState_.dmaBufSupport = true;
  net_.props.ptrSupport = NCCL_PTR_CUDA | NCCL_PTR_DMABUF;

  EXPECT_EQ(UseDmaBuf(true), 1);
  EXPECT_EQ(UseDmaBuf(false), 1);
}

TEST_F(NetProxySetupMicrotest, Setup_ProxyWithoutDmaBuf_NeverUsesDmaBuf) {
  ScopedHook param(g_loadParam, ParamOverride({{"RCCL_FORCE_ENABLE_DMABUF", 1}}));
  req_.useGdr = ncclTopoGdrModeDefault;
  proxyState_.dmaBufSupport = false;
  net_.props.ptrSupport = NCCL_PTR_DMABUF;

  EXPECT_EQ(UseDmaBuf(true), 0);
  EXPECT_EQ(UseDmaBuf(false), 0);
}

TEST_F(NetProxySetupMicrotest, Setup_NoGdr_NeverUsesDmaBuf) {
  req_.useGdr = ncclTopoGdrModeDisable;
  proxyState_.dmaBufSupport = true;
  net_.props.ptrSupport = NCCL_PTR_DMABUF;
  EXPECT_EQ(UseDmaBuf(true), 0);
  EXPECT_EQ(UseDmaBuf(false), 0);

  ScopedHook param(g_loadParam, ParamOverride({{"RCCL_FORCE_ENABLE_DMABUF", 1}}));
  EXPECT_EQ(UseDmaBuf(true), 0);
  EXPECT_EQ(UseDmaBuf(false), 0);
}

// ---------------------------------------------------------------------------
// Client-side setup (sendSetup, recvSetup)
// ---------------------------------------------------------------------------

class NetClientMicrotestBase : public NetMicrotestBase {
 protected:
  static constexpr int kRanks = 4;
  static constexpr int kMyRank = 1;
  static constexpr int kPeerRank = 3;

  std::unique_ptr<ncclTopoSystem> topo_ = std::make_unique<ncclTopoSystem>();
  ncclComm comm_{};
  ncclPeerInfo peerInfo_[kRanks]{};
  int topParentRanks_[kRanks] = {10, 11, 12, 13};
  int topParentLocalRanks_[kRanks] = {20, 21, 22, 23};
  ncclConnector conn_{};
  ncclConnect connectInfo_{};

  void SetUp() override {
    NetMicrotestBase::SetUp();
    strcpy(topo_->nodes[GPU].nodes[0].gpu.gcn, "gfx942");
    comm_.topo = topo_.get();
    comm_.ncclNet = &vtable_;
    comm_.nRanks = kRanks;
    comm_.rank = kMyRank;
    comm_.localRank = 2;
    comm_.cudaDev = 0;
    comm_.useGdr = true;
    comm_.peerInfo = peerInfo_;
    comm_.topParentRanks = topParentRanks_;
    comm_.topParentLocalRanks = topParentLocalRanks_;
    for (int r = 0; r < kRanks; r++) {
      peerInfo_[r].rank = r;
      peerInfo_[r].cudaDev = r;
    }
    peerInfo_[kMyRank].cudaDev = comm_.cudaDev;
  }

  int ConnectInfoInt(size_t offset) {
    int v;
    memcpy(&v, reinterpret_cast<uint8_t*>(&connectInfo_) + offset, sizeof(v));
    return v;
  }
};

class NetSetupMicrotest : public NetClientMicrotestBase {
 protected:
  ncclTopoGraph graph_{};
  ncclTopoGdrMode gdr_ = ncclTopoGdrModeDefault;
  int netDev_ = 1;
  int proxyRank_ = kMyRank;  // what ncclTopoGetNetDev reports
  int intraNetDev_ = -1;     // what ncclTopoGetIntraNetDev reports

  int getNetDevCalls_ = 0, getNetDevPeer_ = -1;
  int intraCalls_ = 0, intraType_ = -1;
  int checkGdrRead_ = -1;
  int connectSend_ = -1, connectProxyRank_ = -1, connectTransport_ = -1;
  int callType_ = -1, callRespSize_ = -1;
  void* callResp_ = nullptr;
  setupReq req_{};

  void SetUp() override {
    NetClientMicrotestBase::SetUp();
    g_ncclTopoGetNetDev = [this](ncclComm*, int, ncclTopoGraph*, int, int peerRank, int64_t* id, int* dev,
                                 int* proxyRank) {
      ++getNetDevCalls_;
      getNetDevPeer_ = peerRank;
      *id = 0x77;
      *dev = netDev_;
      *proxyRank = proxyRank_;
      return ncclSuccess;
    };
    g_ncclTopoGetIntraNetDev = [this](ncclTopoSystem*, int, ncclTopoGraph*, int, int type, int64_t* id, int* dev) {
      ++intraCalls_;
      intraType_ = type;
      *id = 0x66;
      *dev = intraNetDev_;
      return ncclSuccess;
    };
    g_ncclTopoCheckGdr = [this](ncclTopoSystem*, int, int64_t, int read, ncclTopoGdrMode* mode) {
      checkGdrRead_ = read;
      *mode = gdr_;
      return ncclSuccess;
    };
    g_ncclTopoNeedFlush = [](ncclComm*, int64_t, int, int, bool, ncclTopoFlushType* flush) {
      *flush = ncclTopoFlushNone;
      return ncclSuccess;
    };
    g_hipDeviceGetAttributeResult = hipSuccess;
    g_proxyConnect = [this](ncclComm*, int transport, int send, int proxyRank, ncclProxyConnector*) {
      connectTransport_ = transport;
      connectSend_ = send;
      connectProxyRank_ = proxyRank;
      return ncclSuccess;
    };
    g_proxyCallBlocking = [this](ncclComm*, ncclProxyConnector*, int type, void* reqBuff, int reqSize, void* resp,
                                 int respSize) {
      callType_ = type;
      callResp_ = resp;
      callRespSize_ = respSize;
      EXPECT_EQ(reqSize, (int)sizeof(setupReq));
      memcpy(&req_, reqBuff, sizeof(req_));
      if (resp) memset(resp, 0x5A, respSize);
      return ncclSuccess;
    };
  }

  ncclResult_t SendSetup(ncclTopoGraph* graph, int connIndex, int channelId = 2) {
    return sendSetup(&comm_, graph, &peerInfo_[kMyRank], &peerInfo_[kPeerRank], &connectInfo_, &conn_, channelId,
                     connIndex);
  }
  ncclResult_t RecvSetup(ncclTopoGraph* graph, int connIndex, int channelId = 2) {
    return recvSetup(&comm_, graph, &peerInfo_[kMyRank], &peerInfo_[kPeerRank], &connectInfo_, &conn_, channelId,
                     connIndex);
  }
};

TEST_F(NetSetupMicrotest, SendSetup_RingConnection_SetsUpDedicatedBuffersOnOwnProxy) {
  ASSERT_EQ(SendSetup(&graph_, /*connIndex=*/0), ncclSuccess);

  EXPECT_EQ(connectTransport_, TRANSPORT_NET);
  EXPECT_EQ(connectSend_, 1);
  EXPECT_EQ(connectProxyRank_, kMyRank);
  EXPECT_EQ(callType_, ncclProxyMsgSetup);
  EXPECT_EQ(callResp_, nullptr);
  EXPECT_EQ(conn_.conn.shared, 0);
  EXPECT_EQ(req_.shared, 0);
  EXPECT_EQ(req_.isP2p, 0);
  EXPECT_EQ(req_.netDev, 1);
  EXPECT_EQ(req_.useGdr, ncclTopoGdrModeDefault);
  EXPECT_EQ(req_.channelId, 2);
  EXPECT_EQ(req_.connIndex, 0);
  EXPECT_EQ(req_.tpLocalRank, topParentLocalRanks_[comm_.localRank]);
  EXPECT_EQ(req_.tpRank, topParentRanks_[kMyRank]);
  EXPECT_EQ(req_.tpRemoteRank, topParentRanks_[kPeerRank]);
  EXPECT_EQ(req_.sameDevice, 1);
  EXPECT_EQ(checkGdrRead_, 1);
  EXPECT_EQ(intraCalls_, 0);
  EXPECT_EQ(getNetDevPeer_, kPeerRank);
  EXPECT_EQ(ConnectInfoInt(0), topParentRanks_[kMyRank]);
  EXPECT_EQ(ConnectInfoInt(sizeof(ncclNetHandle_t)), (int)ncclTopoGdrModeDefault);
}

TEST_F(NetSetupMicrotest, Setup_P2pConnection_SharesBuffersByDefault) {
  ASSERT_EQ(SendSetup(nullptr, /*connIndex=*/1), ncclSuccess);
  EXPECT_EQ(conn_.conn.shared, 1);
  EXPECT_EQ(req_.shared, 1);
  EXPECT_EQ(req_.isP2p, 1);

  conn_ = {};
  ASSERT_EQ(RecvSetup(nullptr, 1), ncclSuccess);
  EXPECT_EQ(conn_.conn.shared, 1);
  EXPECT_EQ(req_.shared, 1);
}

TEST_F(NetSetupMicrotest, Setup_GraphOrFirstConnection_KeepsDedicatedBuffers) {
  ASSERT_EQ(SendSetup(&graph_, /*connIndex=*/1), ncclSuccess);
  EXPECT_EQ(req_.shared, 0);
  ASSERT_EQ(RecvSetup(&graph_, 1), ncclSuccess);
  EXPECT_EQ(req_.shared, 0);

  ASSERT_EQ(SendSetup(nullptr, /*connIndex=*/0), ncclSuccess);
  EXPECT_EQ(req_.shared, 0);
  ASSERT_EQ(RecvSetup(nullptr, 0), ncclSuccess);
  EXPECT_EQ(req_.shared, 0);
}

TEST_F(NetSetupMicrotest, Setup_NetSharedBuffersParam_OverridesP2pDefault) {
  ScopedHook param(g_loadParam, ParamOverride({{"NET_SHARED_BUFFERS", 0}}));

  ASSERT_EQ(SendSetup(nullptr, 1), ncclSuccess);
  EXPECT_EQ(req_.shared, 0);
  ASSERT_EQ(RecvSetup(nullptr, 1), ncclSuccess);
  EXPECT_EQ(req_.shared, 0);
}

TEST_F(NetSetupMicrotest, Setup_P2pNetConnection_PrefersIntraNodeNic) {
  intraNetDev_ = 3;

  ASSERT_EQ(SendSetup(nullptr, NCCL_CONN_IDX_P2P_NET), ncclSuccess);
  EXPECT_EQ(intraType_, 1);
  EXPECT_EQ(getNetDevCalls_, 0);
  EXPECT_EQ(req_.netDev, 3);

  ASSERT_EQ(RecvSetup(nullptr, NCCL_CONN_IDX_P2P_NET), ncclSuccess);
  EXPECT_EQ(intraType_, 0);
  EXPECT_EQ(getNetDevCalls_, 0);
  EXPECT_EQ(req_.netDev, 3);
}

TEST_F(NetSetupMicrotest, Setup_P2pNetConnectionWithoutIntraNodeNic_FallsBackToTopology) {
  ASSERT_EQ(SendSetup(nullptr, NCCL_CONN_IDX_P2P_NET), ncclSuccess);
  EXPECT_EQ(intraCalls_, 1);
  EXPECT_EQ(getNetDevCalls_, 1);
  EXPECT_EQ(req_.netDev, 1);
}

TEST_F(NetSetupMicrotest, SendSetup_PxnProxy_ConnectsThroughProxyRank) {
  proxyRank_ = 2;

  ASSERT_EQ(SendSetup(&graph_, 0), ncclSuccess);

  EXPECT_EQ(connectProxyRank_, 2);
  EXPECT_TRUE(comm_.useNetPXN);
  EXPECT_EQ(req_.sameDevice, 0);  // peerInfo[2] is on another device
  EXPECT_EQ(ConnectInfoInt(0), topParentRanks_[2]);
}

TEST_F(NetSetupMicrotest, SendSetup_PxnOnNonRingConnection_LeavesCommPxnFlag) {
  proxyRank_ = 2;

  ASSERT_EQ(SendSetup(nullptr, 1), ncclSuccess);

  EXPECT_FALSE(comm_.useNetPXN);
}

TEST_F(NetSetupMicrotest, Setup_Gdr_MarksDirectNic) {
  ASSERT_EQ(SendSetup(&graph_, 0), ncclSuccess);
  EXPECT_TRUE(conn_.conn.flags & NCCL_DIRECT_NIC);
  EXPECT_TRUE(comm_.useGdr);

  conn_ = {};
  ASSERT_EQ(RecvSetup(&graph_, 0), ncclSuccess);
  EXPECT_TRUE(conn_.conn.flags & NCCL_DIRECT_NIC);
}

TEST_F(NetSetupMicrotest, Setup_NoGdrOnRing_ClearsCommGdr) {
  gdr_ = ncclTopoGdrModeDisable;

  ASSERT_EQ(SendSetup(&graph_, 0), ncclSuccess);
  EXPECT_FALSE(conn_.conn.flags & NCCL_DIRECT_NIC);
  EXPECT_FALSE(comm_.useGdr);

  comm_.useGdr = true;
  ASSERT_EQ(RecvSetup(&graph_, 0), ncclSuccess);
  EXPECT_FALSE(conn_.conn.flags & NCCL_DIRECT_NIC);
  EXPECT_FALSE(comm_.useGdr);
}

TEST_F(NetSetupMicrotest, Setup_NoGdrOnOtherConnection_LeavesCommGdr) {
  gdr_ = ncclTopoGdrModeDisable;

  ASSERT_EQ(SendSetup(nullptr, 1), ncclSuccess);
  ASSERT_EQ(RecvSetup(nullptr, 1), ncclSuccess);

  EXPECT_TRUE(comm_.useGdr);
}

TEST_F(NetSetupMicrotest, SendSetup_GdrOnArchWithoutCoherentHdp_RecordsHdpFlushRegister) {
  strcpy(topo_->nodes[GPU].nodes[0].gpu.gcn, "gfx1100");
  int queriedDev = -1;
  ScopedHook attr(g_hipDeviceGetAttribute, [&](int* v, hipDeviceAttribute_t a, int dev) {
    EXPECT_EQ(a, hipDeviceAttributeHdpMemFlushCntl);
    queriedDev = dev;
    *v = 0x1234;
    return hipSuccess;
  });
  peerInfo_[kMyRank].cudaDev = 5;

  ASSERT_EQ(SendSetup(&graph_, 0), ncclSuccess);

  EXPECT_EQ(queriedDev, 5);
  EXPECT_EQ(conn_.conn.curr_hdp_reg, reinterpret_cast<uint32_t*>(0x1234));
  EXPECT_EQ(req_.curr_hdp_reg, reinterpret_cast<uint32_t*>(0x1234));
}

TEST_F(NetSetupMicrotest, SendSetup_GdrOnCoherentArch_SkipsHdpRegister) {
  for (const char* arch : {"gfx90a", "gfx942", "gfx950"}) {
    SCOPED_TRACE(arch);
    strcpy(topo_->nodes[GPU].nodes[0].gpu.gcn, arch);
    ScopedHook attr(g_hipDeviceGetAttribute, [](int*, hipDeviceAttribute_t, int) { return hipErrorInvalidValue; });

    ASSERT_EQ(SendSetup(&graph_, 0), ncclSuccess);

    EXPECT_EQ(attr.calls, 0);
    EXPECT_EQ(req_.curr_hdp_reg, nullptr);
  }
}

TEST_F(NetSetupMicrotest, SendSetup_NoGdr_SkipsHdpRegister) {
  strcpy(topo_->nodes[GPU].nodes[0].gpu.gcn, "gfx1100");
  gdr_ = ncclTopoGdrModeDisable;
  ScopedHook attr(g_hipDeviceGetAttribute, [](int*, hipDeviceAttribute_t, int) { return hipErrorInvalidValue; });

  ASSERT_EQ(SendSetup(&graph_, 0), ncclSuccess);

  EXPECT_EQ(attr.calls, 0);
}

TEST_F(NetSetupMicrotest, RecvSetup_RingConnection_ListensOnOwnNicAndReturnsHandle) {
  gdr_ = ncclTopoGdrModePci;

  ASSERT_EQ(RecvSetup(&graph_, 0), ncclSuccess);

  EXPECT_EQ(connectTransport_, TRANSPORT_NET);
  EXPECT_EQ(connectSend_, 0);
  EXPECT_EQ(connectProxyRank_, kMyRank);
  EXPECT_EQ(getNetDevPeer_, kMyRank);
  EXPECT_EQ(checkGdrRead_, 0);
  EXPECT_EQ(callType_, ncclProxyMsgSetup);
  EXPECT_EQ(callResp_, &connectInfo_);
  EXPECT_EQ(callRespSize_, (int)sizeof(ncclNetHandle_t));
  EXPECT_EQ(req_.netDev, 1);
  EXPECT_EQ(req_.channelId, 2);
  EXPECT_EQ(req_.tpLocalRank, topParentLocalRanks_[comm_.localRank]);
  EXPECT_EQ(req_.tpRank, topParentRanks_[kMyRank]);
  EXPECT_EQ(req_.tpRemoteRank, topParentRanks_[kPeerRank]);
  EXPECT_EQ(req_.sameDevice, 1);
  EXPECT_EQ(ConnectInfoInt(0), 0x5A5A5A5A);  // the proxy's listen handle
  EXPECT_EQ(ConnectInfoInt(sizeof(ncclNetHandle_t)), (int)ncclTopoGdrModePci);
}

TEST_F(NetSetupMicrotest, RecvSetup_Gdr_AsksTopologyWhetherToFlush) {
  g_hipDirectManagedMemAccess = 0;
  bool managedSeen = true;
  int flushNetDev = -1;
  ScopedHook flush(g_ncclTopoNeedFlush, [&](ncclComm*, int64_t netId, int netDev, int rank, bool managed,
                                            ncclTopoFlushType* f) {
    EXPECT_EQ(netId, 0x77);
    EXPECT_EQ(rank, kMyRank);
    flushNetDev = netDev;
    managedSeen = managed;
    *f = ncclTopoFlushAlways;
    return ncclSuccess;
  });

  ASSERT_EQ(RecvSetup(&graph_, 0), ncclSuccess);

  EXPECT_EQ(flush.calls, 1);
  EXPECT_EQ(flushNetDev, 1);
  EXPECT_FALSE(managedSeen);
  EXPECT_EQ(req_.needFlush, ncclTopoFlushAlways);

  g_hipDirectManagedMemAccess = 1;
  ASSERT_EQ(RecvSetup(&graph_, 0), ncclSuccess);
  EXPECT_TRUE(managedSeen);
}

TEST_F(NetSetupMicrotest, RecvSetup_Gdr_RecordsHdpFlushRegister) {
  int hdpDev = -1;
  ScopedHook attr(g_hipDeviceGetAttribute, [&](int* v, hipDeviceAttribute_t a, int dev) {
    if (a == hipDeviceAttributeHdpMemFlushCntl) {
      hdpDev = dev;
      *v = 0x4321;
    } else {
      *v = 0;
    }
    return hipSuccess;
  });
  peerInfo_[kMyRank].cudaDev = 6;

  ASSERT_EQ(RecvSetup(&graph_, 0), ncclSuccess);

  EXPECT_EQ(hdpDev, 6);
  EXPECT_EQ(conn_.conn.curr_hdp_reg, reinterpret_cast<uint32_t*>(0x4321));
  EXPECT_EQ(req_.curr_hdp_reg, reinterpret_cast<uint32_t*>(0x4321));
}

TEST_F(NetSetupMicrotest, RecvSetup_NoGdr_SkipsFlushQuery) {
  gdr_ = ncclTopoGdrModeDisable;
  ScopedHook flush(g_ncclTopoNeedFlush, [](ncclComm*, int64_t, int, int, bool, ncclTopoFlushType*) {
    return ncclInternalError;
  });

  ASSERT_EQ(RecvSetup(&graph_, 0), ncclSuccess);

  EXPECT_EQ(flush.calls, 0);
  EXPECT_EQ(req_.needFlush, ncclTopoFlushNone);
}

TEST_F(NetSetupMicrotest, Setup_DependencyFails_ReturnsError) {
  auto fails = [](auto...) { return ncclInternalError; };
  {
    ScopedHook hook(g_ncclTopoGetNetDev, fails);
    EXPECT_EQ(SendSetup(&graph_, 0), ncclInternalError);
    EXPECT_EQ(RecvSetup(&graph_, 0), ncclInternalError);
  }
  {
    ScopedHook hook(g_ncclTopoGetIntraNetDev, fails);
    EXPECT_EQ(SendSetup(nullptr, NCCL_CONN_IDX_P2P_NET), ncclInternalError);
    EXPECT_EQ(RecvSetup(nullptr, NCCL_CONN_IDX_P2P_NET), ncclInternalError);
  }
  {
    ScopedHook hook(g_ncclTopoCheckGdr, fails);
    EXPECT_EQ(SendSetup(&graph_, 0), ncclInternalError);
    EXPECT_EQ(RecvSetup(&graph_, 0), ncclInternalError);
  }
  {
    ScopedHook hook(g_ncclTopoNeedFlush, fails);
    EXPECT_EQ(RecvSetup(&graph_, 0), ncclInternalError);
  }
  {
    ScopedHook hook(g_proxyConnect, fails);
    EXPECT_EQ(SendSetup(&graph_, 0), ncclInternalError);
    EXPECT_EQ(RecvSetup(&graph_, 0), ncclInternalError);
  }
  {
    ScopedHook hook(g_proxyCallBlocking, fails);
    EXPECT_EQ(SendSetup(&graph_, 0), ncclInternalError);
    EXPECT_EQ(RecvSetup(&graph_, 0), ncclInternalError);
  }
}

}  // namespace
