/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/transport/net.cc, driven through a fake ncclNet_t.
// This TU #includes the hipified net.cc (NET_CC_PATH) so its file-static helpers
// are reachable. No GPU, no librccl.so, no network.

#include <gtest/gtest.h>

#include <fcntl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ScopedHook.h"
#include "fakes/env_fakes.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/proxy_fakes.h"
#include "fakes/rocmwrap_fakes.h"
#include "fakes/topo_net_stubs.h"
#include "fakes/topo_stubs.h"
#include "fakes/transport_net_ib_fakes.h"
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

  // connect: hands out &sendComm, or nothing while commReady is false (still connecting).
  ncclResult_t connectResult = ncclSuccess;
  bool commReady = true;
  int connectCalls = 0;
  int lastConnectDev = -1;
  void* lastConnectCtx = nullptr;
  unsigned char lastConnectHandle[NCCL_NET_HANDLE_MAXSIZE] = {};
  void* lastDevHandleArg = nullptr;
  uint32_t lastDevHandleWord = 0;  // first word behind it: the channel id when it is an ncclNet_ctxt_t
  ncclNetDeviceHandle_t* offloadHandle = nullptr;  // written to *devHandle when set
  char sendComm = 0;

  // accept: like connect, hands out &recvComm unless commReady is false.
  ncclResult_t acceptResult = ncclSuccess;
  int acceptCalls = 0;
  void* lastAcceptListen = nullptr;
  char recvComm = 0;
  std::vector<void*> closedListens;

  struct MrCall {
    void* comm;
    void* data;
    size_t size;
    int type;
    uint64_t offset;
    int fd;
  };
  std::vector<MrCall> regMrs, regMrDmaBufs;
  ncclResult_t regMrResult = ncclSuccess;
  ncclResult_t regMrDmaBufResult = ncclSuccess;  // fails only DMA-BUF registration when set
  int getDeviceMrCalls = 0;
  std::vector<std::pair<void*, void*>> deregMrs;  // (comm, mhandle)
  ncclResult_t deregMrResult = ncclSuccess;
  std::vector<void*> closedSends, closedRecvs;

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

  ncclResult_t connect(void* ctx, int dev, void* handle, void** comm, ncclNetDeviceHandle_t** devHandle) {
    ++connectCalls;
    lastConnectCtx = ctx;
    lastConnectDev = dev;
    memcpy(lastConnectHandle, handle, NCCL_NET_HANDLE_MAXSIZE);
    lastDevHandleArg = devHandle;
    memcpy(&lastDevHandleWord, devHandle, sizeof(lastDevHandleWord));
    if (offloadHandle) *devHandle = offloadHandle;
    if (commReady) *comm = &sendComm;
    return connectResult;
  }

  ncclResult_t accept(void* listenComm, void** comm, ncclNetDeviceHandle_t** devHandle) {
    ++acceptCalls;
    lastAcceptListen = listenComm;
    lastDevHandleArg = devHandle;
    memcpy(&lastDevHandleWord, devHandle, sizeof(lastDevHandleWord));
    if (offloadHandle) *devHandle = offloadHandle;
    if (commReady) *comm = &recvComm;
    return acceptResult;
  }
  ncclResult_t closeListen(void* comm) {
    closedListens.push_back(comm);
    return ncclSuccess;
  }

  // An mhandle is the registered address, so tests can tell which buffer it came from.
  ncclResult_t regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
    regMrs.push_back({comm, data, size, type, 0, -1});
    *mhandle = data;
    return regMrResult;
  }
  ncclResult_t regMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd, void** mhandle) {
    regMrDmaBufs.push_back({comm, data, size, type, offset, fd});
    *mhandle = data;
    return regMrDmaBufResult != ncclSuccess ? regMrDmaBufResult : regMrResult;
  }
  ncclResult_t getDeviceMr(void*, void* mhandle, void** dptr) {
    ++getDeviceMrCalls;
    *dptr = static_cast<char*>(mhandle) + 1;
    return ncclSuccess;
  }
  ncclResult_t deregMr(void* comm, void* mhandle) {
    deregMrs.push_back({comm, mhandle});
    return deregMrResult;
  }
  ncclResult_t closeSend(void* comm) {
    closedSends.push_back(comm);
    return ncclSuccess;
  }
  ncclResult_t closeRecv(void* comm) {
    closedRecvs.push_back(comm);
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
    v.connect = &FakeNet::TrampConnect;
    v.accept = &FakeNet::TrampAccept;
    v.closeListen = &FakeNet::TrampCloseListen;
    v.regMr = &FakeNet::TrampRegMr;
    v.regMrDmaBuf = &FakeNet::TrampRegMrDmaBuf;
    v.getDeviceMr = &FakeNet::TrampGetDeviceMr;
    v.deregMr = &FakeNet::TrampDeregMr;
    v.closeSend = &FakeNet::TrampCloseSend;
    v.closeRecv = &FakeNet::TrampCloseRecv;
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
  static ncclResult_t TrampConnect(void* ctx, int dev, void* handle, void** comm, ncclNetDeviceHandle_t** devHandle) {
    return g_activeNet->connect(ctx, dev, handle, comm, devHandle);
  }
  static ncclResult_t TrampAccept(void* listenComm, void** comm, ncclNetDeviceHandle_t** devHandle) {
    return g_activeNet->accept(listenComm, comm, devHandle);
  }
  static ncclResult_t TrampCloseListen(void* comm) { return g_activeNet->closeListen(comm); }
  static ncclResult_t TrampRegMr(void* comm, void* data, size_t size, int type, void** mhandle) {
    return g_activeNet->regMr(comm, data, size, type, mhandle);
  }
  static ncclResult_t TrampRegMrDmaBuf(void* comm, void* data, size_t size, int type, uint64_t offset, int fd,
                                       void** mhandle) {
    return g_activeNet->regMrDmaBuf(comm, data, size, type, offset, fd, mhandle);
  }
  static ncclResult_t TrampGetDeviceMr(void* comm, void* mhandle, void** dptr) {
    return g_activeNet->getDeviceMr(comm, mhandle, dptr);
  }
  static ncclResult_t TrampDeregMr(void* comm, void* mhandle) { return g_activeNet->deregMr(comm, mhandle); }
  static ncclResult_t TrampCloseSend(void* comm) { return g_activeNet->closeSend(comm); }
  static ncclResult_t TrampCloseRecv(void* comm) { return g_activeNet->closeRecv(comm); }
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
    ResetProxyFakes();
    ResetTopoStubs();
    ResetTransportNetIbFakes();
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

// ---------------------------------------------------------------------------
// Shared p2p buffer pool (sharedNetBuffersInit / Get / Destroy, proxySharedInit)
// ---------------------------------------------------------------------------

// Every device, host and shareable buffer goes through this ledger, so TearDown
// releases exactly what production left allocated.
class NetAllocMicrotestBase : public NetMicrotestBase {
 protected:
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
    ncclProxyProgressState& ps = proxyState_.progressState;
    if (ps.localPeers) {
      for (int r = 0; r < proxyState_.tpLocalnRanks; r++) free(ps.localPeers[r]);
      free(ps.localPeers);
    }
    for (auto*& comms : ps.netComms) free(comms);
    for (void* p : live_) free(p);
    NetMicrotestBase::TearDown();
  }
};

class NetSharedBuffersMicrotest : public NetAllocMicrotestBase {
 protected:
  static constexpr int kLocalRanks = 4;
  static constexpr int kChunk = 1 << 12;

  void SetUp() override {
    NetAllocMicrotestBase::SetUp();
    proxyState_.tpLocalnRanks = kLocalRanks;
    proxyState_.p2pChunkSize = kChunk;
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

// proxySharedInit picks host or device memory from /dev/dxg (WSL); these assertions hold for both.
TEST_F(NetSharedBuffersMicrotest, ProxySharedInit_ReservesSendPoolForConnectionRank) {
  ncclProxyConnection conn{};
  conn.tpLocalRank = 2;
  conn.sameProcess = 1;

  ASSERT_EQ(proxySharedInit(&conn, &proxyState_, /*nChannels=*/3), ncclSuccess);

  EXPECT_EQ(State(2, 0).refcount, 1);
  EXPECT_EQ(State(2, 0).size, (int64_t)3 * NCCL_SHARED_STEPS * kChunk);
  EXPECT_EQ(State(2, 1).refcount, 0);
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
  int UseDmaBuf(bool send) {
    free(conn_.transportResources);
    conn_ = {};
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
  net_.props.maxRecvs = 8;
  net_.props.netDeviceType = NCCL_NET_DEVICE_UNPACK;
  net_.props.netDeviceVersion = 7;

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
  EXPECT_EQ(r->useGdr, ncclTopoGdrModeDefault);
  EXPECT_EQ(r->needFlush, ncclTopoFlushAlways);
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
  req_.tpLocalRank = 4;
  EXPECT_EQ(SendSetup(), ncclInvalidArgument);
  req_.tpLocalRank = -1;
  EXPECT_EQ(RecvSetup(), ncclInvalidArgument);
  EXPECT_EQ(conn_.transportResources, nullptr);
}

TEST_F(NetProxySetupMicrotest, Setup_PropertiesFail_ReturnsError) {
  net_.propsResult = ncclSystemError;
  EXPECT_EQ(SendSetup(), ncclSystemError);
  free(conn_.transportResources);
  conn_ = {};
  EXPECT_EQ(RecvSetup(), ncclSystemError);
  EXPECT_EQ(net_.listenCalls, 0);
}

TEST_F(NetProxySetupMicrotest, Setup_InvalidMaxP2pBytes_Rejected) {
  for (size_t bad : {(size_t)0, (size_t)NCCL_MAX_NET_SIZE_BYTES + 1}) {
    SCOPED_TRACE(bad);
    net_.props.maxP2pBytes = bad;
    free(conn_.transportResources);
    conn_ = {};
    EXPECT_EQ(SendSetup(), ncclInternalError);
    free(conn_.transportResources);
    conn_ = {};
    EXPECT_EQ(RecvSetup(), ncclInternalError);
  }
  EXPECT_EQ(net_.listenCalls, 0);
}

TEST_F(NetProxySetupMicrotest, Setup_MaxP2pBytesAtLimit_Accepted) {
  net_.props.maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  EXPECT_EQ(SendSetup(), ncclSuccess);
  free(conn_.transportResources);
  conn_ = {};
  EXPECT_EQ(RecvSetup(), ncclSuccess);
}

TEST_F(NetProxySetupMicrotest, Setup_WrongResponseSize_Rejected) {
  EXPECT_EQ(SendSetup(sizeof(setupReq), /*respSize=*/4), ncclInternalError);
  EXPECT_EQ(done_, 0);
  free(conn_.transportResources);
  conn_ = {};
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

// ---------------------------------------------------------------------------
// Client-side connect and free (sendConnect, recvConnect, sendFree, recvFree)
// ---------------------------------------------------------------------------

class NetConnectMicrotest : public NetClientMicrotestBase {
 protected:
  static constexpr int kStepBytes = 4096;
  static constexpr size_t kSimpleBytes = (size_t)NCCL_STEPS * kStepBytes;

  ncclProxyConnection proxyConnection_{};
  connectMap map_{};  // what the proxy answers the connect request with
  std::vector<char> hostMem_, devMem_;
  ncclResult_t pollResult_ = ncclSuccess;
  int asyncCalls_ = 0, pollCalls_ = 0;
  int asyncType_ = -1, asyncReqSize_ = -1, asyncRespSize_ = -1;
  void* asyncOpId_ = nullptr;
  std::vector<char> asyncReq_;
  void* sharedDevMems_[4] = {};

  void SetUp() override {
    NetClientMicrotestBase::SetUp();
    comm_.buffSizes[NCCL_PROTO_SIMPLE] = kSimpleBytes;
    comm_.cudaDev = 4;
    g_currentDevice = 5;  // the free path asks HIP, not the comm, which device it is on
    comm_.proxyState = &proxyState_;
    proxyState_.sharedDevMems = sharedDevMems_;
    conn_.proxyConn.connection = &proxyConnection_;
    conn_.proxyConn.sameProcess = 1;
    conn_.proxyConn.rank = 2;
    conn_.proxyConn.tpLocalRank = 1;
    SetConnectInfoGdr(1);
    BuildMap(/*sameProcess=*/1, /*cudaDev=*/comm_.cudaDev, /*shared=*/0);
    g_ncclProxyCallAsync = [this](ncclComm*, ncclProxyConnector*, int type, void* req, int reqSize, int respSize,
                                  void* opId) {
      ++asyncCalls_;
      asyncType_ = type;
      asyncReqSize_ = reqSize;
      asyncRespSize_ = respSize;
      asyncOpId_ = opId;
      asyncReq_.assign(static_cast<char*>(req), static_cast<char*>(req) + reqSize);
      return ncclSuccess;
    };
    g_ncclPollProxyResponse = [this](ncclComm*, ncclProxyConnector*, void* resp, void*) {
      ++pollCalls_;
      if (pollResult_ == ncclSuccess) memcpy(resp, &map_, sizeof(map_));
      return pollResult_;
    };
  }

  void TearDown() override {
    free(conn_.transportResources);
    NetClientMicrotestBase::TearDown();
  }

  // The proxy's map: send/recv control words in host memory and the SIMPLE buffer in device
  // memory, either a dedicated bank or the per-local-rank shared one.
  void BuildMap(int sameProcess, int cudaDev, int shared, bool sharedDevBank = false) {
    map_ = {};
    map_.sameProcess = sameProcess;
    map_.cudaDev = cudaDev;
    map_.shared = shared;
    NCCL_NET_MAP_ADD_POINTER(&map_, 0, 0, sizeof(ncclSendMem), sendMem);
    NCCL_NET_MAP_ADD_POINTER(&map_, 0, 0, sizeof(ncclRecvMem), recvMem);
    NCCL_NET_MAP_ADD_POINTER(&map_, sharedDevBank ? 1 : 0, 1, kSimpleBytes, buffs[NCCL_PROTO_SIMPLE]);
    if (sharedDevBank) map_.mems[NCCL_NET_MAP_SHARED_DEVMEM].size = kSimpleBytes;
    hostMem_.assign(map_.mems[NCCL_NET_MAP_HOSTMEM].size, 0);
    devMem_.assign(map_.mems[NCCL_NET_MAP_DEVMEM].size, 0);
    map_.mems[NCCL_NET_MAP_HOSTMEM].cpuPtr = map_.mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = hostMem_.data();
    map_.mems[NCCL_NET_MAP_DEVMEM].cpuPtr = map_.mems[NCCL_NET_MAP_DEVMEM].gpuPtr = devMem_.data();
  }

  // sendFree / recvFree release the map, so hand them a heap copy.
  connectMap* HeapMap() {
    auto* p = static_cast<connectMap*>(malloc(sizeof(connectMap)));
    *p = map_;
    return p;
  }

  void SetConnectInfoGdr(int useGdr) {
    memcpy(reinterpret_cast<uint8_t*>(&connectInfo_) + sizeof(ncclNetHandle_t), &useGdr, sizeof(useGdr));
  }

  ncclResult_t SendConnect() { return sendConnect(&comm_, &connectInfo_, kRanks, kMyRank, &conn_); }
  ncclResult_t RecvConnect() { return recvConnect(&comm_, &connectInfo_, kRanks, kMyRank, &conn_); }

  ncclSendMem* SendMem() { return reinterpret_cast<ncclSendMem*>(hostMem_.data()); }
  ncclRecvMem* RecvMem() {
    return reinterpret_cast<ncclRecvMem*>(hostMem_.data() + (map_.offsets.recvMem & NCCL_NET_MAP_MASK_OFFSET));
  }
};

TEST_F(NetConnectMicrotest, SendConnect_FirstCall_PostsConnectRequestWithHandleAndHints) {
  memset(&connectInfo_, 0x5A, sizeof(ncclNetHandle_t));
  conn_.p2pOnly = 1;
  comm_.p2pnChannels = 2;
  comm_.p2pnChannelsPerPeer = 1;
  pollResult_ = ncclInProgress;

  EXPECT_EQ(SendConnect(), ncclInProgress);

  EXPECT_EQ(asyncCalls_, 1);
  EXPECT_EQ(asyncType_, ncclProxyMsgConnect);
  EXPECT_EQ(asyncReqSize_, (int)sizeof(netSendConnectArgs));
  EXPECT_EQ(asyncRespSize_, (int)sizeof(connectMap));
  EXPECT_EQ(asyncOpId_, &conn_);
  auto* args = reinterpret_cast<netSendConnectArgs*>(asyncReq_.data());
  EXPECT_EQ(memcmp(args->handle, &connectInfo_, sizeof(ncclNetHandle_t)), 0);
  EXPECT_EQ(args->netAttr.sendCommAttr.maxFlowsPerPeer, 1);  // p2p hints from the comm
  EXPECT_NE(conn_.transportResources, nullptr);
}

TEST_F(NetConnectMicrotest, Connect_StillInProgress_PollsWithoutReposting) {
  pollResult_ = ncclInProgress;
  ASSERT_EQ(SendConnect(), ncclInProgress);
  void* map = conn_.transportResources;

  EXPECT_EQ(SendConnect(), ncclInProgress);

  EXPECT_EQ(asyncCalls_, 1);
  EXPECT_EQ(pollCalls_, 2);
  EXPECT_EQ(conn_.transportResources, map);

  free(conn_.transportResources);
  conn_.transportResources = nullptr;
  ASSERT_EQ(RecvConnect(), ncclInProgress);
  EXPECT_EQ(RecvConnect(), ncclInProgress);
  EXPECT_EQ(asyncCalls_, 2);
}

TEST_F(NetConnectMicrotest, SendConnect_ProxyFails_DropsMapSoNextCallReposts) {
  pollResult_ = ncclSystemError;

  EXPECT_EQ(SendConnect(), ncclSystemError);
  EXPECT_EQ(conn_.transportResources, nullptr);

  pollResult_ = ncclSuccess;
  EXPECT_EQ(SendConnect(), ncclSuccess);
  EXPECT_EQ(asyncCalls_, 2);
}

TEST_F(NetConnectMicrotest, SendConnect_SameProcessSameDevice_WiresConnectorToProxyBuffers) {
  ScopedHook peer(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipSuccess; });

  ASSERT_EQ(SendConnect(), ncclSuccess);

  EXPECT_EQ(peer.calls, 0);
  EXPECT_EQ(conn_.conn.head, &SendMem()->head);
  EXPECT_EQ(conn_.conn.tail, &RecvMem()->tail);
  EXPECT_EQ(conn_.conn.connFifo, RecvMem()->connFifo);
  EXPECT_EQ(conn_.conn.stepSize, kStepBytes);
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_SIMPLE], devMem_.data());
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_LL], nullptr);
  for (int i = 0; i < NCCL_STEPS; i++) {
    EXPECT_EQ(RecvMem()->connFifo[i].offset, -1);
    EXPECT_EQ(RecvMem()->connFifo[i].mode, NCCL_MODE_NORMAL);
  }
}

TEST_F(NetConnectMicrotest, SendConnect_SharedBuffers_UsesOffsetFifoMode) {
  BuildMap(1, comm_.cudaDev, /*shared=*/1);

  ASSERT_EQ(SendConnect(), ncclSuccess);

  EXPECT_EQ(RecvMem()->connFifo[0].mode, NCCL_MODE_OFFSET);
}

TEST_F(NetConnectMicrotest, SendConnect_GdcSyncWord_UsedAsHead) {
  uint64_t gdc = 0;
  map_.mems[NCCL_NET_MAP_GDCMEM].gpuPtr = reinterpret_cast<char*>(&gdc);

  ASSERT_EQ(SendConnect(), ncclSuccess);

  EXPECT_EQ(conn_.conn.head, &gdc);
}

TEST_F(NetConnectMicrotest, SendConnect_SameProcessOtherDevice_EnablesPeerAccess) {
  BuildMap(1, /*cudaDev=*/3, 0);
  int peerDev = -1;
  ScopedHook peer(g_hipDeviceEnablePeerAccess, [&](int dev, unsigned) {
    peerDev = dev;
    return hipSuccess;
  });

  ASSERT_EQ(SendConnect(), ncclSuccess);

  EXPECT_EQ(peerDev, 3);
}

TEST_F(NetConnectMicrotest, SendConnect_PeerAccessAlreadyEnabled_Accepted) {
  BuildMap(1, 3, 0);
  ScopedHook peer(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipErrorPeerAccessAlreadyEnabled; });
  ScopedHook lastError(g_hipGetLastError, [] { return hipSuccess; });

  EXPECT_EQ(SendConnect(), ncclSuccess);
  EXPECT_EQ(lastError.calls, 1);  // clears the sticky error
}

TEST_F(NetConnectMicrotest, SendConnect_PeerAccessFails_ReturnsInternalError) {
  BuildMap(1, 3, 0);
  ScopedHook peer(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipErrorInvalidDevice; });

  EXPECT_EQ(SendConnect(), ncclInternalError);
}

TEST_F(NetConnectMicrotest, SendConnect_OtherProcess_ImportsHostAndDeviceMemory) {
  BuildMap(/*sameProcess=*/0, comm_.cudaDev, 0);
  std::vector<char> proxyHost = hostMem_;  // the proxy's pointers mean nothing in this process
  map_.mems[NCCL_NET_MAP_HOSTMEM].cpuPtr = map_.mems[NCCL_NET_MAP_HOSTMEM].gpuPtr = proxyHost.data();
  int shmRank = -1;
  ScopedHook shm(g_shmImportShareableBuffer, [&](ncclComm*, int rank, shmIpcDesc*, void** hptr, void** dptr,
                                                 shmIpcDesc*) {
    shmRank = rank;
    *hptr = *dptr = hostMem_.data();
    return ncclSuccess;
  });
  std::vector<char> imported(kSimpleBytes);
  int importRank = -1;
  size_t importSize = 0;
  ScopedHook dev(g_ncclP2pImportShareableBuffer, [&](ncclComm*, int rank, size_t size, ncclIpcDesc*, void** ptr,
                                                     void* ownerPtr, ncclMemType_t) {
    importRank = rank;
    importSize = size;
    EXPECT_EQ(ownerPtr, nullptr);
    *ptr = imported.data();
    return ncclSuccess;
  });

  ASSERT_EQ(SendConnect(), ncclSuccess);

  EXPECT_EQ(shmRank, 2);
  EXPECT_EQ(importRank, 2);
  EXPECT_EQ(importSize, kSimpleBytes);
  EXPECT_EQ(conn_.conn.head, &SendMem()->head);
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_SIMPLE], imported.data());
}

TEST_F(NetConnectMicrotest, SendConnect_CuMemOtherDevice_ImportsInsteadOfPeerAccess) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  BuildMap(1, 3, 0);
  ScopedHook peer(g_hipDeviceEnablePeerAccess, [](int, unsigned) { return hipSuccess; });
  ScopedHook shm(g_shmImportShareableBuffer, [](auto...) { return ncclInternalError; });
  std::vector<char> imported(kSimpleBytes);
  ScopedHook dev(g_ncclP2pImportShareableBuffer, [&](ncclComm*, int, size_t, ncclIpcDesc*, void** ptr, void*,
                                                     ncclMemType_t) {
    *ptr = imported.data();
    return ncclSuccess;
  });

  ASSERT_EQ(SendConnect(), ncclSuccess);

  EXPECT_EQ(peer.calls, 0);
  EXPECT_EQ(shm.calls, 0);
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_SIMPLE], imported.data());
}

TEST_F(NetConnectMicrotest, SendConnect_SharedDeviceMemory_ImportedOncePerLocalRank) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  BuildMap(1, /*cudaDev=*/3, /*shared=*/1, /*sharedDevBank=*/true);
  std::vector<char> pool(kSimpleBytes);
  ScopedHook dev(g_ncclP2pImportShareableBuffer, [&](ncclComm*, int, size_t, ncclIpcDesc*, void** ptr, void*,
                                                     ncclMemType_t) {
    *ptr = pool.data();
    return ncclSuccess;
  });

  ASSERT_EQ(SendConnect(), ncclSuccess);
  EXPECT_EQ(sharedDevMems_[1], pool.data());
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_SIMPLE], pool.data());

  free(conn_.transportResources);
  conn_.transportResources = nullptr;
  ASSERT_EQ(SendConnect(), ncclSuccess);
  EXPECT_EQ(dev.calls, 1);
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_SIMPLE], pool.data());
}

TEST_F(NetConnectMicrotest, SendConnect_ImportFails_ReturnsError) {
  BuildMap(0, comm_.cudaDev, 0);
  {
    ScopedHook shm(g_shmImportShareableBuffer, [](auto...) { return ncclSystemError; });
    EXPECT_EQ(SendConnect(), ncclSystemError);
  }

  free(conn_.transportResources);
  conn_.transportResources = nullptr;
  ScopedHook shm(g_shmImportShareableBuffer, [&](ncclComm*, int, shmIpcDesc*, void** hptr, void** dptr, shmIpcDesc*) {
    *hptr = *dptr = hostMem_.data();
    return ncclSuccess;
  });
  ScopedHook dev(g_ncclP2pImportShareableBuffer, [](auto...) { return ncclInternalError; });
  EXPECT_EQ(SendConnect(), ncclInternalError);
}

TEST_F(NetConnectMicrotest, Connect_ProxyNeedsProgress_InstallsProgressFunction) {
  proxyConnection_.needsProxyProgress = 1;
  ASSERT_EQ(SendConnect(), ncclSuccess);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, sendProxyProgress);

  free(conn_.transportResources);
  conn_.transportResources = nullptr;
  ASSERT_EQ(RecvConnect(), ncclSuccess);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, recvProxyProgress);
}

TEST_F(NetConnectMicrotest, Connect_DeviceOffloadedProxy_CopiesHandleAndSkipsProgress) {
  ncclNetDeviceHandle_t handle{};
  handle.netDeviceType = NCCL_NET_DEVICE_UNPACK;
  proxyConnection_.netDeviceHandle = &handle;
  proxyConnection_.mhandles[NCCL_PROTO_SIMPLE] = reinterpret_cast<void*>(0xAB);
  proxyConnection_.needsProxyProgress = 0;
  conn_.proxyConn.proxyProgress = reinterpret_cast<decltype(conn_.proxyConn.proxyProgress)>(0x1);

  ASSERT_EQ(SendConnect(), ncclSuccess);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, nullptr);
  EXPECT_EQ(conn_.conn.netDeviceHandle.netDeviceType, NCCL_NET_DEVICE_UNPACK);
  EXPECT_EQ(conn_.conn.mhandles[NCCL_PROTO_SIMPLE], reinterpret_cast<void*>(0xAB));

  free(conn_.transportResources);
  conn_ = {};
  conn_.proxyConn.connection = &proxyConnection_;
  conn_.proxyConn.sameProcess = 1;
  conn_.proxyConn.proxyProgress = reinterpret_cast<decltype(conn_.proxyConn.proxyProgress)>(0x1);
  ASSERT_EQ(RecvConnect(), ncclSuccess);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, nullptr);
  EXPECT_EQ(conn_.conn.netDeviceHandle.netDeviceType, NCCL_NET_DEVICE_UNPACK);
  EXPECT_EQ(conn_.conn.mhandles[NCCL_PROTO_SIMPLE], reinterpret_cast<void*>(0xAB));
}

TEST_F(NetConnectMicrotest, Connect_RemoteProxy_AlwaysUsesHostProgress) {
  BuildMap(0, comm_.cudaDev, 0);
  conn_.proxyConn.sameProcess = 0;
  ScopedHook shm(g_shmImportShareableBuffer, [&](ncclComm*, int, shmIpcDesc*, void** hptr, void** dptr, shmIpcDesc*) {
    *hptr = *dptr = hostMem_.data();
    return ncclSuccess;
  });
  ScopedHook dev(g_ncclP2pImportShareableBuffer, [&](ncclComm*, int, size_t, ncclIpcDesc*, void** ptr, void*,
                                                     ncclMemType_t) {
    *ptr = devMem_.data();
    return ncclSuccess;
  });

  ASSERT_EQ(SendConnect(), ncclSuccess);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, sendProxyProgress);

  free(conn_.transportResources);
  conn_.transportResources = nullptr;
  ASSERT_EQ(RecvConnect(), ncclSuccess);
  EXPECT_EQ(conn_.proxyConn.proxyProgress, recvProxyProgress);
}

TEST_F(NetConnectMicrotest, Connect_PeerWithoutGdr_ClearsDirectNic) {
  conn_.conn.flags = NCCL_DIRECT_NIC;
  ASSERT_EQ(SendConnect(), ncclSuccess);
  EXPECT_TRUE(conn_.conn.flags & NCCL_DIRECT_NIC);

  SetConnectInfoGdr(0);
  free(conn_.transportResources);
  conn_.transportResources = nullptr;
  ASSERT_EQ(SendConnect(), ncclSuccess);
  EXPECT_FALSE(conn_.conn.flags & NCCL_DIRECT_NIC);

  conn_.conn.flags = NCCL_DIRECT_NIC;
  free(conn_.transportResources);
  conn_.transportResources = nullptr;
  ASSERT_EQ(RecvConnect(), ncclSuccess);
  EXPECT_FALSE(conn_.conn.flags & NCCL_DIRECT_NIC);
}

TEST_F(NetConnectMicrotest, RecvConnect_FirstCall_PostsConnectRequestWithProxyRank) {
  int proxyRank = 7;
  memcpy(&connectInfo_, &proxyRank, sizeof(proxyRank));
  comm_.nChannels = 4;
  pollResult_ = ncclInProgress;

  EXPECT_EQ(RecvConnect(), ncclInProgress);

  EXPECT_EQ(asyncType_, ncclProxyMsgConnect);
  EXPECT_EQ(asyncReqSize_, (int)sizeof(netRecvConnectArgs));
  EXPECT_EQ(asyncRespSize_, (int)sizeof(connectMap));
  EXPECT_EQ(asyncOpId_, &conn_);
  auto* args = reinterpret_cast<netRecvConnectArgs*>(asyncReq_.data());
  EXPECT_EQ(args->proxyRank, 7);
  EXPECT_EQ(args->netAttr.recvCommAttr.maxFlowsPerPeer, 4);  // collective hints from the comm
}

TEST_F(NetConnectMicrotest, RecvConnect_ProxyFails_ReturnsError) {
  pollResult_ = ncclSystemError;

  EXPECT_EQ(RecvConnect(), ncclSystemError);
}

TEST_F(NetConnectMicrotest, RecvConnect_SharedMap_WiresConnectorWithOffsetFifo) {
  BuildMap(1, comm_.cudaDev, /*shared=*/1);

  ASSERT_EQ(RecvConnect(), ncclSuccess);

  EXPECT_EQ(conn_.conn.head, &SendMem()->head);
  EXPECT_EQ(conn_.conn.tail, &RecvMem()->tail);
  EXPECT_EQ(conn_.conn.connFifo, RecvMem()->connFifo);
  EXPECT_EQ(conn_.conn.stepSize, kStepBytes);
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_SIMPLE], devMem_.data());
  EXPECT_EQ(conn_.conn.buffs[NCCL_PROTO_LL128], nullptr);
  EXPECT_EQ(RecvMem()->connFifo[NCCL_STEPS - 1].mode, NCCL_MODE_OFFSET);
}

TEST_F(NetConnectMicrotest, RecvConnect_GdcSyncWord_UsedAsTail) {
  uint64_t gdc = 0;
  map_.mems[NCCL_NET_MAP_GDCMEM].gpuPtr = reinterpret_cast<char*>(&gdc);

  ASSERT_EQ(RecvConnect(), ncclSuccess);

  EXPECT_EQ(conn_.conn.tail, &gdc);
  EXPECT_EQ(conn_.conn.head, &SendMem()->head);
}

TEST_F(NetConnectMicrotest, Free_NoMap_ReturnsSuccess) {
  EXPECT_EQ(sendFree(&comm_, &conn_), ncclSuccess);
  EXPECT_EQ(recvFree(&comm_, &conn_), ncclSuccess);
}

TEST_F(NetConnectMicrotest, SendFree_SameProcessSameDevice_ReleasesOnlyTheMap) {
  ScopedHook ipcClose(g_hipIpcCloseMemHandle, [](void*) { return hipSuccess; });
  ScopedHook shmClose(g_shmIpcClose, [](shmIpcDesc*) { return ncclSuccess; });
  map_.cudaDev = g_currentDevice;
  conn_.transportResources = HeapMap();

  EXPECT_EQ(sendFree(&comm_, &conn_), ncclSuccess);
  conn_.transportResources = nullptr;

  EXPECT_EQ(ipcClose.calls, 0);
  EXPECT_EQ(shmClose.calls, 0);
}

TEST_F(NetConnectMicrotest, SendFree_OtherDeviceLegacyIpc_ClosesImportedHandle) {
  map_.cudaDev = 3;
  void* closed = nullptr;
  ScopedHook ipcClose(g_hipIpcCloseMemHandle, [&](void* p) {
    closed = p;
    return hipSuccess;
  });
  conn_.transportResources = HeapMap();

  EXPECT_EQ(sendFree(&comm_, &conn_), ncclSuccess);
  conn_.transportResources = nullptr;

  EXPECT_EQ(closed, devMem_.data());
}

TEST_F(NetConnectMicrotest, SendFree_OtherDeviceCuMem_ReleasesShareableBuffer) {
  map_.cudaDev = 3;
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ScopedHook ipcFree(g_ncclP2pFreeShareableBuffer, [](ncclIpcDesc*) { return ncclSuccess; });
  ScopedHook retain(g_hipMemRetainAllocationHandle, [](hipMemGenericAllocationHandle_t*, void*) { return hipSuccess; });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });
  ScopedHook range(g_hipMemGetAddressRange, [](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t p) {
    *base = p;
    *size = kSimpleBytes;
    return hipSuccess;
  });
  ScopedHook unmap(g_hipMemUnmap, [](void*, size_t) { return hipSuccess; });
  void* freed = nullptr;
  ScopedHook addrFree(g_hipMemAddressFree, [&](void* p, size_t) {
    freed = p;
    return hipSuccess;
  });
  conn_.transportResources = HeapMap();

  EXPECT_EQ(sendFree(&comm_, &conn_), ncclSuccess);
  conn_.transportResources = nullptr;

  EXPECT_EQ(ipcFree.calls, 1);
  EXPECT_EQ(freed, devMem_.data());
}

TEST_F(NetConnectMicrotest, SendFree_OtherProcess_ClosesHostShm) {
  map_.sameProcess = 0;
  map_.cudaDev = g_currentDevice;
  ScopedHook shmClose(g_shmIpcClose, [](shmIpcDesc*) { return ncclSuccess; });
  conn_.transportResources = HeapMap();

  EXPECT_EQ(sendFree(&comm_, &conn_), ncclSuccess);
  conn_.transportResources = nullptr;

  EXPECT_EQ(shmClose.calls, 1);
}

TEST_F(NetConnectMicrotest, RecvFree_ReleasesMap) {
  conn_.transportResources = HeapMap();

  EXPECT_EQ(recvFree(&comm_, &conn_), ncclSuccess);
  conn_.transportResources = nullptr;
}

// ---------------------------------------------------------------------------
// Proxy-side send connect (sendProxyConnect)
// ---------------------------------------------------------------------------

std::vector<int> g_exportedFds;
hsa_status_t FakeExportDmaBuf(const void*, size_t, int* fd, uint64_t* offset) {
  *fd = open("/dev/null", O_RDONLY);
  *offset = 0x40;
  g_exportedFds.push_back(*fd);
  return HSA_STATUS_SUCCESS;
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
// Builds with the cuMem DMA-BUF export: each export hands back a fresh fd, recorded here.
struct CuMemExport {
  void* ptr;
  size_t size;
  unsigned long long flags;
  int fd;
};
struct CuMemExportRecorder : std::vector<CuMemExport> {
  hipError_t result = hipSuccess;
  void Install() {
    g_hipMemGetHandleForAddressRange = [this](void* handle, hipDeviceptr_t ptr, size_t size, hipMemRangeHandleType,
                                              unsigned long long flags) {
      if (result != hipSuccess) return result;
      int fd = open("/dev/null", O_RDONLY);
      *static_cast<int*>(handle) = fd;
      push_back({ptr, size, flags, fd});
      return hipSuccess;
    };
  }
  // Closes what the code under test left open, e.g. a bank fd no free released.
  void CloseOpen() {
    for (const auto& e : *this)
      if (fcntl(e.fd, F_GETFD) != -1) close(e.fd);
  }
};
#endif

// Shared by the send and recv proxy-connect suites; Res is that side's resources type.
template <typename Res>
class NetProxyConnectMicrotest : public NetAllocMicrotestBase {
 protected:
  static constexpr int kSizes[NCCL_NUM_PROTOCOLS] = {1 << 12, 1 << 13, 1 << 14};  // LL, LL128, SIMPLE
  std::deque<ncclProxyConnection> conns_;
  connectMap resp_{};
  int done_ = -1;
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  CuMemExportRecorder exports_;
#endif

  void SetUp() override {
    NetAllocMicrotestBase::SetUp();
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
    exports_.Install();
#endif
    proxyState_.tpLocalnRanks = 4;
    proxyState_.tpnRanks = 8;
    proxyState_.p2pnChannels = 2;
    proxyState_.p2pChunkSize = 1 << 10;
    for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) proxyState_.buffSizes[p] = kSizes[p];
    rcclPrimaryNicInfo = {rcclIBNicTypeDefault, 0, 0};
    g_exportedFds.clear();
  }

  void TearDown() override {
    for (auto& c : conns_) {
      auto* r = static_cast<Res*>(c.transportResources);
      if (r) {
        free(r->gdrDesc);
        if (r->netDeviceHandle != net_.offloadHandle) free(r->netDeviceHandle);
      }
      free(r);
    }
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
    exports_.CloseOpen();
#endif
    NetAllocMicrotestBase::TearDown();
  }

  // A connection whose proxy setup already ran: the resources the proxy connect reads.
  Res* AddConn(int shared = 0, int useGdr = 0, int tpLocalRank = 1) {
    conns_.emplace_back();
    conns_.back().sameProcess = 1;
    auto* r = static_cast<Res*>(calloc(1, sizeof(Res)));
    r->tpLocalRank = tpLocalRank;
    r->tpRemoteRank = 5;
    r->netDev = 2;
    r->channelId = 3;
    r->maxRecvs = 1;
    r->sameDevice = 1;
    r->shared = shared;
    r->useGdr = static_cast<ncclTopoGdrMode>(useGdr);
    conns_.back().transportResources = r;
    return r;
  }

  // Marks connection i connected, frees it, and drops the resources pointer the free released.
  template <typename FreeFn>
  ncclResult_t FreeConn(size_t i, FreeFn freeFn) {
    conns_[i].state = connConnected;
    ncclResult_t ret = freeFn(&conns_[i], &proxyState_);
    if (ret == ncclSuccess) conns_[i].transportResources = nullptr;
    return ret;
  }

  static char* Buffer(Res* r, int p) { return NCCL_NET_MAP_GET_POINTER(&r->map, cpu, buffs[p]); }
  static bool InDeviceMemory(Res* r, int p) { return NCCL_NET_MAP_DEV_MEM(&r->map, buffs[p]); }
  int RegMrType(void* data) {
    for (const auto& m : net_.regMrs)
      if (m.data == data) return m.type;
    return 0;
  }
};

class NetSendProxyConnectMicrotest : public NetProxyConnectMicrotest<sendNetResources> {
 protected:
  netSendConnectArgs args_{};

  void SetUp() override {
    NetProxyConnectMicrotest::SetUp();
    memset(args_.handle, 0x5A, sizeof(args_.handle));
    args_.netAttr = NCCL_NET_ATTR_INIT;
    args_.netAttr.op = BIT(ncclFuncSend);
  }

  ncclResult_t Connect(size_t i = 0, int respSize = sizeof(connectMap)) {
    done_ = -1;
    return sendProxyConnect(&conns_[i], &proxyState_, &args_, sizeof(args_), &resp_, respSize, &done_);
  }
};

TEST_F(NetSendProxyConnectMicrotest, WrongRequestSize_Rejected) {
  AddConn();

  EXPECT_EQ(sendProxyConnect(&conns_[0], &proxyState_, &args_, sizeof(args_) - 1, &resp_, sizeof(resp_), &done_),
            ncclInternalError);
  EXPECT_EQ(net_.connectCalls, 0);
}

TEST_F(NetSendProxyConnectMicrotest, Dedicated_ConnectsToPeerHandleWithHints) {
  sendNetResources* r = AddConn();

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(net_.setNetAttrCalls, 1);
  EXPECT_EQ(net_.lastAttr.op, BIT(ncclFuncSend));
  EXPECT_EQ(net_.connectCalls, 1);
  EXPECT_EQ(net_.lastConnectCtx, &net_);
  EXPECT_EQ(net_.lastConnectDev, 2);
  EXPECT_EQ(memcmp(net_.lastConnectHandle, args_.handle, sizeof(args_.handle)), 0);
  EXPECT_EQ(net_.lastDevHandleArg, &r->netDeviceHandle);
  EXPECT_EQ(r->netSendComm, &net_.sendComm);
  EXPECT_EQ(conns_[0].proxyAppendPtr, &conns_[0].proxyAppend);
  EXPECT_EQ(conns_[0].needsProxyProgress, 1);
  EXPECT_EQ(net_.getDeviceMrCalls, 0);
}

TEST_F(NetSendProxyConnectMicrotest, StillConnecting_ReportsInProgressWithoutBuffers) {
  AddConn();
  net_.commReady = false;

  EXPECT_EQ(Connect(), ncclInProgress);
  EXPECT_EQ(done_, 0);
  EXPECT_EQ(hostAllocs_, 0);

  net_.commReady = true;
  EXPECT_EQ(Connect(), ncclSuccess);
  EXPECT_EQ(done_, 1);
}

TEST_F(NetSendProxyConnectMicrotest, ConnectFails_ClosesPartialComm) {
  AddConn();
  net_.connectResult = ncclRemoteError;

  EXPECT_EQ(Connect(), ncclRemoteError);
  ASSERT_EQ(net_.closedSends.size(), 1u);
  EXPECT_EQ(net_.closedSends[0], &net_.sendComm);

  AddConn();
  net_.commReady = false;
  EXPECT_EQ(Connect(1), ncclRemoteError);
  EXPECT_EQ(net_.closedSends.size(), 1u);
}

TEST_F(NetSendProxyConnectMicrotest, Ainic_PassesChannelToPlugin) {
  rcclPrimaryNicInfo.type = rcclIBNicTypeAINIC;
  sendNetResources* dedicated = AddConn(/*shared=*/0);
  sendNetResources* shared = AddConn(/*shared=*/1);
  sendNetResources* sharedComms = AddConn(/*shared=*/1, 0, /*tpLocalRank=*/2);
  sharedComms->maxRecvs = 2;
  sendNetResources* res[] = {dedicated, shared, sharedComms};

  for (size_t i = 0; i < 3; i++) {
    SCOPED_TRACE(i);
    ASSERT_EQ(Connect(i), ncclSuccess);
    EXPECT_NE(net_.lastDevHandleArg, &res[i]->netDeviceHandle);
    EXPECT_EQ(net_.lastDevHandleWord, 3u);
  }
}

TEST_F(NetSendProxyConnectMicrotest, IbPlugins_ApplyP2pPolicy) {
  sendNetResources* r = AddConn();
  r->isP2p = 1;
  struct RestoreIbPlugins {
    ncclNet_t ib = ncclNetIb, cast = netIbCast;
    ~RestoreIbPlugins() {
      ncclNetIb = ib;
      netIbCast = cast;
    }
  } restore;
  ncclNetIb = netIbCast = vtable_;
  void* policyHandle = nullptr;
  int policyP2p = -1;
  ScopedHook ib(g_rcclNetP2pPolicy, [&](void* h, int p2p) {
    policyHandle = h;
    policyP2p = p2p;
    return ncclSuccess;
  });
  ScopedHook cast(g_rcclCastNetP2pPolicy, [](void*, int) { return ncclSuccess; });

  proxyState_.ncclNet = &ncclNetIb;
  EXPECT_EQ(Connect(), ncclSuccess);
  EXPECT_EQ(policyHandle, args_.handle);
  EXPECT_EQ(policyP2p, 1);
  EXPECT_EQ(cast.calls, 0);

  AddConn();
  proxyState_.ncclNet = &netIbCast;
  EXPECT_EQ(Connect(1), ncclSuccess);
  EXPECT_EQ(cast.calls, 1);
  EXPECT_EQ(ib.calls, 1);
}

TEST_F(NetSendProxyConnectMicrotest, DedicatedHostBuffers_RegistersEveryProtocol) {
  sendNetResources* r = AddConn();
  ScopedHook dev(g_hipGetDevice, [](int* d) {
    *d = 2;
    return hipSuccess;
  });

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(net_.regMrs.size(), (size_t)NCCL_NUM_PROTOCOLS);
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    SCOPED_TRACE(p);
    ASSERT_NE(Buffer(r, p), nullptr);
    EXPECT_FALSE(InDeviceMemory(r, p));
    EXPECT_EQ(r->buffSizes[p], kSizes[p]);
    EXPECT_EQ(RegMrType(Buffer(r, p)), NCCL_PTR_HOST);
    EXPECT_EQ(r->mhandles[p], Buffer(r, p));
  }
  EXPECT_EQ(net_.regMrs[0].comm, &net_.sendComm);
  EXPECT_EQ(net_.regMrs[0].size, (size_t)kSizes[NCCL_PROTO_LL]);
  EXPECT_EQ(hostAllocs_, 1);
  EXPECT_EQ(devAllocs_, 0);
  EXPECT_EQ(r->sendMem->head, 0u);
  EXPECT_EQ(r->recvMem->connFifo[NCCL_STEPS - 1].size, -1);
  EXPECT_EQ(resp_.cudaDev, 2);
  EXPECT_EQ(resp_.sameProcess, 1);
  EXPECT_EQ(memcmp(&resp_, &r->map, sizeof(resp_)), 0);
}

TEST_F(NetSendProxyConnectMicrotest, DedicatedGdr_KeepsOnlyLlInHostMemory) {
  sendNetResources* r = AddConn(0, /*useGdr=*/1);

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_FALSE(InDeviceMemory(r, NCCL_PROTO_LL));
  EXPECT_TRUE(InDeviceMemory(r, NCCL_PROTO_LL128));
  EXPECT_TRUE(InDeviceMemory(r, NCCL_PROTO_SIMPLE));
  EXPECT_EQ(RegMrType(Buffer(r, NCCL_PROTO_LL)), NCCL_PTR_HOST);
  EXPECT_EQ(RegMrType(Buffer(r, NCCL_PROTO_LL128)), NCCL_PTR_CUDA);
  EXPECT_EQ(RegMrType(Buffer(r, NCCL_PROTO_SIMPLE)), NCCL_PTR_CUDA);
  EXPECT_EQ(devAllocs_, 1);
  EXPECT_EQ(shareableAllocs_, 0);
}

TEST_F(NetSendProxyConnectMicrotest, DedicatedGdrShareable_AllocatesIpcAlignedBank) {
  sendNetResources* r = AddConn(0, 1);
  conns_[0].sameProcess = 0;
  ScopedHook shm(g_shmAllocateShareableBuffer, [this](size_t n, bool, void*, void** h, void** d) {
    *h = *d = Alloc(n);
    return ncclSuccess;
  });
  size_t allocated = 0;
  ScopedHook alloc(g_ncclP2pAllocateShareableBuffer, [&](size_t n, int, ncclIpcDesc*, void** p, int,
                                                         ncclMemManager*, ncclMemType_t) {
    allocated = n;
    *p = Alloc(n);
    return ncclSuccess;
  });

  ASSERT_EQ(Connect(), ncclSuccess);
  EXPECT_EQ(allocated % CUDA_IPC_MIN, 0u);
  EXPECT_GE(allocated, (size_t)kSizes[NCCL_PROTO_LL128] + kSizes[NCCL_PROTO_SIMPLE]);
  EXPECT_EQ(devAllocs_, 0);

  sendNetResources* cuMemConn = AddConn(0, 1);
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ASSERT_EQ(Connect(1), ncclSuccess);
  EXPECT_EQ(alloc.calls, 2);
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  ASSERT_EQ(exports_.size(), 1u);  // cuMem banks are always exported for DMA-BUF registration
  EXPECT_EQ(cuMemConn->map.mems[NCCL_NET_MAP_DEVMEM].dmaBufFd, exports_[0].fd);
#else
  (void)cuMemConn;
#endif
}

TEST_F(NetSendProxyConnectMicrotest, RemoteProxy_ZeroesShmControlWords) {
  sendNetResources* r = AddConn();
  conns_[0].sameProcess = 0;
  ScopedHook shm(g_shmAllocateShareableBuffer, [this](size_t n, bool, void*, void** h, void** d) {
    *h = *d = Alloc(n);
    memset(*h, 0xFF, n);
    return ncclSuccess;
  });

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(shm.calls, 1);
  EXPECT_EQ(hostAllocs_, 0);
  EXPECT_EQ(resp_.sameProcess, 0);
  EXPECT_EQ(r->sendMem->head, 0u);
  EXPECT_EQ(r->sendMem->ptrExchange, nullptr);
  EXPECT_EQ(r->recvMem->tail, 0u);
}

TEST_F(NetSendProxyConnectMicrotest, SharedBuffers_UsesPoolAndWithholdsCredits) {
  sendNetResources* r = AddConn(/*shared=*/1);

  ASSERT_EQ(Connect(), ncclSuccess);

  ncclProxyPeer* peer = proxyState_.progressState.localPeers[1];
  ASSERT_NE(peer, nullptr);
  EXPECT_EQ(conns_[0].proxyAppendPtr, peer->send.proxyAppend + 3);
  EXPECT_EQ(Buffer(r, NCCL_PROTO_SIMPLE), peer->send.hostBuff);
  EXPECT_EQ(r->buffSizes[NCCL_PROTO_SIMPLE], peer->send.size);
  EXPECT_EQ(Buffer(r, NCCL_PROTO_LL), nullptr);
  EXPECT_EQ(Buffer(r, NCCL_PROTO_LL128), nullptr);
  EXPECT_EQ(net_.regMrs.size(), 1u);
  EXPECT_EQ(r->sendMem->head, (uint64_t)-NCCL_STEPS);
  EXPECT_EQ(proxyState_.progressState.netComms[2], nullptr);  // one grouped recv: no shared comm
}

TEST_F(NetSendProxyConnectMicrotest, SharedGdrWithLlBuffers_StagesLlAndLl128) {
  sendNetResources* r = AddConn(1, /*useGdr=*/1);
  proxyState_.allocP2pNetLLBuffers = true;

  ASSERT_EQ(Connect(), ncclSuccess);

  ncclProxyPeer* peer = proxyState_.progressState.localPeers[1];
  EXPECT_EQ(Buffer(r, NCCL_PROTO_SIMPLE), peer->send.cudaBuff);
  EXPECT_FALSE(InDeviceMemory(r, NCCL_PROTO_LL));
  EXPECT_TRUE(InDeviceMemory(r, NCCL_PROTO_LL128));
  EXPECT_EQ(r->buffSizes[NCCL_PROTO_LL], kSizes[NCCL_PROTO_LL]);
  EXPECT_EQ(r->buffSizes[NCCL_PROTO_LL128], kSizes[NCCL_PROTO_LL128]);
  EXPECT_EQ(devAllocs_, 2);  // the pool and the LL128 staging bank
  EXPECT_EQ(net_.regMrs.size(), (size_t)NCCL_NUM_PROTOCOLS);
}

TEST_F(NetSendProxyConnectMicrotest, SharedComms_OneLocalRankConnectsOthersReuse) {
  AddConn(1, 0, /*tpLocalRank=*/1)->maxRecvs = 2;
  AddConn(1, 0, /*tpLocalRank=*/2)->maxRecvs = 2;
  AddConn(1, 0, /*tpLocalRank=*/1)->maxRecvs = 2;

  ASSERT_EQ(Connect(0), ncclSuccess);
  ASSERT_EQ(Connect(1), ncclSuccess);
  ASSERT_EQ(Connect(2), ncclSuccess);

  EXPECT_EQ(net_.connectCalls, 1);
  ncclSharedNetComms* comms = proxyState_.progressState.netComms[2] + 5;
  EXPECT_EQ(comms->sendComm[3], &net_.sendComm);
  EXPECT_EQ(comms->sendRefCount[3], 3);
  EXPECT_EQ(static_cast<sendNetResources*>(conns_[1].transportResources)->netSendComm, &net_.sendComm);
}

TEST_F(NetSendProxyConnectMicrotest, SharedCommsPeerStillConnecting_Waits) {
  AddConn(1, 0, 1)->maxRecvs = 2;
  AddConn(1, 0, 2)->maxRecvs = 2;
  net_.commReady = false;

  EXPECT_EQ(Connect(0), ncclInProgress);
  EXPECT_EQ(Connect(1), ncclInProgress);
  EXPECT_EQ(net_.connectCalls, 1);

  net_.commReady = true;
  EXPECT_EQ(Connect(0), ncclSuccess);
  EXPECT_EQ(Connect(1), ncclSuccess);
  EXPECT_EQ(net_.connectCalls, 2);
}

TEST_F(NetSendProxyConnectMicrotest, SharedCommsDisabled_ConnectsPerConnection) {
  ScopedHook param(g_loadParam, ParamOverride({{"NET_SHARED_COMMS", 0}}));
  AddConn(1, 0, 1)->maxRecvs = 2;
  AddConn(1, 0, 2)->maxRecvs = 2;

  ASSERT_EQ(Connect(0), ncclSuccess);
  ASSERT_EQ(Connect(1), ncclSuccess);

  EXPECT_EQ(net_.connectCalls, 2);
  EXPECT_EQ(proxyState_.progressState.netComms[2], nullptr);
}

TEST_F(NetSendProxyConnectMicrotest, GdrCopy_HeadLivesInGdcMemory) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  sendNetResources* r = AddConn(/*shared=*/1);

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_NE(r->gdcSync, nullptr);
  EXPECT_EQ(*r->gdcSync, (uint64_t)-NCCL_STEPS);
  EXPECT_EQ(r->sendMem->head, 0u);
  EXPECT_EQ(resp_.mems[NCCL_NET_MAP_GDCMEM].cpuPtr, reinterpret_cast<char*>(r->gdcSync));
  EXPECT_EQ(resp_.mems[NCCL_NET_MAP_GDCMEM].size, (ssize_t)sizeof(uint64_t));
}

TEST_F(NetSendProxyConnectMicrotest, GdrCopyNotApplicable_HeadStaysInHostMemory) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  AddConn()->sameDevice = 0;
  {
    ScopedHook param(g_loadParam, ParamOverride({{"GDRCOPY_SYNC_ENABLE", 0}}));
    AddConn();
    ASSERT_EQ(Connect(1), ncclSuccess);
  }
  ASSERT_EQ(Connect(0), ncclSuccess);

  for (size_t i = 0; i < 2; i++)
    EXPECT_EQ(static_cast<sendNetResources*>(conns_[i].transportResources)->gdcSync, nullptr);
}

TEST_F(NetSendProxyConnectMicrotest, DeviceOffload_ExportsDeviceMhandles) {
  ncclNetDeviceHandle_t handle{};
  handle.needsProxyProgress = 0;
  net_.offloadHandle = &handle;
  sendNetResources* r = AddConn();

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(conns_[0].netDeviceHandle, &handle);
  EXPECT_EQ(conns_[0].needsProxyProgress, 0);
  EXPECT_EQ(net_.getDeviceMrCalls, NCCL_NUM_PROTOCOLS);
  EXPECT_EQ(conns_[0].mhandles[NCCL_PROTO_SIMPLE], Buffer(r, NCCL_PROTO_SIMPLE) + 1);

  vtable_.getDeviceMr = nullptr;
  AddConn();
  ASSERT_EQ(Connect(1), ncclSuccess);
  EXPECT_EQ(net_.getDeviceMrCalls, NCCL_NUM_PROTOCOLS);
}

TEST_F(NetSendProxyConnectMicrotest, HsaDmaBuf_RegistersDeviceBuffersByFd) {
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  proxyState_.dmaBufSupport = true;
  sendNetResources* r = AddConn(0, 1);
  r->useDmaBuf = 1;

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(net_.regMrDmaBufs.size(), 2u);  // LL128 and SIMPLE; LL stays in host memory
  EXPECT_EQ(RegMrType(Buffer(r, NCCL_PROTO_LL)), NCCL_PTR_HOST);
  for (const auto& m : net_.regMrDmaBufs) {
    EXPECT_EQ(m.type, NCCL_PTR_CUDA);
    EXPECT_EQ(m.offset, 0x40u);
  }
  EXPECT_EQ(net_.regMrDmaBufs[1].data, Buffer(r, NCCL_PROTO_SIMPLE));
  EXPECT_EQ(net_.regMrDmaBufs[1].fd, g_exportedFds[1]);
  for (int fd : g_exportedFds) EXPECT_EQ(fcntl(fd, F_GETFD), -1) << "fd " << fd << " left open";
}

TEST_F(NetSendProxyConnectMicrotest, DmaBufUnavailable_FallsBackToRegMr) {
  proxyState_.dmaBufSupport = true;
  AddConn(0, 1)->useDmaBuf = 1;  // no HSA export entry point
  ASSERT_EQ(Connect(0), ncclSuccess);

  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  proxyState_.dmaBufSupport = false;
  AddConn(0, 1)->useDmaBuf = 1;
  ASSERT_EQ(Connect(1), ncclSuccess);

  EXPECT_TRUE(net_.regMrDmaBufs.empty());
  EXPECT_EQ(net_.regMrs.size(), 2u * NCCL_NUM_PROTOCOLS);
}

TEST_F(NetSendProxyConnectMicrotest, DmaBufExportFails_ReturnsError) {
  pfn_hsa_amd_portable_export_dmabuf = [](const void*, size_t, int*, uint64_t*) { return HSA_STATUS_ERROR; };
  proxyState_.dmaBufSupport = true;
  AddConn(0, 1)->useDmaBuf = 1;

  EXPECT_EQ(Connect(), ncclUnhandledCudaError);
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
TEST_F(NetSendProxyConnectMicrotest, CuMemDmaBuf_RegistersDeviceBuffersAgainstBankFd) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  sendNetResources* r = AddConn(0, ncclTopoGdrModePci);
  r->useDmaBuf = 1;

  ASSERT_EQ(Connect(), ncclSuccess);

  connectMapMem& bank = r->map.mems[NCCL_NET_MAP_DEVMEM];
  ASSERT_EQ(exports_.size(), 1u);
  EXPECT_EQ(exports_[0].ptr, bank.gpuPtr);
  EXPECT_EQ(exports_[0].size, (size_t)bank.size);
  EXPECT_EQ(exports_[0].flags, (unsigned long long)hipMemRangeFlagDmaBufMappingTypePcie);
  ASSERT_EQ(net_.regMrDmaBufs.size(), 2u);  // LL128 and SIMPLE
  for (const auto& m : net_.regMrDmaBufs) {
    EXPECT_EQ(m.fd, bank.dmaBufFd);
    EXPECT_EQ(m.offset, (uint64_t)(static_cast<char*>(m.data) - bank.gpuPtr));
  }
  EXPECT_EQ(RegMrType(Buffer(r, NCCL_PROTO_LL)), NCCL_PTR_HOST);
}

TEST_F(NetSendProxyConnectMicrotest, CuMemDmaBufPoolBuffer_ExportedAndClosedOnItsOwn) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  sendNetResources* r = AddConn(/*shared=*/1, ncclTopoGdrModeDefault);
  r->useDmaBuf = 1;

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(exports_.size(), 1u);  // no dedicated bank, so only the pool buffer
  EXPECT_EQ(exports_[0].ptr, Buffer(r, NCCL_PROTO_SIMPLE));
  EXPECT_EQ(exports_[0].flags, 0u);
  ASSERT_EQ(net_.regMrDmaBufs.size(), 1u);
  EXPECT_EQ(net_.regMrDmaBufs[0].fd, exports_[0].fd);
  EXPECT_EQ(net_.regMrDmaBufs[0].offset, 0u);
  EXPECT_EQ(fcntl(exports_[0].fd, F_GETFD), -1);
}

TEST_F(NetSendProxyConnectMicrotest, CuMemDmaBufPoolBesideStagingBank_ExportedSeparately) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  proxyState_.allocP2pNetLLBuffers = true;
  sendNetResources* r = AddConn(/*shared=*/1, ncclTopoGdrModeDefault);
  r->useDmaBuf = 1;

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(exports_.size(), 2u);  // the LL128 staging bank, then the pool buffer
  ASSERT_EQ(net_.regMrDmaBufs.size(), 2u);
  EXPECT_EQ(net_.regMrDmaBufs[0].data, Buffer(r, NCCL_PROTO_LL128));
  EXPECT_EQ(net_.regMrDmaBufs[0].fd, exports_[0].fd);
  EXPECT_EQ(net_.regMrDmaBufs[1].data, Buffer(r, NCCL_PROTO_SIMPLE));
  EXPECT_EQ(net_.regMrDmaBufs[1].fd, exports_[1].fd);
  EXPECT_EQ(net_.regMrDmaBufs[1].offset, 0u);
}

TEST_F(NetSendProxyConnectMicrotest, CuMemBankExportFails_ReturnsError) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  exports_.result = hipErrorNotSupported;
  AddConn(0, 1);

  EXPECT_EQ(Connect(), ncclUnhandledCudaError);
}
#endif

TEST_F(NetSendProxyConnectMicrotest, RegistrationFails_ReturnsError) {
  AddConn();
  net_.regMrResult = ncclSystemError;

  EXPECT_EQ(Connect(), ncclSystemError);
}

TEST_F(NetSendProxyConnectMicrotest, WrongResponseSize_Rejected) {
  AddConn();

  EXPECT_EQ(Connect(0, sizeof(connectMap) - 1), ncclInternalError);
}

// ---------------------------------------------------------------------------
// Proxy-side recv connect (recvProxyConnect)
// ---------------------------------------------------------------------------

class NetRecvProxyConnectMicrotest : public NetProxyConnectMicrotest<recvNetResources> {
 protected:
  netRecvConnectArgs args_{};
  char listenComm_ = 0;

  void SetUp() override {
    NetProxyConnectMicrotest::SetUp();
    args_.proxyRank = 6;
    args_.netAttr = NCCL_NET_ATTR_INIT;
    args_.netAttr.op = BIT(ncclFuncRecv);
  }

  recvNetResources* AddRecvConn(int shared = 0, int useGdr = 0, int tpLocalRank = 1) {
    recvNetResources* r = AddConn(shared, useGdr, tpLocalRank);
    r->netListenComm = &listenComm_;
    return r;
  }

  ncclResult_t Connect(size_t i = 0, int respSize = sizeof(connectMap)) {
    done_ = -1;
    return recvProxyConnect(&conns_[i], &proxyState_, &args_, sizeof(args_), &resp_, respSize, &done_);
  }
};

TEST_F(NetRecvProxyConnectMicrotest, WrongRequestSize_Rejected) {
  AddRecvConn();

  EXPECT_EQ(recvProxyConnect(&conns_[0], &proxyState_, &args_, sizeof(args_) + 1, &resp_, sizeof(resp_), &done_),
            ncclInternalError);
  EXPECT_EQ(net_.acceptCalls, 0);
}

TEST_F(NetRecvProxyConnectMicrotest, Dedicated_AcceptsOnListenCommAndClosesIt) {
  recvNetResources* r = AddRecvConn();

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(done_, 1);
  EXPECT_EQ(r->tpRemoteProxyRank, 6);
  EXPECT_EQ(net_.setNetAttrCalls, 1);
  EXPECT_EQ(net_.lastAttr.op, BIT(ncclFuncRecv));
  EXPECT_EQ(net_.lastAcceptListen, &listenComm_);
  EXPECT_EQ(net_.lastDevHandleArg, &r->netDeviceHandle);
  EXPECT_EQ(r->netRecvComm, &net_.recvComm);
  ASSERT_EQ(net_.closedListens.size(), 1u);
  EXPECT_EQ(net_.closedListens[0], &listenComm_);
  EXPECT_EQ(conns_[0].proxyAppendPtr, &conns_[0].proxyAppend);
  EXPECT_EQ(conns_[0].needsProxyProgress, 1);
  EXPECT_EQ(net_.getDeviceMrCalls, 0);
}

TEST_F(NetRecvProxyConnectMicrotest, StillAccepting_ReportsInProgressAndKeepsListening) {
  AddRecvConn();
  net_.commReady = false;

  EXPECT_EQ(Connect(), ncclInProgress);

  EXPECT_EQ(done_, 0);
  EXPECT_TRUE(net_.closedListens.empty());
  EXPECT_EQ(hostAllocs_, 0);
}

TEST_F(NetRecvProxyConnectMicrotest, AcceptFails_ReturnsError) {
  AddRecvConn();
  net_.acceptResult = ncclRemoteError;

  EXPECT_EQ(Connect(), ncclRemoteError);
  EXPECT_TRUE(net_.closedListens.empty());
}

TEST_F(NetRecvProxyConnectMicrotest, Ainic_PassesChannelToPlugin) {
  rcclPrimaryNicInfo.type = rcclIBNicTypeAINIC;
  recvNetResources* dedicated = AddRecvConn(0);
  recvNetResources* shared = AddRecvConn(1);
  recvNetResources* sharedComms = AddRecvConn(1, 0, 2);
  sharedComms->maxRecvs = 2;
  recvNetResources* res[] = {dedicated, shared, sharedComms};

  for (size_t i = 0; i < 3; i++) {
    SCOPED_TRACE(i);
    ASSERT_EQ(Connect(i), ncclSuccess);
    EXPECT_NE(net_.lastDevHandleArg, &res[i]->netDeviceHandle);
    EXPECT_EQ(net_.lastDevHandleWord, 3u);
  }
}

TEST_F(NetRecvProxyConnectMicrotest, RemoteProxy_Rejected) {
  AddRecvConn();
  conns_[0].sameProcess = 0;

  EXPECT_EQ(Connect(), ncclInternalError);
  EXPECT_EQ(hostAllocs_, 0);
}

TEST_F(NetRecvProxyConnectMicrotest, DedicatedHostBuffers_RegistersEveryProtocol) {
  recvNetResources* r = AddRecvConn();

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(net_.regMrs.size(), (size_t)NCCL_NUM_PROTOCOLS);
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    SCOPED_TRACE(p);
    ASSERT_NE(Buffer(r, p), nullptr);
    EXPECT_FALSE(InDeviceMemory(r, p));
    EXPECT_EQ(r->buffSizes[p], kSizes[p]);
    EXPECT_EQ(RegMrType(Buffer(r, p)), NCCL_PTR_HOST);
    EXPECT_EQ(r->mhandles[p], Buffer(r, p));
  }
  EXPECT_EQ(net_.regMrs[0].comm, &net_.recvComm);
  EXPECT_EQ(net_.regMrs[0].size, (size_t)kSizes[NCCL_PROTO_LL]);
  EXPECT_EQ(hostAllocs_, 1);
  EXPECT_EQ(r->recvMem->connFifo[NCCL_STEPS - 1].size, -1);
  EXPECT_EQ(resp_.sameProcess, 1);
  EXPECT_EQ(memcmp(&resp_, &r->map, sizeof(resp_)), 0);
}

TEST_F(NetRecvProxyConnectMicrotest, DedicatedGdr_PutsEveryProtocolInDeviceMemory) {
  recvNetResources* r = AddRecvConn(0, 1);

  ASSERT_EQ(Connect(), ncclSuccess);

  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    SCOPED_TRACE(p);
    EXPECT_TRUE(InDeviceMemory(r, p));
    EXPECT_EQ(RegMrType(Buffer(r, p)), NCCL_PTR_CUDA);
  }
  EXPECT_EQ(devAllocs_, 1);
  EXPECT_EQ(shareableAllocs_, 0);
}

TEST_F(NetRecvProxyConnectMicrotest, DedicatedGdrCuMem_AllocatesShareableBank) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  size_t allocated = 0;
  ScopedHook alloc(g_ncclP2pAllocateShareableBuffer, [&](size_t n, int, ncclIpcDesc*, void** p, int,
                                                         ncclMemManager*, ncclMemType_t) {
    allocated = n;
    *p = Alloc(n);
    return ncclSuccess;
  });
  recvNetResources* r = AddRecvConn(0, 1);

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(allocated, (size_t)kSizes[0] + kSizes[1] + kSizes[2]);
  EXPECT_EQ(devAllocs_, 0);
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  ASSERT_EQ(exports_.size(), 1u);  // cuMem banks are always exported for DMA-BUF registration
  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_DEVMEM].dmaBufFd, exports_[0].fd);
#else
  (void)r;
#endif
}

TEST_F(NetRecvProxyConnectMicrotest, SharedBuffers_UsesRecvPool) {
  recvNetResources* r = AddRecvConn(1);

  ASSERT_EQ(Connect(), ncclSuccess);

  ncclProxyPeer* peer = proxyState_.progressState.localPeers[1];
  ASSERT_NE(peer, nullptr);
  EXPECT_EQ(conns_[0].proxyAppendPtr, peer->recv.proxyAppend + 3);
  EXPECT_EQ(Buffer(r, NCCL_PROTO_SIMPLE), peer->recv.hostBuff);
  EXPECT_EQ(r->buffSizes[NCCL_PROTO_SIMPLE], peer->recv.size);
  EXPECT_EQ(Buffer(r, NCCL_PROTO_LL), nullptr);
  EXPECT_EQ(Buffer(r, NCCL_PROTO_LL128), nullptr);
  EXPECT_EQ(net_.regMrs.size(), 1u);
  EXPECT_EQ(proxyState_.progressState.netComms[2], nullptr);
}

TEST_F(NetRecvProxyConnectMicrotest, SharedGdrWithLlBuffers_StagesLlAndLl128) {
  recvNetResources* r = AddRecvConn(1, 1);
  proxyState_.allocP2pNetLLBuffers = true;

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(Buffer(r, NCCL_PROTO_SIMPLE), proxyState_.progressState.localPeers[1]->recv.cudaBuff);
  EXPECT_FALSE(InDeviceMemory(r, NCCL_PROTO_LL));
  EXPECT_TRUE(InDeviceMemory(r, NCCL_PROTO_LL128));
  EXPECT_EQ(r->buffSizes[NCCL_PROTO_LL], kSizes[NCCL_PROTO_LL]);
  EXPECT_EQ(r->buffSizes[NCCL_PROTO_LL128], kSizes[NCCL_PROTO_LL128]);
  EXPECT_EQ(devAllocs_, 2);
  EXPECT_EQ(net_.regMrs.size(), (size_t)NCCL_NUM_PROTOCOLS);
}

TEST_F(NetRecvProxyConnectMicrotest, DedicatedWithLlBuffersFlag_AddsNoStaging) {
  recvNetResources* r = AddRecvConn(0, 1);
  proxyState_.allocP2pNetLLBuffers = true;

  ASSERT_EQ(Connect(), ncclSuccess);

  EXPECT_EQ(r->map.mems[NCCL_NET_MAP_DEVMEM].size, (ssize_t)(kSizes[0] + kSizes[1] + kSizes[2]));
}

TEST_F(NetRecvProxyConnectMicrotest, SharedComms_KeyedByProxyRankAndReused) {
  AddRecvConn(1, 0, 1)->maxRecvs = 2;
  AddRecvConn(1, 0, 2)->maxRecvs = 2;
  AddRecvConn(1, 0, 1)->maxRecvs = 2;

  ASSERT_EQ(Connect(0), ncclSuccess);
  ASSERT_EQ(Connect(1), ncclSuccess);
  ASSERT_EQ(Connect(2), ncclSuccess);

  EXPECT_EQ(net_.acceptCalls, 1);
  ncclSharedNetComms* comms = proxyState_.progressState.netComms[2] + 6;
  EXPECT_EQ(comms->recvComm[3], &net_.recvComm);
  EXPECT_EQ(comms->recvRefCount[3], 3);
  EXPECT_EQ(static_cast<recvNetResources*>(conns_[1].transportResources)->netRecvComm, &net_.recvComm);
}

TEST_F(NetRecvProxyConnectMicrotest, SharedCommsPeerStillAccepting_Waits) {
  AddRecvConn(1, 0, 1)->maxRecvs = 2;
  AddRecvConn(1, 0, 2)->maxRecvs = 2;
  net_.commReady = false;

  EXPECT_EQ(Connect(0), ncclInProgress);
  EXPECT_EQ(Connect(1), ncclInProgress);
  EXPECT_EQ(net_.acceptCalls, 1);
}

TEST_F(NetRecvProxyConnectMicrotest, SharedCommsDisabled_AcceptsPerConnection) {
  ScopedHook param(g_loadParam, ParamOverride({{"NET_SHARED_COMMS", 0}}));
  AddRecvConn(1, 0, 1)->maxRecvs = 2;
  AddRecvConn(1, 0, 2)->maxRecvs = 2;

  ASSERT_EQ(Connect(0), ncclSuccess);
  ASSERT_EQ(Connect(1), ncclSuccess);

  EXPECT_EQ(net_.acceptCalls, 2);
  EXPECT_EQ(proxyState_.progressState.netComms[2], nullptr);
}

TEST_F(NetRecvProxyConnectMicrotest, GdrCopy_TailAndFlushWordsLiveInGdcMemory) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  ScopedHook param(g_loadParam, ParamOverride({{"GDRCOPY_FLUSH_ENABLE", 1}}));
  recvNetResources* r = AddRecvConn();

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_NE(r->gdcSync, nullptr);
  EXPECT_EQ(r->gdcFlush, r->gdcSync + 1);
  EXPECT_EQ(resp_.mems[NCCL_NET_MAP_GDCMEM].cpuPtr, reinterpret_cast<char*>(r->gdcSync));
  EXPECT_EQ(resp_.mems[NCCL_NET_MAP_GDCMEM].size, (ssize_t)sizeof(uint64_t));
}

TEST_F(NetRecvProxyConnectMicrotest, GdrCopyKnobs_SelectSyncAndFlushIndependently) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  {
    ScopedHook param(g_loadParam, ParamOverride({{"GDRCOPY_SYNC_ENABLE", 0}, {"GDRCOPY_FLUSH_ENABLE", 1}}));
    AddRecvConn();
    ASSERT_EQ(Connect(0), ncclSuccess);
  }
  AddRecvConn()->sameDevice = 0;
  ASSERT_EQ(Connect(1), ncclSuccess);
  AddRecvConn();
  ASSERT_EQ(Connect(2), ncclSuccess);

  auto* noSync = static_cast<recvNetResources*>(conns_[0].transportResources);
  EXPECT_EQ(noSync->gdcSync, nullptr);
  EXPECT_NE(noSync->gdcFlush, nullptr);
  EXPECT_EQ(static_cast<recvNetResources*>(conns_[1].transportResources)->gdcSync, nullptr);
  auto* defaults = static_cast<recvNetResources*>(conns_[2].transportResources);
  EXPECT_NE(defaults->gdcSync, nullptr);
  EXPECT_EQ(defaults->gdcFlush, nullptr);
}

TEST_F(NetRecvProxyConnectMicrotest, UnpackDevice_AllocatesHandleForPluginAndExportsMhandles) {
  recvNetResources* r = AddRecvConn();
  r->netDeviceType = NCCL_NET_DEVICE_UNPACK;
  r->netDeviceVersion = NCCL_NET_DEVICE_UNPACK_VERSION;

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_NE(r->netDeviceHandle, nullptr);
  EXPECT_EQ(net_.lastDevHandleArg, &r->netDeviceHandle);
  EXPECT_EQ(conns_[0].netDeviceHandle, r->netDeviceHandle);
  EXPECT_EQ(conns_[0].needsProxyProgress, r->netDeviceHandle->needsProxyProgress);
  EXPECT_EQ(net_.getDeviceMrCalls, NCCL_NUM_PROTOCOLS);
  EXPECT_EQ(conns_[0].mhandles[NCCL_PROTO_LL], Buffer(r, NCCL_PROTO_LL) + 1);

  vtable_.getDeviceMr = nullptr;
  AddRecvConn()->netDeviceType = NCCL_NET_DEVICE_UNPACK;
  ASSERT_EQ(Connect(1), ncclSuccess);
  EXPECT_EQ(net_.getDeviceMrCalls, NCCL_NUM_PROTOCOLS);
}

TEST_F(NetRecvProxyConnectMicrotest, HsaDmaBuf_RegistersDeviceBuffersByFd) {
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  proxyState_.dmaBufSupport = true;
  recvNetResources* r = AddRecvConn(0, 1);
  r->useDmaBuf = 1;

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(net_.regMrDmaBufs.size(), (size_t)NCCL_NUM_PROTOCOLS);
  EXPECT_TRUE(net_.regMrs.empty());
  EXPECT_EQ(net_.regMrDmaBufs[0].data, Buffer(r, NCCL_PROTO_LL));
  EXPECT_EQ(net_.regMrDmaBufs[0].comm, &net_.recvComm);
  EXPECT_EQ(net_.regMrDmaBufs[0].offset, 0x40u);
  EXPECT_EQ(net_.regMrDmaBufs[0].fd, g_exportedFds[0]);
  for (int fd : g_exportedFds) EXPECT_EQ(fcntl(fd, F_GETFD), -1) << "fd " << fd << " left open";
}

TEST_F(NetRecvProxyConnectMicrotest, HsaDmaBufWithHostStaging_KeepsHostBufferOnRegMr) {
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  proxyState_.dmaBufSupport = true;
  proxyState_.allocP2pNetLLBuffers = true;
  recvNetResources* r = AddRecvConn(/*shared=*/1, /*useGdr=*/1);
  r->useDmaBuf = 1;

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(net_.regMrs.size(), 1u);
  EXPECT_EQ(net_.regMrs[0].data, Buffer(r, NCCL_PROTO_LL));
  EXPECT_EQ(net_.regMrDmaBufs.size(), 2u);
}

TEST_F(NetRecvProxyConnectMicrotest, DmaBufUnavailable_FallsBackToRegMr) {
  proxyState_.dmaBufSupport = true;
  AddRecvConn(0, 1)->useDmaBuf = 1;
  ASSERT_EQ(Connect(0), ncclSuccess);

  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  proxyState_.dmaBufSupport = false;
  AddRecvConn(0, 1)->useDmaBuf = 1;
  ASSERT_EQ(Connect(1), ncclSuccess);

  EXPECT_TRUE(net_.regMrDmaBufs.empty());
  EXPECT_EQ(net_.regMrs.size(), 2u * NCCL_NUM_PROTOCOLS);
}

TEST_F(NetRecvProxyConnectMicrotest, DmaBufExportFails_ReturnsError) {
  pfn_hsa_amd_portable_export_dmabuf = [](const void*, size_t, int*, uint64_t*) { return HSA_STATUS_ERROR; };
  proxyState_.dmaBufSupport = true;
  AddRecvConn(0, 1)->useDmaBuf = 1;

  EXPECT_EQ(Connect(), ncclUnhandledCudaError);
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
TEST_F(NetRecvProxyConnectMicrotest, CuMemDmaBuf_RegistersDeviceBuffersAgainstBankFd) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  recvNetResources* r = AddRecvConn(0, ncclTopoGdrModePci);
  r->useDmaBuf = 1;

  ASSERT_EQ(Connect(), ncclSuccess);

  connectMapMem& bank = r->map.mems[NCCL_NET_MAP_DEVMEM];
  ASSERT_EQ(exports_.size(), 1u);
  EXPECT_EQ(exports_[0].ptr, bank.gpuPtr);
  EXPECT_EQ(exports_[0].size, (size_t)bank.size);
  EXPECT_EQ(exports_[0].flags, (unsigned long long)hipMemRangeFlagDmaBufMappingTypePcie);
  ASSERT_EQ(net_.regMrDmaBufs.size(), (size_t)NCCL_NUM_PROTOCOLS);
  for (const auto& m : net_.regMrDmaBufs) {
    EXPECT_EQ(m.comm, &net_.recvComm);
    EXPECT_EQ(m.fd, bank.dmaBufFd);
    EXPECT_EQ(m.offset, (uint64_t)(static_cast<char*>(m.data) - bank.gpuPtr));
  }
}

TEST_F(NetRecvProxyConnectMicrotest, CuMemDmaBufPoolBuffer_ExportedAndClosedOnItsOwn) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  recvNetResources* r = AddRecvConn(/*shared=*/1, ncclTopoGdrModeDefault);
  r->useDmaBuf = 1;

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(exports_.size(), 1u);
  EXPECT_EQ(exports_[0].ptr, Buffer(r, NCCL_PROTO_SIMPLE));
  ASSERT_EQ(net_.regMrDmaBufs.size(), 1u);
  EXPECT_EQ(net_.regMrDmaBufs[0].fd, exports_[0].fd);
  EXPECT_EQ(net_.regMrDmaBufs[0].offset, 0u);
  EXPECT_EQ(fcntl(exports_[0].fd, F_GETFD), -1);
}

TEST_F(NetRecvProxyConnectMicrotest, CuMemDmaBufPoolBesideStagingBank_ExportedSeparately) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  proxyState_.allocP2pNetLLBuffers = true;
  recvNetResources* r = AddRecvConn(/*shared=*/1, ncclTopoGdrModeDefault);
  r->useDmaBuf = 1;

  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(exports_.size(), 2u);  // the LL128 staging bank, then the pool buffer
  EXPECT_EQ(RegMrType(Buffer(r, NCCL_PROTO_LL)), NCCL_PTR_HOST);
  ASSERT_EQ(net_.regMrDmaBufs.size(), 2u);
  EXPECT_EQ(net_.regMrDmaBufs[0].data, Buffer(r, NCCL_PROTO_LL128));
  EXPECT_EQ(net_.regMrDmaBufs[0].fd, exports_[0].fd);
  EXPECT_EQ(net_.regMrDmaBufs[1].data, Buffer(r, NCCL_PROTO_SIMPLE));
  EXPECT_EQ(net_.regMrDmaBufs[1].fd, exports_[1].fd);
}

TEST_F(NetRecvProxyConnectMicrotest, CuMemBankExportFails_ReturnsError) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  exports_.result = hipErrorNotSupported;
  AddRecvConn(0, 1);

  EXPECT_EQ(Connect(), ncclUnhandledCudaError);
}
#endif

TEST_F(NetRecvProxyConnectMicrotest, RegistrationFails_ReturnsError) {
  AddRecvConn();
  net_.regMrResult = ncclSystemError;

  EXPECT_EQ(Connect(), ncclSystemError);
}

TEST_F(NetRecvProxyConnectMicrotest, WrongResponseSize_Rejected) {
  AddRecvConn();

  EXPECT_EQ(Connect(0, sizeof(connectMap) + 1), ncclInternalError);
}

// ---------------------------------------------------------------------------
// Proxy-side free (sendProxyFree, recvProxyFree)
// ---------------------------------------------------------------------------

class NetSendProxyFreeMicrotest : public NetSendProxyConnectMicrotest {
 protected:
  ncclResult_t Free(size_t i = 0) { return FreeConn(i, sendProxyFree); }
};

TEST_F(NetSendProxyFreeMicrotest, Connected_DeregistersClosesAndReleasesEverything) {
  AddConn(0, /*useGdr=*/1);
  ASSERT_EQ(Connect(), ncclSuccess);
  sendNetResources* r = static_cast<sendNetResources*>(conns_[0].transportResources);
  std::vector<void*> mhandles(r->mhandles, r->mhandles + NCCL_NUM_PROTOCOLS);

  ASSERT_EQ(Free(), ncclSuccess);

  ASSERT_EQ(net_.deregMrs.size(), (size_t)NCCL_NUM_PROTOCOLS);
  for (int p = 0; p < NCCL_NUM_PROTOCOLS; p++) {
    EXPECT_EQ(net_.deregMrs[p].first, &net_.sendComm);
    EXPECT_EQ(net_.deregMrs[p].second, mhandles[p]);
  }
  ASSERT_EQ(net_.closedSends.size(), 1u);
  EXPECT_EQ(net_.closedSends[0], &net_.sendComm);
  EXPECT_EQ(hostFrees_, 1);
  EXPECT_EQ(devFrees_, 1);
  EXPECT_TRUE(live_.empty());
}

TEST_F(NetSendProxyFreeMicrotest, RegisteredUserBuffers_DeregisteredAndFreed) {
  AddConn();
  ASSERT_EQ(Connect(), ncclSuccess);
  for (uintptr_t h : {0x100, 0x200}) {
    auto* mh = static_cast<proxyMemHandle*>(calloc(1, sizeof(proxyMemHandle)));
    mh->handle = reinterpret_cast<void*>(h);
    ncclIntruQueueEnqueue(&conns_[0].proxyMemHandleQueue, mh);
  }

  ASSERT_EQ(Free(), ncclSuccess);

  ASSERT_EQ(net_.deregMrs.size(), 2u + NCCL_NUM_PROTOCOLS);
  EXPECT_EQ(net_.deregMrs[0].second, reinterpret_cast<void*>(0x100));
  EXPECT_EQ(net_.deregMrs[1].second, reinterpret_cast<void*>(0x200));
  EXPECT_TRUE(ncclIntruQueueEmpty(&conns_[0].proxyMemHandleQueue));
}

TEST_F(NetSendProxyFreeMicrotest, RemoteProxy_ClosesShmAndReleasesShareableBank) {
  AddConn(0, 1);
  conns_[0].sameProcess = 0;
  ScopedHook shm(g_shmAllocateShareableBuffer, [this](size_t n, bool, void*, void** h, void** d) {
    *h = *d = Alloc(n);
    return ncclSuccess;
  });
  ASSERT_EQ(Connect(), ncclSuccess);
  ScopedHook shmClose(g_shmIpcClose, [](shmIpcDesc*) { return ncclSuccess; });
  ScopedHook ipcFree(g_ncclP2pFreeShareableBuffer, [](ncclIpcDesc*) { return ncclSuccess; });

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(shmClose.calls, 1);
  EXPECT_EQ(ipcFree.calls, 1);
  EXPECT_EQ(hostFrees_, 0);
  EXPECT_EQ(devFrees_, 1);
}

TEST_F(NetSendProxyFreeMicrotest, CuMemDeviceBank_ReleasesShareableBuffer) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  AddConn(0, 1);
  ASSERT_EQ(Connect(), ncclSuccess);
  ScopedHook ipcFree(g_ncclP2pFreeShareableBuffer, [](ncclIpcDesc*) { return ncclSuccess; });
  ScopedHook retain(g_hipMemRetainAllocationHandle, [](hipMemGenericAllocationHandle_t*, void*) { return hipSuccess; });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });
  ScopedHook unmap(g_hipMemUnmap, [](void*, size_t) { return hipSuccess; });
  ScopedHook addrFree(g_hipMemAddressFree, [this](void* p, size_t) {
    Release(p);
    return hipSuccess;
  });

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(ipcFree.calls, 1);
  EXPECT_TRUE(live_.empty());
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  ASSERT_EQ(exports_.size(), 1u);
  EXPECT_EQ(fcntl(exports_[0].fd, F_GETFD), -1);
#endif
}

TEST_F(NetSendProxyFreeMicrotest, DmaBufFd_Closed) {
  AddConn(0, 1);
  ASSERT_EQ(Connect(), ncclSuccess);
  int fd = open("/dev/null", O_RDONLY);
  static_cast<sendNetResources*>(conns_[0].transportResources)->map.mems[NCCL_NET_MAP_DEVMEM].dmaBufFd = fd;

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(fcntl(fd, F_GETFD), -1);
}

TEST_F(NetSendProxyFreeMicrotest, GdrCopy_ReleasesGdcMemory) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  AddConn();
  ASSERT_EQ(Connect(), ncclSuccess);
  ASSERT_EQ(devAllocs_, 1);

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(devFrees_, 1);
  EXPECT_TRUE(live_.empty());
}

TEST_F(NetSendProxyFreeMicrotest, SharedBuffers_ReturnPoolAndCloseComm) {
  AddConn(/*shared=*/1);
  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(net_.closedSends.size(), 1u);
  EXPECT_TRUE(live_.empty());
}

TEST_F(NetSendProxyFreeMicrotest, SharedComms_LastUserClosesComm) {
  AddConn(1, 0, 1)->maxRecvs = 2;
  AddConn(1, 0, 2)->maxRecvs = 2;
  ASSERT_EQ(Connect(0), ncclSuccess);
  ASSERT_EQ(Connect(1), ncclSuccess);
  ncclSharedNetComms* comms = proxyState_.progressState.netComms[2] + 5;

  ASSERT_EQ(Free(0), ncclSuccess);
  EXPECT_EQ(comms->sendRefCount[3], 1);
  EXPECT_TRUE(net_.closedSends.empty());

  ASSERT_EQ(Free(1), ncclSuccess);
  ASSERT_EQ(net_.closedSends.size(), 1u);
  EXPECT_EQ(net_.closedSends[0], &net_.sendComm);
}

TEST_F(NetSendProxyFreeMicrotest, SharedInitOnly_ReturnsPool) {
  conns_.emplace_back();
  conns_[0].tpLocalRank = 2;
  conns_[0].sameProcess = 1;
  ASSERT_EQ(sharedNetBuffersInit(&proxyState_, 0, 2, 0, 1, 1, nullptr, nullptr, nullptr, nullptr), ncclSuccess);
  conns_[0].state = connSharedInitialized;

  ASSERT_EQ(sendProxyFree(&conns_[0], &proxyState_), ncclSuccess);

  EXPECT_EQ(proxyState_.progressState.localPeers, nullptr);
  EXPECT_TRUE(net_.closedSends.empty());
}

TEST_F(NetSendProxyFreeMicrotest, NeverConnected_FreesOnlyResources) {
  for (proxyConnectState state : {connInitialized, connSetupDone}) {
    AddConn();
    conns_.back().state = state;
    ASSERT_EQ(sendProxyFree(&conns_.back(), &proxyState_), ncclSuccess);
    conns_.back().transportResources = nullptr;
  }

  EXPECT_TRUE(net_.deregMrs.empty());
  EXPECT_TRUE(net_.closedSends.empty());
}

TEST_F(NetSendProxyFreeMicrotest, CuMemWithoutDeviceBank_ReleasesNoShareableBuffer) {
  AddConn(/*shared=*/1);
  ASSERT_EQ(Connect(), ncclSuccess);
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ScopedHook ipcFree(g_ncclP2pFreeShareableBuffer, [](ncclIpcDesc*) { return ncclSuccess; });

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(ipcFree.calls, 0);
}

TEST_F(NetSendProxyFreeMicrotest, BrokenSharedPool_ReturnsError) {
  AddConn(1);
  ASSERT_EQ(Connect(), ncclSuccess);
  proxyState_.progressState.localPeers[1]->send.size = 0;

  EXPECT_EQ(Free(), ncclInternalError);
}

TEST_F(NetSendProxyFreeMicrotest, DeregistrationFails_ReturnsError) {
  AddConn();
  ASSERT_EQ(Connect(), ncclSuccess);
  net_.deregMrResult = ncclSystemError;

  EXPECT_EQ(Free(), ncclSystemError);
}

class NetRecvProxyFreeMicrotest : public NetRecvProxyConnectMicrotest {
 protected:
  ncclResult_t Free(size_t i = 0) { return FreeConn(i, recvProxyFree); }
};

TEST_F(NetRecvProxyFreeMicrotest, Connected_DeregistersClosesAndReleasesEverything) {
  AddRecvConn(0, 1);
  ASSERT_EQ(Connect(), ncclSuccess);
  auto* mh = static_cast<proxyMemHandle*>(calloc(1, sizeof(proxyMemHandle)));
  mh->handle = reinterpret_cast<void*>(0x100);
  ncclIntruQueueEnqueue(&conns_[0].proxyMemHandleQueue, mh);

  ASSERT_EQ(Free(), ncclSuccess);

  ASSERT_EQ(net_.deregMrs.size(), 1u + NCCL_NUM_PROTOCOLS);
  EXPECT_EQ(net_.deregMrs[0].second, reinterpret_cast<void*>(0x100));
  for (const auto& d : net_.deregMrs) EXPECT_EQ(d.first, &net_.recvComm);
  ASSERT_EQ(net_.closedRecvs.size(), 1u);
  EXPECT_EQ(net_.closedRecvs[0], &net_.recvComm);
  EXPECT_EQ(hostFrees_, 1);
  EXPECT_EQ(devFrees_, 1);
  EXPECT_TRUE(live_.empty());
}

TEST_F(NetRecvProxyFreeMicrotest, CuMemDeviceBankAndDmaBufFd_Released) {
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  AddRecvConn(0, 1);
  ASSERT_EQ(Connect(), ncclSuccess);
#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
  ASSERT_EQ(exports_.size(), 1u);
  int fd = exports_[0].fd;
#else
  int fd = open("/dev/null", O_RDONLY);
  static_cast<recvNetResources*>(conns_[0].transportResources)->map.mems[NCCL_NET_MAP_DEVMEM].dmaBufFd = fd;
#endif
  ScopedHook ipcFree(g_ncclP2pFreeShareableBuffer, [](ncclIpcDesc*) { return ncclSuccess; });
  ScopedHook retain(g_hipMemRetainAllocationHandle, [](hipMemGenericAllocationHandle_t*, void*) { return hipSuccess; });
  ScopedHook release(g_hipMemRelease, [](hipMemGenericAllocationHandle_t) { return hipSuccess; });
  ScopedHook unmap(g_hipMemUnmap, [](void*, size_t) { return hipSuccess; });
  ScopedHook addrFree(g_hipMemAddressFree, [this](void* p, size_t) {
    Release(p);
    return hipSuccess;
  });

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(ipcFree.calls, 1);
  EXPECT_EQ(fcntl(fd, F_GETFD), -1);
  EXPECT_TRUE(live_.empty());
}

TEST_F(NetRecvProxyFreeMicrotest, GdrCopy_ReleasesGdcMemory) {
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  AddRecvConn();
  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(devFrees_, 1);
  EXPECT_TRUE(live_.empty());
}

TEST_F(NetRecvProxyFreeMicrotest, SharedBuffers_ReturnPoolAndCloseComm) {
  AddRecvConn(1);
  ASSERT_EQ(Connect(), ncclSuccess);

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(net_.closedRecvs.size(), 1u);
  EXPECT_TRUE(live_.empty());
}

TEST_F(NetRecvProxyFreeMicrotest, SharedComms_LastUserClosesComm) {
  AddRecvConn(1, 0, 1)->maxRecvs = 2;
  AddRecvConn(1, 0, 2)->maxRecvs = 2;
  ASSERT_EQ(Connect(0), ncclSuccess);
  ASSERT_EQ(Connect(1), ncclSuccess);
  ncclSharedNetComms* comms = proxyState_.progressState.netComms[2] + 6;

  ASSERT_EQ(Free(0), ncclSuccess);
  EXPECT_EQ(comms->recvRefCount[3], 1);
  EXPECT_TRUE(net_.closedRecvs.empty());

  ASSERT_EQ(Free(1), ncclSuccess);
  ASSERT_EQ(net_.closedRecvs.size(), 1u);
  EXPECT_EQ(net_.closedRecvs[0], &net_.recvComm);
}

TEST_F(NetRecvProxyFreeMicrotest, SharedInitOnly_ReturnsPool) {
  conns_.emplace_back();
  conns_[0].tpLocalRank = 2;
  conns_[0].sameProcess = 1;
  ASSERT_EQ(sharedNetBuffersInit(&proxyState_, 0, 2, 1, 1, 1, nullptr, nullptr, nullptr, nullptr), ncclSuccess);
  conns_[0].state = connSharedInitialized;

  ASSERT_EQ(recvProxyFree(&conns_[0], &proxyState_), ncclSuccess);

  EXPECT_EQ(proxyState_.progressState.localPeers, nullptr);
}

TEST_F(NetRecvProxyFreeMicrotest, NeverConnected_FreesOnlyResources) {
  for (proxyConnectState state : {connInitialized, connSetupDone}) {
    AddRecvConn();
    conns_.back().state = state;
    ASSERT_EQ(recvProxyFree(&conns_.back(), &proxyState_), ncclSuccess);
    conns_.back().transportResources = nullptr;
  }

  EXPECT_TRUE(net_.deregMrs.empty());
  EXPECT_TRUE(net_.closedRecvs.empty());
}

TEST_F(NetRecvProxyFreeMicrotest, CuMemWithoutDeviceBank_ReleasesNoShareableBuffer) {
  AddRecvConn(1);
  ASSERT_EQ(Connect(), ncclSuccess);
  ScopedHook cuMem(g_cuMemEnable, [] { return 1; });
  ScopedHook ipcFree(g_ncclP2pFreeShareableBuffer, [](ncclIpcDesc*) { return ncclSuccess; });

  ASSERT_EQ(Free(), ncclSuccess);

  EXPECT_EQ(ipcFree.calls, 0);
}

TEST_F(NetRecvProxyFreeMicrotest, BrokenSharedPool_ReturnsError) {
  AddRecvConn(1);
  ASSERT_EQ(Connect(), ncclSuccess);
  proxyState_.progressState.localPeers[1]->recv.size = 0;

  EXPECT_EQ(Free(), ncclInternalError);
}

TEST_F(NetRecvProxyFreeMicrotest, DeregistrationFails_ReturnsError) {
  AddRecvConn();
  ASSERT_EQ(Connect(), ncclSuccess);
  net_.deregMrResult = ncclSystemError;

  EXPECT_EQ(Free(), ncclSystemError);
}

// ---------------------------------------------------------------------------
// Client-side buffer registration (netRegisterBuffer, ncclNetLocalRegisterBuffer,
// ncclNetGraphRegisterBuffer, cleanupNet, ncclNetDeregBuffer)
// ---------------------------------------------------------------------------

class NetRegisterMicrotest : public NetMicrotestBase {
 protected:
  static constexpr size_t kSegment = 1 << 16;
  static constexpr int kPeers = 2;

  ncclComm comm_{};
  std::vector<char> mem_ = std::vector<char>(4 * kSegment);
  ncclReg reg_{};
  ncclReg* slots_[1] = {&reg_};
  ncclConnector peers_[kPeers]{};
  ncclConnector* peerConns_[kPeers] = {&peers_[0], &peers_[1]};
  void* handles_[kPeers] = {};
  int flag_ = -1;
  std::vector<netRegInfo> registered_;
  std::vector<ncclProxyConnector*> registeredOn_;
  std::deque<ncclRegNetHandles> preexisting_;
  ncclIntruQueue<ncclCommCallback, &ncclCommCallback::next> cleanup_ = {};

  void SetUp() override {
    NetMicrotestBase::SetUp();
    reg_.begAddr = reinterpret_cast<uintptr_t>(mem_.data());
    reg_.endAddr = reg_.begAddr + mem_.size();
    comm_.regCache.slots = slots_;
    comm_.regCache.population = 1;
    for (auto& p : peers_) p.conn.flags = NCCL_DIRECT_NIC;
    // Each kSegment-aligned slice of mem_ is its own physical segment.
    g_hipMemGetAddressRange = [this](hipDeviceptr_t* base, size_t* size, hipDeviceptr_t ptr) {
      size_t off = static_cast<char*>(ptr) - mem_.data();
      *base = mem_.data() + off / kSegment * kSegment;
      *size = kSegment;
      return hipSuccess;
    };
    g_regLocalIsValid = [](ncclReg*, bool* valid) {
      *valid = true;
      return ncclSuccess;
    };
    g_proxyCallBlocking = [this](ncclComm*, ncclProxyConnector* conn, int type, void* req, int reqSize, void* resp,
                                 int respSize) {
      EXPECT_EQ(type, ncclProxyMsgRegister);
      EXPECT_EQ(reqSize, (int)sizeof(netRegInfo));
      EXPECT_EQ(respSize, (int)sizeof(void*));
      registered_.push_back(*static_cast<netRegInfo*>(req));
      registeredOn_.push_back(conn);
      *static_cast<void**>(resp) = reinterpret_cast<void*>(0x1000 + registered_.size());
      return ncclSuccess;
    };
  }

  void TearDown() override {
    for (ncclRegNetHandles* h = reg_.netHandleHead; h;) {
      ncclRegNetHandles* next = h->next;
      bool owned = true;
      for (auto& p : preexisting_) owned &= (&p != h);
      if (owned) free(h);
      h = next;
    }
    while (!ncclIntruQueueEmpty(&cleanup_)) free(ncclIntruQueueDequeue(&cleanup_));
    NetMicrotestBase::TearDown();
  }

  ncclResult_t LocalRegister(const void* buff, size_t size, int nPeers = kPeers) {
    return ncclNetLocalRegisterBuffer(&comm_, buff, size, peerConns_, nPeers, &flag_, handles_);
  }
  ncclResult_t GraphRegister(const void* buff, size_t size, int* nElts = nullptr) {
    return ncclNetGraphRegisterBuffer(&comm_, buff, size, peerConns_, kPeers, &flag_, handles_, &cleanup_, nElts);
  }
};

TEST_F(NetRegisterMicrotest, LocalRegister_NothingToRegister_LeavesBufferUnregistered) {
  EXPECT_EQ(ncclNetLocalRegisterBuffer(nullptr, mem_.data(), 64, peerConns_, kPeers, &flag_, handles_), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(LocalRegister(nullptr, 64), ncclSuccess);
  EXPECT_EQ(LocalRegister(mem_.data(), 0), ncclSuccess);
  EXPECT_EQ(LocalRegister(mem_.data(), 64, /*nPeers=*/0), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  EXPECT_TRUE(registered_.empty());
}

TEST_F(NetRegisterMicrotest, LocalRegister_NotLocallyRegistered_LeavesBufferUnregistered) {
  ScopedHook valid(g_regLocalIsValid, [](ncclReg*, bool* v) {
    *v = false;
    return ncclSuccess;
  });

  ASSERT_EQ(LocalRegister(mem_.data(), 64), ncclSuccess);

  EXPECT_EQ(flag_, 0);
  EXPECT_TRUE(registered_.empty());
}

TEST_F(NetRegisterMicrotest, LocalRegister_ValidRecord_RegistersWholeRecordWithEachPeer) {
  ASSERT_EQ(LocalRegister(mem_.data() + 100, 64), ncclSuccess);

  EXPECT_EQ(flag_, 1);
  ASSERT_EQ(registered_.size(), 2u);
  EXPECT_EQ(registeredOn_[0], &peers_[0].proxyConn);
  EXPECT_EQ(registeredOn_[1], &peers_[1].proxyConn);
  EXPECT_EQ(registered_[0].buffer, reg_.begAddr);
  EXPECT_EQ(registered_[0].size, mem_.size());
  EXPECT_EQ(registered_[0].numSegments, 1);
  EXPECT_EQ(handles_[0], reinterpret_cast<void*>(0x1001));
  EXPECT_EQ(handles_[1], reinterpret_cast<void*>(0x1002));
  EXPECT_TRUE(reg_.state & NET_REG_COMPLETE);
  std::set<std::pair<ncclProxyConnector*, void*>> remembered;
  for (ncclRegNetHandles* h = reg_.netHandleHead; h; h = h->next) remembered.insert({h->proxyConn, h->handle});
  EXPECT_EQ(remembered, (std::set<std::pair<ncclProxyConnector*, void*>>{
                            {&peers_[0].proxyConn, reinterpret_cast<void*>(0x1001)},
                            {&peers_[1].proxyConn, reinterpret_cast<void*>(0x1002)}}));
}

TEST_F(NetRegisterMicrotest, LocalRegister_PeerAlreadyRegistered_ReusesHandle) {
  preexisting_.push_back({reinterpret_cast<void*>(0xAB), &peers_[0].proxyConn, nullptr});
  reg_.netHandleHead = &preexisting_.back();

  ASSERT_EQ(LocalRegister(mem_.data(), 64), ncclSuccess);

  EXPECT_EQ(flag_, 1);
  EXPECT_EQ(handles_[0], reinterpret_cast<void*>(0xAB));
  ASSERT_EQ(registered_.size(), 1u);
  EXPECT_EQ(registeredOn_[0], &peers_[1].proxyConn);
}

TEST_F(NetRegisterMicrotest, LocalRegister_EveryPeerAlreadyRegistered_RegisteredWithoutProxyCalls) {
  preexisting_.push_back({reinterpret_cast<void*>(0xAB), &peers_[0].proxyConn, nullptr});
  preexisting_.push_back({reinterpret_cast<void*>(0xCD), &peers_[1].proxyConn, &preexisting_[0]});
  reg_.netHandleHead = &preexisting_.back();

  ASSERT_EQ(LocalRegister(mem_.data(), 64), ncclSuccess);

  EXPECT_EQ(flag_, 1);
  EXPECT_EQ(handles_[1], reinterpret_cast<void*>(0xCD));
  EXPECT_TRUE(registered_.empty());
}

TEST_F(NetRegisterMicrotest, LocalRegister_MissingPeer_Skipped) {
  peerConns_[0] = nullptr;

  ASSERT_EQ(LocalRegister(mem_.data(), 64), ncclSuccess);

  EXPECT_EQ(flag_, 1);
  ASSERT_EQ(registered_.size(), 1u);
  EXPECT_EQ(registeredOn_[0], &peers_[1].proxyConn);
}

TEST_F(NetRegisterMicrotest, LocalRegister_PeerWithoutGdr_LeavesBufferUnregistered) {
  peers_[0].conn.flags = 0;

  ASSERT_EQ(LocalRegister(mem_.data(), 64), ncclSuccess);

  EXPECT_EQ(flag_, 0);
  EXPECT_TRUE(registered_.empty());
}

TEST_F(NetRegisterMicrotest, LocalRegister_ProxyDeclines_LeavesBufferUnregistered) {
  ScopedHook call(g_proxyCallBlocking, [](ncclComm*, ncclProxyConnector*, int, void*, int, void* resp, int) {
    *static_cast<void**>(resp) = nullptr;
    return ncclSuccess;
  });

  ASSERT_EQ(LocalRegister(mem_.data(), 64), ncclSuccess);

  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(reg_.netHandleHead, nullptr);
}

TEST_F(NetRegisterMicrotest, LocalRegister_DependencyFails_ReturnsErrorUnregistered) {
  {
    ScopedHook call(g_proxyCallBlocking, [](auto...) { return ncclSystemError; });
    EXPECT_EQ(LocalRegister(mem_.data(), 64), ncclSystemError);
    EXPECT_EQ(flag_, 0);
  }
  {
    ScopedHook valid(g_regLocalIsValid, [](auto...) { return ncclInternalError; });
    EXPECT_EQ(LocalRegister(mem_.data(), 64), ncclInternalError);
    EXPECT_EQ(flag_, 0);
  }
  {
    ScopedHook range(g_hipMemGetAddressRange, [](auto...) { return hipErrorInvalidValue; });
    EXPECT_EQ(LocalRegister(mem_.data(), 64), ncclUnhandledCudaError);
    EXPECT_EQ(flag_, 0);
  }
}

TEST_F(NetRegisterMicrotest, LocalRegister_MultiSegmentBuffer_RegisteredOnlyWhenEnabled) {
  ASSERT_EQ(LocalRegister(mem_.data(), 2 * kSegment), ncclSuccess);
  EXPECT_EQ(flag_, 0);
  EXPECT_TRUE(registered_.empty());

  ScopedHook param(g_loadParam, ParamOverride({{"MULTI_SEGMENT_REGISTER", 1}}));
  ASSERT_EQ(LocalRegister(mem_.data(), 2 * kSegment), ncclSuccess);
  EXPECT_EQ(flag_, 1);
  ASSERT_FALSE(registered_.empty());
  EXPECT_EQ(registered_[0].numSegments, 2);
}

TEST_F(NetRegisterMicrotest, GraphRegister_Registered_QueuesCleanupThatDeregisters) {
  void* graphBase = nullptr;
  size_t graphSize = 0;
  ScopedHook graphReg(g_commGraphRegister, [&](ncclComm*, void* buff, size_t size, void** handle) {
    graphBase = buff;
    graphSize = size;
    *handle = &reg_;
    return ncclSuccess;
  });
  ncclReg* deregistered = nullptr;
  ScopedHook graphDereg(g_commGraphDeregister, [&](ncclComm*, ncclReg* r) {
    deregistered = r;
    return ncclSuccess;
  });
  int nElts = 0;

  ASSERT_EQ(GraphRegister(mem_.data() + kSegment + 8, 64, &nElts), ncclSuccess);

  EXPECT_EQ(flag_, 1);
  EXPECT_EQ(graphBase, mem_.data() + kSegment);
  EXPECT_EQ(graphSize, kSegment);
  EXPECT_EQ(nElts, 1);
  EXPECT_EQ(graphDereg.calls, 0);
  ncclCommCallback* cb = ncclIntruQueueDequeue(&cleanup_);
  ASSERT_NE(cb, nullptr);
  EXPECT_EQ(cb->fn(&comm_, cb), ncclSuccess);  // frees cb
  EXPECT_EQ(deregistered, &reg_);
}

TEST_F(NetRegisterMicrotest, GraphRegister_NotRegistered_ReleasesGraphRecord) {
  peers_[1].conn.flags = 0;
  ScopedHook graphReg(g_commGraphRegister, [this](ncclComm*, void*, size_t, void** handle) {
    *handle = &reg_;
    return ncclSuccess;
  });
  ScopedHook graphDereg(g_commGraphDeregister, [](ncclComm*, ncclReg*) { return ncclSuccess; });

  ASSERT_EQ(GraphRegister(mem_.data(), 64), ncclSuccess);

  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(graphDereg.calls, 1);
  EXPECT_TRUE(ncclIntruQueueEmpty(&cleanup_));
}

TEST_F(NetRegisterMicrotest, GraphRegister_NothingToRegisterOrMultiSegment_Skipped) {
  ScopedHook graphReg(g_commGraphRegister, [](auto...) { return ncclInternalError; });

  EXPECT_EQ(GraphRegister(nullptr, 64), ncclSuccess);
  EXPECT_EQ(ncclNetGraphRegisterBuffer(&comm_, mem_.data(), 64, peerConns_, 0, &flag_, handles_, &cleanup_, nullptr),
            ncclSuccess);
  EXPECT_EQ(GraphRegister(mem_.data(), 2 * kSegment), ncclSuccess);

  EXPECT_EQ(flag_, 0);
  EXPECT_EQ(graphReg.calls, 0);
}

TEST_F(NetRegisterMicrotest, GraphRegister_NoGraphRecord_LeavesBufferUnregistered) {
  ScopedHook graphReg(g_commGraphRegister, [](ncclComm*, void*, size_t, void** handle) {
    *handle = nullptr;
    return ncclSuccess;
  });
  ScopedHook graphDereg(g_commGraphDeregister, [](ncclComm*, ncclReg*) { return ncclSuccess; });

  ASSERT_EQ(GraphRegister(mem_.data(), 64), ncclSuccess);

  EXPECT_EQ(flag_, 0);
  EXPECT_TRUE(registered_.empty());
}

TEST_F(NetRegisterMicrotest, GraphRegister_GraphRecordFails_ReturnsError) {
  ScopedHook graphReg(g_commGraphRegister, [](auto...) { return ncclInternalError; });

  EXPECT_EQ(GraphRegister(mem_.data(), 64), ncclInternalError);
  EXPECT_EQ(flag_, 0);
}

TEST_F(NetRegisterMicrotest, DeregBuffer_SendsHandleToProxy) {
  void* sent = nullptr;
  ScopedHook call(g_proxyCallBlocking, [&](ncclComm*, ncclProxyConnector*, int type, void* req, int reqSize, void*,
                                           int respSize) {
    EXPECT_EQ(type, ncclProxyMsgDeregister);
    EXPECT_EQ(reqSize, (int)sizeof(void*));
    EXPECT_EQ(respSize, 0);
    sent = *static_cast<void**>(req);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclNetDeregBuffer(&comm_, &peers_[0].proxyConn, reinterpret_cast<void*>(0x77)), ncclSuccess);
  EXPECT_EQ(sent, reinterpret_cast<void*>(0x77));

  ScopedHook fail(g_proxyCallBlocking, [](auto...) { return ncclSystemError; });
  EXPECT_EQ(ncclNetDeregBuffer(&comm_, &peers_[0].proxyConn, nullptr), ncclSystemError);
}

// ---------------------------------------------------------------------------
// Proxy-side buffer registration (send/recvProxyRegBuffer, send/recvProxyDeregBuffer)
// ---------------------------------------------------------------------------

class NetProxyRegBufferMicrotest : public NetMicrotestBase {
 protected:
  ncclProxyConnection sendConn_{}, recvConn_{};
  sendNetResources sendRes_{};
  recvNetResources recvRes_{};
  netRegInfo info_{0x10000, 4096, 1};
  void* resp_ = reinterpret_cast<void*>(0x1);
  int done_ = 0;

  void SetUp() override {
    NetMicrotestBase::SetUp();
    sendRes_.netSendComm = &net_.sendComm;
    recvRes_.netRecvComm = &net_.recvComm;
    sendConn_.transportResources = &sendRes_;
    recvConn_.transportResources = &recvRes_;
    g_exportedFds.clear();
  }

  void TearDown() override {
    for (auto* q : {&sendConn_.proxyMemHandleQueue, &recvConn_.proxyMemHandleQueue})
      while (!ncclIntruQueueEmpty(q)) free(ncclIntruQueueDequeue(q));
    ResetRocmWrapFakes();
    NetMicrotestBase::TearDown();
  }

  ncclResult_t Reg(bool send, int reqSize = sizeof(netRegInfo), int respSize = sizeof(void*)) {
    done_ = 0;
    resp_ = reinterpret_cast<void*>(0x1);
    return send ? sendProxyRegBuffer(&sendConn_, &proxyState_, &info_, reqSize, &resp_, respSize, &done_)
                : recvProxyRegBuffer(&recvConn_, &proxyState_, &info_, reqSize, &resp_, respSize, &done_);
  }
  ncclResult_t Dereg(bool send, void* handle, int reqSize = sizeof(void*)) {
    done_ = 0;
    return send ? sendProxyDeregBuffer(&sendConn_, &proxyState_, &handle, reqSize, &done_)
                : recvProxyDeregBuffer(&recvConn_, &proxyState_, &handle, reqSize, &done_);
  }
  ncclProxyConnection& Conn(bool send) { return send ? sendConn_ : recvConn_; }
  void* Comm(bool send) { return send ? (void*)&net_.sendComm : (void*)&net_.recvComm; }
};

TEST_F(NetProxyRegBufferMicrotest, Reg_WrongMessageSizes_Rejected) {
  for (bool send : {true, false}) {
    EXPECT_EQ(Reg(send, sizeof(netRegInfo) - 1), ncclInternalError);
    EXPECT_EQ(Reg(send, sizeof(netRegInfo), 4), ncclInternalError);
  }
  EXPECT_TRUE(net_.regMrs.empty());
}

TEST_F(NetProxyRegBufferMicrotest, Reg_SingleSegment_RegistersAndRemembersHandle) {
  for (bool send : {true, false}) {
    SCOPED_TRACE(send);
    net_.regMrs.clear();

    ASSERT_EQ(Reg(send), ncclSuccess);

    EXPECT_EQ(done_, 1);
    ASSERT_EQ(net_.regMrs.size(), 1u);
    EXPECT_EQ(net_.regMrs[0].comm, Comm(send));
    EXPECT_EQ(net_.regMrs[0].data, reinterpret_cast<void*>(0x10000));
    EXPECT_EQ(net_.regMrs[0].size, 4096u);
    EXPECT_EQ(net_.regMrs[0].type, NCCL_PTR_CUDA);
    EXPECT_EQ(resp_, reinterpret_cast<void*>(0x10000));
    ASSERT_FALSE(ncclIntruQueueEmpty(&Conn(send).proxyMemHandleQueue));
    EXPECT_EQ(ncclIntruQueueHead(&Conn(send).proxyMemHandleQueue)->handle, resp_);
  }
}

TEST_F(NetProxyRegBufferMicrotest, Reg_MultiSegmentWithoutDmaBuf_Declines) {
  info_.numSegments = 2;
  for (bool send : {true, false}) {
    ASSERT_EQ(Reg(send), ncclSuccess);
    EXPECT_EQ(done_, 1);
    EXPECT_EQ(resp_, nullptr);
    EXPECT_TRUE(ncclIntruQueueEmpty(&Conn(send).proxyMemHandleQueue));
  }
  EXPECT_TRUE(net_.regMrs.empty());
}

TEST_F(NetProxyRegBufferMicrotest, Reg_PluginFails_DeclinesWithoutError) {
  net_.regMrResult = ncclSystemError;
  for (bool send : {true, false}) {
    ASSERT_EQ(Reg(send), ncclSuccess);
    EXPECT_EQ(resp_, nullptr);
    EXPECT_TRUE(ncclIntruQueueEmpty(&Conn(send).proxyMemHandleQueue));
  }
}

TEST_F(NetProxyRegBufferMicrotest, Reg_HsaDmaBuf_RegistersByFdEvenAcrossSegments) {
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  sendRes_.useDmaBuf = recvRes_.useDmaBuf = 1;
  info_.numSegments = 2;
  for (bool send : {true, false}) {
    SCOPED_TRACE(send);
    net_.regMrDmaBufs.clear();

    ASSERT_EQ(Reg(send), ncclSuccess);

    ASSERT_EQ(net_.regMrDmaBufs.size(), 1u);
    EXPECT_EQ(net_.regMrDmaBufs[0].comm, Comm(send));
    EXPECT_EQ(net_.regMrDmaBufs[0].fd, g_exportedFds.back());
    EXPECT_EQ(fcntl(g_exportedFds.back(), F_GETFD), -1);
    EXPECT_EQ(resp_, reinterpret_cast<void*>(0x10000));
  }
  EXPECT_TRUE(net_.regMrs.empty());
}

TEST_F(NetProxyRegBufferMicrotest, Reg_HsaDmaBufFails_FallsBackToRegMr) {
  pfn_hsa_amd_portable_export_dmabuf = [](const void*, size_t, int*, uint64_t*) { return HSA_STATUS_ERROR; };
  sendRes_.useDmaBuf = recvRes_.useDmaBuf = 1;
  for (bool send : {true, false}) ASSERT_EQ(Reg(send), ncclSuccess);

  EXPECT_EQ(net_.regMrs.size(), 2u);
}

TEST_F(NetProxyRegBufferMicrotest, Reg_DmaBufNotSelected_UsesRegMr) {
  pfn_hsa_amd_portable_export_dmabuf = FakeExportDmaBuf;
  for (bool send : {true, false}) ASSERT_EQ(Reg(send), ncclSuccess);

  EXPECT_TRUE(net_.regMrDmaBufs.empty());
  EXPECT_EQ(net_.regMrs.size(), 2u);
}

#if CUDA_VERSION >= 11070 || NCCL_CUMEM_DMABUF_EXPORT_GATE
class NetProxyRegBufferCuMemMicrotest : public NetProxyRegBufferMicrotest {
 protected:
  CuMemExportRecorder exports_;

  void SetUp() override {
    NetProxyRegBufferMicrotest::SetUp();
    g_cuMemEnable = [] { return 1; };
    sendRes_.useDmaBuf = recvRes_.useDmaBuf = 1;
    exports_.Install();
  }
  void TearDown() override {
    exports_.CloseOpen();
    NetProxyRegBufferMicrotest::TearDown();
  }
};

TEST_F(NetProxyRegBufferCuMemMicrotest, Reg_RegistersByExportedFdAndClosesIt) {
  info_.numSegments = 2;  // DMA-BUF handles multi-segment buffers
  for (bool send : {true, false}) {
    SCOPED_TRACE(send);
    net_.regMrDmaBufs.clear();

    ASSERT_EQ(Reg(send), ncclSuccess);

    ASSERT_EQ(net_.regMrDmaBufs.size(), 1u);
    EXPECT_EQ(net_.regMrDmaBufs[0].comm, Comm(send));
    EXPECT_EQ(net_.regMrDmaBufs[0].fd, exports_.back().fd);
    EXPECT_EQ(net_.regMrDmaBufs[0].offset, 0u);
    EXPECT_EQ(resp_, reinterpret_cast<void*>(0x10000));
    EXPECT_EQ(fcntl(exports_.back().fd, F_GETFD), -1);
  }
  EXPECT_TRUE(net_.regMrs.empty());
}

TEST_F(NetProxyRegBufferCuMemMicrotest, Reg_ExportFails_FallsBackToRegMr) {
  exports_.result = hipErrorNotSupported;
  for (bool send : {true, false}) ASSERT_EQ(Reg(send), ncclSuccess);

  EXPECT_TRUE(net_.regMrDmaBufs.empty());
  EXPECT_EQ(net_.regMrs.size(), 2u);
}

TEST_F(NetProxyRegBufferCuMemMicrotest, Reg_DmaBufRegistrationFails_ClosesFdAndFallsBackToRegMr) {
  net_.regMrDmaBufResult = ncclSystemError;
  for (bool send : {true, false}) {
    SCOPED_TRACE(send);

    ASSERT_EQ(Reg(send), ncclSuccess);

    EXPECT_EQ(fcntl(exports_.back().fd, F_GETFD), -1);
    EXPECT_EQ(resp_, reinterpret_cast<void*>(0x10000));
  }
  EXPECT_EQ(net_.regMrs.size(), 2u);
}
#endif

TEST_F(NetProxyRegBufferMicrotest, Dereg_WrongRequestSize_Rejected) {
  for (bool send : {true, false}) EXPECT_EQ(Dereg(send, nullptr, 4), ncclInternalError);
  EXPECT_TRUE(net_.deregMrs.empty());
}

TEST_F(NetProxyRegBufferMicrotest, Dereg_RegisteredHandle_ForgottenAndDeregistered) {
  for (bool send : {true, false}) {
    SCOPED_TRACE(send);
    net_.deregMrs.clear();
    ASSERT_EQ(Reg(send), ncclSuccess);
    void* handle = resp_;

    ASSERT_EQ(Dereg(send, handle), ncclSuccess);

    EXPECT_EQ(done_, 1);
    EXPECT_TRUE(ncclIntruQueueEmpty(&Conn(send).proxyMemHandleQueue));
    ASSERT_EQ(net_.deregMrs.size(), 1u);
    EXPECT_EQ(net_.deregMrs[0].first, Comm(send));
    EXPECT_EQ(net_.deregMrs[0].second, handle);
  }
}

TEST_F(NetProxyRegBufferMicrotest, Dereg_OneOfSeveralHandles_ForgetsOnlyThatOne) {
  for (bool send : {true, false}) {
    SCOPED_TRACE(send);
    info_.buffer = 0x10000;
    ASSERT_EQ(Reg(send), ncclSuccess);
    void* first = resp_;
    info_.buffer = 0x20000;
    ASSERT_EQ(Reg(send), ncclSuccess);

    ASSERT_EQ(Dereg(send, resp_), ncclSuccess);

    ASSERT_FALSE(ncclIntruQueueEmpty(&Conn(send).proxyMemHandleQueue));
    EXPECT_EQ(ncclIntruQueueHead(&Conn(send).proxyMemHandleQueue)->handle, first);
    EXPECT_EQ(ncclIntruQueueHead(&Conn(send).proxyMemHandleQueue)->next, nullptr);
  }
}

TEST_F(NetProxyRegBufferMicrotest, Dereg_PluginFails_ReturnsError) {
  net_.deregMrResult = ncclSystemError;
  for (bool send : {true, false}) EXPECT_EQ(Dereg(send, reinterpret_cast<void*>(0x5)), ncclSystemError);
}

}  // namespace
