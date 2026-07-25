/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Suite G: gin_plugin_anvil_sdma.cc host plugin unit tests.

#include "gin_anvil_plugin_test_stubs.h"

#include "gin/gin_host_anvil_sdma.h"
#include "comm.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_ipc_table.h"
#include "nccl_device/gin/anvil_sdma/gin_anvil_sdma_device_host_common.h"
#include "nccl_device/net_device.h"
#include "plugin/nccl_gin.h"

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <cstdlib>
#include <cstring>
#include <vector>

namespace RcclUnitTesting
{

class ScopedEnv {
 public:
  ScopedEnv(const char* name, const char* value) : name_(name) {
    if (const char* prev = getenv(name)) {
      had_ = true;
      prev_ = prev;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnv() {
    if (had_) {
      setenv(name_, prev_.c_str(), 1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  std::string prev_;
  bool had_{false};
};

struct GinAnvilMockComm {
  ncclComm comm{};
  char bootstrapPlaceholder{0};

  GinAnvilMockComm() { reset(); }

  void reset() {
    std::memset(&comm, 0, sizeof(comm));
    comm.bootstrap = &bootstrapPlaceholder;
    comm.devrState.lsaSelf = 0;
    comm.devrState.lsaSize = 2;
    comm.devrState.bigSize = 0x100000;
  }

  ncclComm* get() { return &comm; }
};

class GinAnvilPluginTest : public ::testing::Test {
 protected:
  ncclGin_t& plugin_ = ncclGinAnvilSdmaPlugin;
  GinAnvilMockComm mockComm_;

  void SetUp() override {
    ncclGinAnvilPluginTestResetHostState();
    ncclGinAnvilIpcTableTestReset();
    GinAnvilPluginStubs::Reset();
    mockComm_.reset();
    int ndev = 0;
    if (hipGetDeviceCount(&ndev) == hipSuccess && ndev > 0) {
      ASSERT_EQ(hipSetDevice(0), hipSuccess);
    }
    setenv("NCCL_GIN_TYPE", "5", 1);
    GinAnvilPluginStubs::SetProbeResult(1);
    GinAnvilPluginStubs::SetBootstrapNranks(1);
  }

  void TearDown() override { unsetenv("NCCL_GIN_TYPE"); }

  void initCtx(void** ictx) {
    ASSERT_EQ(plugin_.init(ictx, 0, nullptr), ncclSuccess);
    ncclGinAnvilSetInitContext(*ictx, mockComm_.get());
  }

  void connectColl(void* ictx, void** coll) {
    void* listen = nullptr;
    char handle[NCCL_NET_HANDLE_MAXSIZE] = {};
    ASSERT_EQ(plugin_.listen(ictx, 0, handle, &listen), ncclSuccess);
    void* handles[1] = {handle};
    ASSERT_EQ(plugin_.connect(ictx, handles, 1, 0, listen, coll), ncclSuccess);
    ASSERT_EQ(plugin_.closeListen(listen), ncclSuccess);
  }
};

// G1: wrong NCCL_GIN_TYPE.
TEST_F(GinAnvilPluginTest, Init_WrongGinType) {
  ScopedEnv ty("NCCL_GIN_TYPE", "4");
  void* ictx = nullptr;
  EXPECT_EQ(plugin_.init(&ictx, 0, nullptr), ncclInternalError);
  EXPECT_EQ(ictx, nullptr);
}

// G2: probe failure.
TEST_F(GinAnvilPluginTest, Init_ProbeFails) {
  GinAnvilPluginStubs::SetProbeResult(0);
  void* ictx = nullptr;
  EXPECT_EQ(plugin_.init(&ictx, 0, nullptr), ncclInternalError);
}

// G3: devices + properties.
TEST_F(GinAnvilPluginTest, Devices_GetProperties) {
  int ndev = -1;
  ASSERT_EQ(plugin_.devices(&ndev), ncclSuccess);
  EXPECT_EQ(ndev, 1);

  ncclNetProperties_v12_t props{};
  ASSERT_EQ(plugin_.getProperties(0, &props), ncclSuccess);
  EXPECT_STREQ(props.name, "gin-anvil-sdma");
  EXPECT_EQ(props.netDeviceType, NCCL_NET_DEVICE_GIN_ANVIL_SDMA);
  EXPECT_EQ(props.netDeviceVersion, NCCL_GIN_ANVIL_SDMA_NET_VERSION);
}

// G4: connect bootstrap failure.
TEST_F(GinAnvilPluginTest, Connect_BootstrapFail) {
  GinAnvilPluginStubs::SetBootstrapFail(true);
  void* ictx = nullptr;
  initCtx(&ictx);
  void* listen = nullptr;
  char handle[NCCL_NET_HANDLE_MAXSIZE] = {};
  ASSERT_EQ(plugin_.listen(ictx, 0, handle, &listen), ncclSuccess);
  void* coll = nullptr;
  void* handles[1] = {handle};
  EXPECT_EQ(plugin_.connect(ictx, handles, 1, 0, listen, &coll), ncclSystemError);
  EXPECT_EQ(coll, nullptr);
  plugin_.closeListen(listen);
  plugin_.finalize(ictx);
}

// G5: factory create failure.
TEST_F(GinAnvilPluginTest, Connect_FactoryFail) {
  GinAnvilPluginStubs::SetFactoryCreateFail(true);
  void* ictx = nullptr;
  initCtx(&ictx);
  void* listen = nullptr;
  char handle[NCCL_NET_HANDLE_MAXSIZE] = {};
  ASSERT_EQ(plugin_.listen(ictx, 0, handle, &listen), ncclSuccess);
  void* coll = nullptr;
  void* handles[1] = {handle};
  EXPECT_EQ(plugin_.connect(ictx, handles, 1, 0, listen, &coll), ncclSystemError);
  plugin_.closeListen(listen);
  plugin_.finalize(ictx);
}

// G6: successful connect + closeColl.
TEST_F(GinAnvilPluginTest, Connect_Success) {
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  ASSERT_NE(coll, nullptr);
  EXPECT_EQ(plugin_.closeColl(coll), ncclSuccess);
  EXPECT_EQ(plugin_.finalize(ictx), ncclSuccess);
}

// G7: regMrSym LSA resolution failure.
TEST_F(GinAnvilPluginTest, RegMrSym_LsaFail) {
  GinAnvilPluginStubs::SetLsaAddrFail(true);
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  void* mhandle = nullptr;
  void* ginHandle = nullptr;
  void* data = reinterpret_cast<void*>(0x1000);
  EXPECT_EQ(plugin_.regMrSym(coll, data, 4096, 0, 0, &mhandle, &ginHandle), ncclSystemError);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G8: refcount on duplicate registration.
TEST_F(GinAnvilPluginTest, RegMrSym_Refcount) {
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  void* data = reinterpret_cast<void*>(0x2000);
  void* mh1 = nullptr;
  void* gh1 = nullptr;
  void* mh2 = nullptr;
  void* gh2 = nullptr;
  ASSERT_EQ(plugin_.regMrSym(coll, data, 4096, 0, 0, &mh1, &gh1), ncclSuccess);
  ASSERT_EQ(plugin_.regMrSym(coll, data, 4096, 0, 0, &mh2, &gh2), ncclSuccess);
  EXPECT_NE(mh1, mh2);
  EXPECT_EQ(plugin_.deregMrSym(coll, mh1), ncclSuccess);
  EXPECT_EQ(plugin_.deregMrSym(coll, mh2), ncclSuccess);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G9: regMrSymDmaBuf delegates to regMrSym.
TEST_F(GinAnvilPluginTest, RegMrSymDmaBuf) {
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  void* mhandle = nullptr;
  void* ginHandle = nullptr;
  ASSERT_EQ(plugin_.regMrSymDmaBuf(coll, reinterpret_cast<void*>(0x3000), 1024, 0, 0, -1, 0, &mhandle,
                                  &ginHandle),
            ncclSuccess);
  EXPECT_EQ(plugin_.deregMrSym(coll, mhandle), ncclSuccess);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G10: createContext missing SDMA infrastructure.
TEST_F(GinAnvilPluginTest, CreateContext_MissingInfra) {
  GinAnvilPluginStubs::SetFactoryNullHandles(true);
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  ncclGinConfig_v13_t cfg{};
  cfg.nSignals = 0;
  cfg.nCounters = 0;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  EXPECT_EQ(plugin_.createContext(coll, &cfg, &ginCtx, &devHandle), ncclSystemError);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G11: createContext success + env-driven fields.
TEST_F(GinAnvilPluginTest, CreateContext_EnvAndCounters) {
  ScopedEnv th("NCCL_GIN_ANVIL_SDMA_THRESHOLD", "256");
  ScopedEnv fs("NCCL_GIN_ANVIL_SDMA_FUSED_SIGNAL", "1");
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  ncclGinConfig_v13_t cfg{};
  cfg.nSignals = 2;
  cfg.nCounters = 1;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll, &cfg, &ginCtx, &devHandle), ncclSuccess);
  ASSERT_NE(devHandle, nullptr);
  EXPECT_EQ(devHandle->netDeviceType, NCCL_NET_DEVICE_GIN_ANVIL_SDMA);

  ncclGinAnvilSdmaGPUContext hostCtx{};
  ASSERT_EQ(hipMemcpy(&hostCtx, devHandle->handle, sizeof(hostCtx), hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_EQ(hostCtx.layoutMagic, NCCL_GIN_ANVIL_SDMA_LAYOUT_MAGIC);
  EXPECT_EQ(hostCtx.sdmaThreshold, 256u);
  EXPECT_EQ(hostCtx.fusedSdmaSignal, 1u);
  EXPECT_NE(hostCtx.counters, nullptr);

  EXPECT_EQ(plugin_.destroyContext(ginCtx), ncclSuccess);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G12: bind resource window signals — invalid args.
TEST_F(GinAnvilPluginTest, BindSignals_InvalidArgs) {
  EXPECT_EQ(ncclGinAnvilBindResourceWindowSignals(nullptr, reinterpret_cast<void*>(0x100), 0, 1, 1),
            ncclInvalidArgument);
  EXPECT_EQ(ncclGinAnvilBindResourceWindowSignals(mockComm_.get(), nullptr, 0, 1, 1),
            ncclInvalidArgument);
  EXPECT_EQ(ncclGinAnvilBindResourceWindowSignals(mockComm_.get(), reinterpret_cast<void*>(0x100), 0, 0,
                                                  1),
            ncclInvalidArgument);
}

// G13: signal slot out of range (two pending contexts, one slot).
TEST_F(GinAnvilPluginTest, BindSignals_SlotOutOfRange) {
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  ncclGinConfig_v13_t cfg{};
  cfg.nSignals = 1;
  void* ginCtx1 = nullptr;
  void* ginCtx2 = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll, &cfg, &ginCtx1, &devHandle), ncclSuccess);
  ASSERT_EQ(plugin_.createContext(coll, &cfg, &ginCtx2, &devHandle), ncclSuccess);

  char arena[4096] = {};
  EXPECT_EQ(ncclGinAnvilBindResourceWindowSignals(mockComm_.get(), arena, 0, 1, 1),
            ncclInvalidArgument);

  plugin_.destroyContext(ginCtx1);
  plugin_.destroyContext(ginCtx2);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G14: successful signal bind clears pending list.
TEST_F(GinAnvilPluginTest, BindSignals_Success) {
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  ncclGinConfig_v13_t cfg{};
  cfg.nSignals = 2;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll, &cfg, &ginCtx, &devHandle), ncclSuccess);

  char arena[4096] = {};
  EXPECT_EQ(ncclGinAnvilBindResourceWindowSignals(mockComm_.get(), arena, 0, 1, 2), ncclSuccess);

  ncclGinAnvilSdmaGPUContext hostCtx{};
  ASSERT_EQ(hipMemcpy(&hostCtx, devHandle->handle, sizeof(hostCtx), hipMemcpyDeviceToHost), hipSuccess);
  EXPECT_NE(hostCtx.signals, nullptr);

  plugin_.destroyContext(ginCtx);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G15: dereg null, progress, queryLastError.
TEST_F(GinAnvilPluginTest, Misc_NoOpPaths) {
  EXPECT_EQ(plugin_.deregMrSym(nullptr, nullptr), ncclSuccess);
  EXPECT_EQ(plugin_.destroyContext(nullptr), ncclSuccess);
  bool hasError = true;
  EXPECT_EQ(plugin_.queryLastError(nullptr, &hasError), ncclSuccess);
  EXPECT_FALSE(hasError);
  EXPECT_EQ(plugin_.ginProgress(nullptr), ncclSuccess);
}

// G16: env int parsing via channels (invalid -> default 1).
TEST_F(GinAnvilPluginTest, Connect_EnvNumChannelsClamp) {
  ScopedEnv ch("NCCL_GIN_ANVIL_SDMA_NUM_CHANNELS", "0");
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  ASSERT_NE(coll, nullptr);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G17: bind signals — LSA flat addr resolution fails.
TEST_F(GinAnvilPluginTest, BindSignals_LsaResolveFail) {
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  ncclGinConfig_v13_t cfg{};
  cfg.nSignals = 1;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll, &cfg, &ginCtx, &devHandle), ncclSuccess);

  GinAnvilPluginStubs::SetLsaAddrFail(true);
  char arena[4096] = {};
  EXPECT_EQ(ncclGinAnvilBindResourceWindowSignals(mockComm_.get(), arena, 0, 1, 1), ncclSystemError);

  plugin_.destroyContext(ginCtx);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G18: bind signals — IPC table full during LSA signal registration.
TEST_F(GinAnvilPluginTest, BindSignals_IpcTableFull) {
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  ncclGinConfig_v13_t cfg{};
  cfg.nSignals = 1;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll, &cfg, &ginCtx, &devHandle), ncclSuccess);

  std::vector<void*> locals;
  locals.reserve(NCCL_GIN_ANVIL_IPC_MAX_BUFS);
  for (int i = 0; i < NCCL_GIN_ANVIL_IPC_MAX_BUFS; ++i) {
    void* local = reinterpret_cast<void*>(static_cast<uintptr_t>(0x50000000ULL + i * 0x10000ULL));
    ASSERT_EQ(ncclGinAnvilIpcTableRegisterVmm(local, 4096, 0, 2, 0x100000), 0);
    locals.push_back(local);
  }

  char arena[4096] = {};
  EXPECT_EQ(ncclGinAnvilBindResourceWindowSignals(mockComm_.get(), arena, 0, 1, 1), ncclSystemError);

  for (void* local : locals) {
    (void)ncclGinAnvilIpcTableUnregister(local);
  }
  plugin_.destroyContext(ginCtx);
  plugin_.closeColl(coll);
  plugin_.finalize(ictx);
}

// G19: closeColl tears down SDMA handle after successful bind.
TEST_F(GinAnvilPluginTest, CloseColl_AfterSignalBind) {
  void* ictx = nullptr;
  initCtx(&ictx);
  void* coll = nullptr;
  connectColl(ictx, &coll);
  ncclGinConfig_v13_t cfg{};
  cfg.nSignals = 1;
  void* ginCtx = nullptr;
  ncclNetDeviceHandle_v11_t* devHandle = nullptr;
  ASSERT_EQ(plugin_.createContext(coll, &cfg, &ginCtx, &devHandle), ncclSuccess);

  char arena[4096] = {};
  ASSERT_EQ(ncclGinAnvilBindResourceWindowSignals(mockComm_.get(), arena, 0, 1, 1), ncclSuccess);

  EXPECT_EQ(plugin_.destroyContext(ginCtx), ncclSuccess);
  EXPECT_EQ(plugin_.closeColl(coll), ncclSuccess);
  plugin_.finalize(ictx);
}

}  // namespace RcclUnitTesting
