/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/register/sendrecv_reg.cc, the send/recv buffer
// registration entry points. This TU #includes the hipified sendrecv_reg.cc
// (SENDRECV_REG_CC_PATH). No GPU, no librccl.so.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "ScopedHook.h"
#include "fakes/nccl_fakes.h"
#include "fakes/transport_p2p_reg_fakes.h"
#include "fakes/transport_reg_stubs.h"
#include "fakes/transport_stubs.h"

#include "fakes/transport_p2p_reg_seam.h"
#include SENDRECV_REG_CC_PATH
#include "fakes/transport_p2p_reg_seam_undef.h"

namespace {

void* Ptr(uintptr_t addr) { return reinterpret_cast<void*>(addr); }

constexpr size_t kSize = 0x4000;
constexpr int kPeer = 5;

class RegisterSendRecvMicrotestBase : public ::testing::Test {
 protected:
  std::unique_ptr<ncclComm> comm_ = std::make_unique<ncclComm>();
  ncclConnector conn_{};
  ncclCommCallbackQueue cleanupQueue_{};
  void* const buff_ = Ptr(0x100000);

  void SetUp() override { g_loadParam = ParamOverride({{"LOCAL_REGISTER", 1}, {"GRAPH_REGISTER", 1}}); }

  void TearDown() override {
    ResetNcclFakes();
    ResetTransportP2pRegFakes();
    ResetTransportStubs();
  }

  // HIP graph capture with graph registration enabled.
  void UseGraphCapture(int localRegister = 1) {
    comm_->planner.persistent = true;
    g_loadParam = ParamOverride({{"LOCAL_REGISTER", localRegister}, {"GRAPH_REGISTER", 1}});
  }
};

// ---------------------------------------------------------------------------
// ncclRegisterP2pNetBuffer
// ---------------------------------------------------------------------------

class RegisterP2pNetMicrotest : public RegisterSendRecvMicrotestBase {
 protected:
  int regFlag_ = 1;
  void* handle_ = nullptr;

  ncclResult_t Run() {
    return ncclRegisterP2pNetBuffer(comm_.get(), buff_, kSize, &conn_, &regFlag_, &handle_, &cleanupQueue_);
  }
};

TEST_F(RegisterP2pNetMicrotest, LocalRegisterEnabled_RegistersThroughTheOneConnector) {
  ScopedHook local(g_ncclNetLocalRegisterBuffer, [&](ncclComm* c, const void* buff, size_t size,
                                                     ncclConnector** conns, int nPeers, int* flag, void** handle) {
    EXPECT_EQ(c, comm_.get());
    EXPECT_EQ(buff, buff_);
    EXPECT_EQ(size, kSize);
    EXPECT_EQ(nPeers, 1);
    EXPECT_EQ(conns[0], &conn_);
    *flag = 1;
    *handle = Ptr(0xabc);
    return ncclSuccess;
  });
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(local.calls, 1);
  EXPECT_EQ(regFlag_, 1);
  EXPECT_EQ(handle_, Ptr(0xabc));
}

TEST_F(RegisterP2pNetMicrotest, LocalRegisterFails_LeavesBufferUnregistered) {
  ScopedHook local(g_ncclNetLocalRegisterBuffer,
                   [](ncclComm*, const void*, size_t, ncclConnector**, int, int*, void**) { return ncclSystemError; });
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(local.calls, 1);
  EXPECT_EQ(regFlag_, 0);
}

TEST_F(RegisterP2pNetMicrotest, LocalRegisterDisabled_LeavesBufferUnregistered) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 0}, {"GRAPH_REGISTER", 0}});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(regFlag_, 0);
}

TEST_F(RegisterP2pNetMicrotest, UnpackNetDevice_LeavesBufferUnregistered) {
  comm_->netDeviceType = NCCL_NET_DEVICE_UNPACK;
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(regFlag_, 0);
}

TEST_F(RegisterP2pNetMicrotest, GraphCapture_RegistersLocallyNotPerGraph) {
  UseGraphCapture();
  ScopedHook local(g_ncclNetLocalRegisterBuffer, [](ncclComm*, const void*, size_t, ncclConnector**, int, int* flag,
                                                    void**) {
    *flag = 1;
    return ncclSuccess;
  });
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(local.calls, 1);
  EXPECT_EQ(regFlag_, 1);
}

TEST_F(RegisterP2pNetMicrotest, GraphCaptureWithoutLocalRegister_LeavesBufferUnregistered) {
  UseGraphCapture(/*localRegister=*/0);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(regFlag_, 0);
}

// ---------------------------------------------------------------------------
// ncclRegisterP2pIpcBuffer
// ---------------------------------------------------------------------------

class RegisterP2pIpcMicrotest : public RegisterSendRecvMicrotestBase {
 protected:
  int regFlag_ = 1;
  void* regAddr_ = Ptr(0x5e17);
  uintptr_t peerAddrs_[1] = {};

  ncclResult_t Run() {
    return ncclRegisterP2pIpcBuffer(comm_.get(), buff_, kSize, kPeer, &regFlag_, &regAddr_, &cleanupQueue_);
  }

  // Local IPC registration reporting `registered`, at peer table peerAddrs_ and offset 0x40.
  void LocalRegisters(int registered) {
    g_ncclIpcLocalRegisterBuffer = [this, registered](ncclComm*, const void*, size_t, int*, int, ncclIpcRegType,
                                                      int* flag, uintptr_t* offset, uintptr_t** addrs) {
      *flag = registered;
      *offset = 0x40;
      *addrs = peerAddrs_;
      return ncclSuccess;
    };
  }
};

TEST_F(RegisterP2pIpcMicrotest, LocalRegisterEnabled_RegistersForTheOnePeer) {
  ScopedHook local(g_ncclIpcLocalRegisterBuffer,
                   [&](ncclComm* c, const void* buff, size_t size, int* peers, int nPeers, ncclIpcRegType type,
                       int* flag, uintptr_t*, uintptr_t**) {
                     EXPECT_EQ(c, comm_.get());
                     EXPECT_EQ(buff, buff_);
                     EXPECT_EQ(size, kSize);
                     EXPECT_EQ(nPeers, 1);
                     EXPECT_EQ(peers[0], kPeer);
                     EXPECT_EQ(type, NCCL_IPC_SENDRECV);
                     *flag = 0;
                     return ncclSuccess;
                   });
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(local.calls, 1);
}

TEST_F(RegisterP2pIpcMicrotest, Registered_ReturnsPeerAddressPlusOffset) {
  LocalRegisters(1);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(regFlag_, 1);
  EXPECT_EQ(regAddr_, reinterpret_cast<char*>(peerAddrs_) + 0x40);
}

TEST_F(RegisterP2pIpcMicrotest, NotRegistered_LeavesAddressUntouched) {
  LocalRegisters(0);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(regFlag_, 0);
  EXPECT_EQ(regAddr_, Ptr(0x5e17));
}

TEST_F(RegisterP2pIpcMicrotest, LocalRegisterFails_LeavesBufferUnregistered) {
  ScopedHook local(g_ncclIpcLocalRegisterBuffer, [](ncclComm*, const void*, size_t, int*, int, ncclIpcRegType, int*,
                                                    uintptr_t*, uintptr_t**) { return ncclSystemError; });
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(local.calls, 1);
  EXPECT_EQ(regFlag_, 0);
  EXPECT_EQ(regAddr_, Ptr(0x5e17));
}

TEST_F(RegisterP2pIpcMicrotest, LocalRegisterDisabled_LeavesBufferUnregistered) {
  g_loadParam = ParamOverride({{"LOCAL_REGISTER", 0}, {"GRAPH_REGISTER", 0}});
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(regFlag_, 0);
}

TEST_F(RegisterP2pIpcMicrotest, GraphCapture_RegistersLocallyNotPerGraph) {
  UseGraphCapture();
  LocalRegisters(1);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(regFlag_, 1);
}

TEST_F(RegisterP2pIpcMicrotest, GraphCaptureWithoutLocalRegister_LeavesBufferUnregistered) {
  UseGraphCapture(/*localRegister=*/0);
  EXPECT_EQ(Run(), ncclSuccess);
  EXPECT_EQ(regFlag_, 0);
}

}  // namespace
