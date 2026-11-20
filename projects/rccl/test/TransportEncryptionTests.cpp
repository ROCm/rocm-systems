/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclSetEncryption(): a stored PSK must bring the bootstrap up over TLS or refuse it, never plaintext.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

constexpr char kPsk[] = "rccl-transport-encryption-key-32";  // the 32-byte minimum

ncclEncryptionConfig_t makeConfig(int mode, const char* psk) {
  ncclEncryptionConfig_t config = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
  config.mode = mode;
  config.psk = psk;
  return config;
}

// The one rank connects to the bootstrap root that ncclGetUniqueId started.
ncclResult_t initOneRank(ncclComm_t* comm) {
  ncclUniqueId id;
  ncclResult_t res = ncclGetUniqueId(&id);
  return res != ncclSuccess ? res : ncclCommInitRank(comm, 1, id, 0);
}

}  // namespace

TEST(TransportEncryption, PskGatesBootstrapSockets) {
  RUN_ISOLATED_TEST("TransportEncryption.PskGatesBootstrapSockets", []() {
    // Refused: no initializer magic, no key, a 31-byte key, an unknown mode.
    ncclEncryptionConfig_t config = makeConfig(NCCL_ENCRYPTION_MODE_PSK, kPsk);
    config.magic = 0;
    EXPECT_EQ(ncclSetEncryption(&config), ncclInvalidArgument);
    config = makeConfig(NCCL_ENCRYPTION_MODE_PSK, nullptr);
    EXPECT_EQ(ncclSetEncryption(&config), ncclInvalidArgument);
    config = makeConfig(NCCL_ENCRYPTION_MODE_PSK, kPsk + 1);
    EXPECT_EQ(ncclSetEncryption(&config), ncclInvalidArgument);
    config = makeConfig(NCCL_ENCRYPTION_MODE_PSK + 1, kPsk);
    EXPECT_EQ(ncclSetEncryption(&config), ncclInvalidArgument);

    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1) GTEST_SKIP() << "requires a GPU";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    config = makeConfig(NCCL_ENCRYPTION_MODE_PSK, kPsk);
    ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
    ncclComm_t comm = nullptr;
    ncclResult_t res = initOneRank(&comm);
    if (res == ncclSuccess) {
      // Only a TLS_BACKEND=OPENSSL3 build may bring the bootstrap up, and only over TLS.
      EXPECT_NE(dlopen("libssl.so.3", RTLD_NOW | RTLD_NOLOAD), nullptr)
          << "a PSK was set, yet the bootstrap came up without loading OpenSSL";
      EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
    } else {  // the failed comm is left to the isolated child's exit
      EXPECT_EQ(res, ncclInvalidUsage) << "without a TLS backend the first connect must refuse the PSK";
    }

    // A NULL config resets to NCCL_ENCRYPTION_MODE_NONE: the same init then runs in plaintext.
    ASSERT_EQ(ncclSetEncryption(nullptr), ncclSuccess);
    ASSERT_EQ(initOneRank(&comm), ncclSuccess);
    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
  });
}

// Control: the same init, never calling ncclSetEncryption.
TEST(TransportEncryption, PlaintextBootstrapByDefault) {
  RUN_ISOLATED_TEST("TransportEncryption.PlaintextBootstrapByDefault", []() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1) GTEST_SKIP() << "requires a GPU";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ncclComm_t comm = nullptr;
    ASSERT_EQ(initOneRank(&comm), ncclSuccess);
    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
  });
}

}  // namespace RcclUnitTesting
