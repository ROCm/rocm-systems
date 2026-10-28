/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NCCL 2.32.3 (AICOMRCCL-3381): ncclSetEncryption() turns on TLS for RCCL's own sockets.
// RCCL builds without a TLS backend unless TLS_BACKEND=OPENSSL3, so a stored PSK must then
// fail the next connect rather than fall back to plaintext. dlsym and a mirrored config
// keep this file building against an nccl.h that predates the API.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <dlfcn.h>

#include "common/ProcessIsolatedTestRunner.hpp"

namespace RcclUnitTesting {
namespace {

// Layout of ncclEncryptionConfig_t (ncclEncryptionConfig_v23200).
struct EncryptionConfig {
  size_t size;
  unsigned int magic;
  unsigned int version;
  int mode;  // NCCL_ENCRYPTION_MODE_NONE = 0, NCCL_ENCRYPTION_MODE_PSK = 1
  const char* psk;
};
using SetEncryptionFn = ncclResult_t (*)(const EncryptionConfig*);

constexpr int kModePsk = 1;
constexpr char kPsk[] = "rccl-transport-encryption-key-32";  // the 32-byte minimum

EncryptionConfig makeConfig(int mode, const char* psk) {
  return {sizeof(EncryptionConfig), NCCL_API_MAGIC, NCCL_VERSION_CODE, mode, psk};
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
    auto setEncryption = reinterpret_cast<SetEncryptionFn>(dlsym(RTLD_DEFAULT, "ncclSetEncryption"));
    ASSERT_NE(setEncryption, nullptr) << "librccl does not export ncclSetEncryption";

    // Refused: no initializer magic, no key, a 31-byte key, an unknown mode.
    EncryptionConfig config = makeConfig(kModePsk, kPsk);
    config.magic = 0;
    EXPECT_EQ(setEncryption(&config), ncclInvalidArgument);
    config = makeConfig(kModePsk, nullptr);
    EXPECT_EQ(setEncryption(&config), ncclInvalidArgument);
    config = makeConfig(kModePsk, kPsk + 1);
    EXPECT_EQ(setEncryption(&config), ncclInvalidArgument);
    config = makeConfig(kModePsk + 1, kPsk);
    EXPECT_EQ(setEncryption(&config), ncclInvalidArgument);

    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1) GTEST_SKIP() << "requires a GPU";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    config = makeConfig(kModePsk, kPsk);
    ASSERT_EQ(setEncryption(&config), ncclSuccess);
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
    ASSERT_EQ(setEncryption(nullptr), ncclSuccess);
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
