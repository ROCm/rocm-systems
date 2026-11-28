/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// ncclSetEncryption(): a stored PSK must bring the bootstrap up over TLS or refuse it, never plaintext.

#include <gtest/gtest.h>
#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include "bootstrap.h"
#include "common/ProcessIsolatedTestRunner.hpp"

#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

namespace RcclUnitTesting {
namespace {

constexpr char kPsk[] = "rccl-transport-encryption-key-32";  // the 32-byte minimum

ncclEncryptionConfig_t makeConfig(int mode, const char* psk) {
  ncclEncryptionConfig_t config = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
  config.mode = mode;
  config.psk = psk;
  return config;
}

#if defined(NCCL_TLS_BACKEND_OPENSSL3)
// Sends the root named by `id` a valid plaintext socket hello. Returns true if the
// root drops the connection, which only a listener that demands TLS does.
bool rootRefusesPlaintextHello(const ncclUniqueId& id) {
  struct ncclBootstrapHandle handle;
  memcpy(&handle, &id, sizeof(handle));
  int fd = socket(handle.addr.sa.sa_family, SOCK_STREAM, 0);
  if (fd < 0) return false;
  struct timeval timeout = {5, 0};  // a plaintext root keeps the socket open waiting for more
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  socklen_t len = handle.addr.sa.sa_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
  char hello[NCCL_SOCKET_PLAIN_HELLO_BYTES];
  enum ncclSocketType type = ncclSocketTypeBootstrap;
  memcpy(hello, &handle.magic, sizeof(handle.magic));
  memcpy(hello + sizeof(handle.magic), &type, sizeof(type));
  bool refused = false;
  if (connect(fd, &handle.addr.sa, len) == 0 && send(fd, hello, sizeof(hello), MSG_NOSIGNAL) == sizeof(hello)) {
    char buf[256];  // drain any TLS alert the root sends before it closes
    ssize_t n;
    while ((n = recv(fd, buf, sizeof(buf), 0)) > 0) {}
    refused = n == 0 || errno == ECONNRESET;
  }
  close(fd);
  return refused;
}
#endif

}  // namespace

TEST(TransportEncryption, PskGatesBootstrapSockets) {
  RUN_ISOLATED_TEST("TransportEncryption.PskGatesBootstrapSockets", []() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1) GTEST_SKIP() << "requires a GPU";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);

    ncclEncryptionConfig_t config = makeConfig(NCCL_ENCRYPTION_MODE_PSK, kPsk);
    ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
    // A refused config (31-byte key) must leave the stored PSK in force for the init below.
    config = makeConfig(NCCL_ENCRYPTION_MODE_PSK, kPsk + 1);
    EXPECT_EQ(ncclSetEncryption(&config), ncclInvalidArgument);
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    ncclComm_t comm = nullptr;
    ncclResult_t res = ncclCommInitRank(&comm, 1, id, 0);
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
    ASSERT_EQ(res, ncclSuccess) << "an OPENSSL3 build must bring the bootstrap up over TLS";
    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
    // A fresh root under the same PSK must drop a plaintext peer: the bootstrap required TLS.
    ncclUniqueId probeId;
    ASSERT_EQ(ncclGetUniqueId(&probeId), ncclSuccess);
    EXPECT_TRUE(rootRefusesPlaintextHello(probeId)) << "a PSK was set, yet the root accepted a plaintext hello";
#else  // the failed comm is left to the isolated child's exit
    EXPECT_EQ(res, ncclInvalidUsage) << "without a TLS backend the first connect must refuse the PSK";
#endif

    // A NULL config resets to NCCL_ENCRYPTION_MODE_NONE: the same init then runs in plaintext.
    ASSERT_EQ(ncclSetEncryption(nullptr), ncclSuccess);
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
  });
}

// Control: the same init, never calling ncclSetEncryption.
TEST(TransportEncryption, PlaintextBootstrapByDefault) {
  RUN_ISOLATED_TEST("TransportEncryption.PlaintextBootstrapByDefault", []() {
    int deviceCount = 0;
    if (hipGetDeviceCount(&deviceCount) != hipSuccess || deviceCount < 1) GTEST_SKIP() << "requires a GPU";
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    ncclUniqueId id;
    ASSERT_EQ(ncclGetUniqueId(&id), ncclSuccess);
    ncclComm_t comm = nullptr;
    ASSERT_EQ(ncclCommInitRank(&comm, 1, id, 0), ncclSuccess);
    EXPECT_EQ(ncclCommDestroy(comm), ncclSuccess);
  });
}

}  // namespace RcclUnitTesting
