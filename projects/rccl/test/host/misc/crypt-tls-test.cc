/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/crypt.cc as built with TLS_BACKEND=OPENSSL3: the OpenSSL
// contexts cryptInitLib() sets up. libssl.so.3 is dlopen'd at run time, as in the library.

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>

#include "nccl.h"
#include "debug.h"
#include "socket.h"
#include "crypt.h"

// --- Logging: this binary links no shared nccl fakes.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel level, unsigned long, const char*, int, const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::vfprintf(stderr, fmt, ap);
  va_end(ap);
  std::fputc('\n', stderr);
}

// --- Socket and OS seams. Context setup does no socket I/O.
int64_t ncclParamPollTimeOut() { return 0; }
ncclResult_t ncclOsSocketProgressOpt(int, struct ncclSocket*, void*, int, int*, int, int*) { return ncclInternalError; }
void ncclOsPollSocket(ncclSocketDescriptor, int) {}
const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, size_t bufLen, const int) {
  std::snprintf(buf, bufLen, "peer");
  return buf;
}
ncclOsLibraryHandle ncclOsDlopen(const char* filename) { return dlopen(filename, RTLD_NOW | RTLD_LOCAL); }
void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) { return dlsym(handle, symbol); }
void ncclOsDlclose(ncclOsLibraryHandle handle) { if (handle != nullptr) dlclose(handle); }
const char* ncclOsDlerror() { return dlerror(); }

#ifndef NCCL_TLS_BACKEND_OPENSSL3
#define NCCL_TLS_BACKEND_OPENSSL3 1
#endif
#include CRYPT_CC_PATH

namespace {

const char kPsk32[] = "0123456789abcdef0123456789abcdef";

class CryptTlsMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ncclEncryptionConfig_t config = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
    config.mode = NCCL_ENCRYPTION_MODE_PSK;
    config.psk = kPsk32;
    ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
    bool encrypted = false;
    ncclResult_t res = ncclGetCryptConnectionMode(&encrypted);  // derives the key: cryptInitLib()
    if (res != ncclSuccess && ncclCryptSslHandle == nullptr) GTEST_SKIP() << "libssl.so.3/libcrypto.so.3 not loadable";
    ASSERT_EQ(res, ncclSuccess);
    ASSERT_TRUE(encrypted);
    ASSERT_NE(ncclCryptServerCtx, nullptr);
    ASSERT_NE(ncclCryptClientCtx, nullptr);
  }
  void TearDown() override { ncclSetEncryption(nullptr); }
};

// Control: the contexts are set up as TLS 1.3 only, whether or not tickets are suppressed.
TEST_F(CryptTlsMicrotest, Control_ContextsAreTls13Only) {
  for (SSL_CTX* ctx : {ncclCryptServerCtx, ncclCryptClientCtx}) {
    EXPECT_EQ(ncclCryptOpenSsl.pfn_SSL_CTX_ctrl(ctx, SSL_CTRL_GET_MIN_PROTO_VERSION, 0, nullptr), TLS1_3_VERSION);
    EXPECT_EQ(ncclCryptOpenSsl.pfn_SSL_CTX_ctrl(ctx, SSL_CTRL_GET_MAX_PROTO_VERSION, 0, nullptr), TLS1_3_VERSION);
  }
}

// NCCL 2.32.6 "unread TLS session tickets": OpenSSL's TLS 1.3 server sends 2 session tickets after
// each handshake by default. A write-only bootstrap client never reads them, so its close() finds
// unread data and Linux sends RST, dropping the payload bytes still queued. The server context must
// issue none (SSL_OP_NO_TICKET alone does not do that: it only makes the tickets stateful).
TEST_F(CryptTlsMicrotest, ServerIssuesNoSessionTickets) {
  using GetNumTicketsFn = size_t (*)(const SSL_CTX*);
  auto getNumTickets = (GetNumTicketsFn)ncclOsDlsym(ncclCryptSslHandle, "SSL_CTX_get_num_tickets");
  ASSERT_NE(getNumTickets, nullptr);
  EXPECT_EQ(getNumTickets(ncclCryptServerCtx), 0u)
    << "the TLS server context still sends session tickets that write-only bootstrap clients never read";
}

}  // namespace
