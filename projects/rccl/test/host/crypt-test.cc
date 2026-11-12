/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/misc/crypt.cc as built without a TLS backend: the encryption-mode
// API and the plaintext socket hello. The socket layer is seamed in this TU.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>

#include "nccl.h"
#include "debug.h"
#include "socket.h"
#include "crypt.h"

// --- Logging: this binary links no shared nccl fakes.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;  // log everything
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;

namespace {
struct LoggedMessage {
  ncclDebugLogLevel level;
  std::string text;
};
std::vector<LoggedMessage> g_logged;

bool Logged(ncclDebugLogLevel level, const char* needle) {
  for (const auto& m : g_logged) {
    if (m.level == level && m.text.find(needle) != std::string::npos) return true;
  }
  return false;
}
}  // namespace

void ncclDebugLog(ncclDebugLogLevel level, unsigned long /*flags*/, const char* /*filefunc*/, int /*line*/,
                  const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  g_logged.push_back({level, buf});
}

// --- Socket layer seams.
namespace {
struct ProgressCall {
  int op;
  int size;
  int offsetBefore;
};
std::vector<ProgressCall> g_progressCalls;
// Each call moves *offset by this many bytes (capped at size), copies g_peerBytes into a recv
// buffer, and reports closed/result as scripted.
int g_progressBytes = 0;
int g_progressClosed = 0;
ncclResult_t g_progressResult = ncclSuccess;
std::vector<char> g_peerBytes;
std::vector<int> g_polls;
int64_t g_pollTimeOut = 0;
}  // namespace

int64_t ncclParamPollTimeOut() { return g_pollTimeOut; }

ncclResult_t ncclOsSocketProgressOpt(int op, struct ncclSocket* /*sock*/, void* ptr, int size, int* offset,
                                     int /*block*/, int* closed) {
  g_progressCalls.push_back({op, size, *offset});
  if (g_progressResult != ncclSuccess) return g_progressResult;
  int n = std::min(g_progressBytes, size - *offset);
  if (op == NCCL_SOCKET_RECV && n > 0 && (size_t)(*offset + n) <= g_peerBytes.size()) {
    std::memcpy((char*)ptr + *offset, g_peerBytes.data() + *offset, n);
  }
  *offset += n;
  *closed = g_progressClosed;
  return ncclSuccess;
}

void ncclOsPollSocket(ncclSocketDescriptor /*sock*/, int op) { g_polls.push_back(op); }

const char* ncclSocketToString(const union ncclSocketAddress* /*addr*/, char* buf, size_t bufLen,
                               const int /*numericHostForm*/) {
  std::snprintf(buf, bufLen, "peer");
  return buf;
}

// The no-TLS arm under test; a -DTLS_BACKEND=OPENSSL3 build defines this project-wide.
#undef NCCL_TLS_BACKEND_OPENSSL3
#include CRYPT_CC_PATH

namespace {

constexpr int kHelloBytes = NCCL_SOCKET_PLAIN_HELLO_BYTES;
constexpr uint64_t kMagic = 0x1234567890abcdefull;
const char kPsk32[] = "0123456789abcdef0123456789abcdef";  // exactly ncclPskMinBytes

class CryptMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(ncclSetEncryption(nullptr), ncclSuccess);  // back to plaintext, no stored PSK
    g_logged.clear();
    g_progressCalls.clear();
    g_polls.clear();
    g_progressBytes = kHelloBytes;
    g_progressClosed = 0;
    g_progressResult = ncclSuccess;
    g_pollTimeOut = 0;
    std::memset(&sock, 0, sizeof(sock));
    sock.magic = kMagic;
    sock.type = ncclSocketTypeBootstrap;
    SetPeerHello(kMagic, ncclSocketTypeBootstrap);
  }
  void TearDown() override { ncclSetEncryption(nullptr); }

  static void SetPeerHello(uint64_t magic, ncclSocketType type) {
    g_peerBytes.assign(kHelloBytes, 0);
    std::memcpy(g_peerBytes.data(), &magic, sizeof(magic));
    std::memcpy(g_peerBytes.data() + sizeof(magic), &type, sizeof(type));
  }

  static ncclEncryptionConfig_t Config(int mode, const char* psk) {
    ncclEncryptionConfig_t config = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
    config.mode = mode;
    config.psk = psk;
    return config;
  }

  ncclSocket sock;
};

// Encryption mode API

TEST_F(CryptMicrotest, SetEncryption_NullConfigIsPlaintext) {
  ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_PSK, kPsk32);
  ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  ASSERT_EQ(ncclSetEncryption(nullptr), ncclSuccess);
  EXPECT_EQ(ncclCryptApiPsk, nullptr);
  EXPECT_EQ(ncclCryptApiPskLen, 0u);
  bool encrypted = true;
  ASSERT_EQ(ncclGetCryptConnectionMode(&encrypted), ncclSuccess);
  EXPECT_FALSE(encrypted);
  EXPECT_EQ(ncclCryptApiMode, NCCL_ENCRYPTION_MODE_NONE);
}

TEST_F(CryptMicrotest, SetEncryption_RejectsAConfigWithoutTheInitializerMagic) {
  ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_NONE, nullptr);
  config.magic = 0;
  EXPECT_EQ(ncclSetEncryption(&config), ncclInvalidArgument);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "NCCL_ENCRYPTION_CONFIG_INITIALIZER"));
}

TEST_F(CryptMicrotest, SetEncryption_PskModeRequiresAPsk) {
  ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_PSK, nullptr);
  EXPECT_EQ(ncclSetEncryption(&config), ncclInvalidArgument);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "requires config->psk"));
  EXPECT_EQ(ncclCryptApiMode, NCCL_ENCRYPTION_MODE_NONE);
}

TEST_F(CryptMicrotest, SetEncryption_PskMustBeAtLeast32Bytes) {
  std::string shortPsk(kPsk32, 31);
  ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_PSK, shortPsk.c_str());
  EXPECT_EQ(ncclSetEncryption(&config), ncclInvalidArgument);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "at least 32 bytes"));
  EXPECT_EQ(ncclCryptApiPsk, nullptr);

  config.psk = kPsk32;
  ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  EXPECT_EQ(ncclCryptApiMode, NCCL_ENCRYPTION_MODE_PSK);
  ASSERT_EQ(ncclCryptApiPskLen, 32u);
  EXPECT_EQ(std::memcmp(ncclCryptApiPsk, kPsk32, 32), 0);
  EXPECT_NE((const void*)ncclCryptApiPsk, (const void*)kPsk32);  // a copy, not the caller's buffer
}

TEST_F(CryptMicrotest, SetEncryption_RejectsAnUnknownMode) {
  ncclEncryptionConfig_t config = Config(2, nullptr);
  EXPECT_EQ(ncclSetEncryption(&config), ncclInvalidArgument);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "config->mode=2"));
}

// Only config->size bytes are read: fields past it keep the initializer's plaintext defaults.
TEST_F(CryptMicrotest, SetEncryption_ReadsOnlyTheCallersSize) {
  ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_PSK, nullptr);
  config.size = offsetof(ncclEncryptionConfig_t, mode);
  ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  EXPECT_EQ(ncclCryptApiMode, NCCL_ENCRYPTION_MODE_NONE);
}

// A larger caller struct is clamped to the size this library knows.
TEST_F(CryptMicrotest, SetEncryption_ClampsALargerCallerSize) {
  struct {
    ncclEncryptionConfig_t config;
    char trailing[64];
  } big;
  big.config = Config(NCCL_ENCRYPTION_MODE_PSK, kPsk32);
  big.config.size = sizeof(big);
  std::memset(big.trailing, 0x5a, sizeof(big.trailing));
  ASSERT_EQ(ncclSetEncryption(&big.config), ncclSuccess);
  EXPECT_EQ(ncclCryptApiMode, NCCL_ENCRYPTION_MODE_PSK);
  EXPECT_EQ(ncclCryptApiPskLen, 32u);
}

TEST_F(CryptMicrotest, SetEncryption_BackToPlaintextDropsTheStoredPsk) {
  ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_PSK, kPsk32);
  ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  config = Config(NCCL_ENCRYPTION_MODE_NONE, nullptr);
  ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  EXPECT_EQ(ncclCryptApiMode, NCCL_ENCRYPTION_MODE_NONE);
  EXPECT_EQ(ncclCryptApiPsk, nullptr);
  EXPECT_EQ(ncclCryptApiPskLen, 0u);
}

// Without a TLS backend a configured PSK cannot be installed, so every connection fails loudly
// rather than silently going out in plaintext.
TEST_F(CryptMicrotest, ConnectionMode_PskWithoutTlsSupportIsInvalidUsage) {
  ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_PSK, kPsk32);
  ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  bool encrypted = true;
  EXPECT_EQ(ncclGetCryptConnectionMode(&encrypted), ncclInvalidUsage);
  EXPECT_FALSE(encrypted);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "built without TLS support"));
}

TEST_F(CryptMicrotest, ConnectionMode_UnknownStoredModeIsInvalidArgument) {
  ncclCryptApiMode = 7;
  bool encrypted = true;
  EXPECT_EQ(ncclGetCryptConnectionMode(&encrypted), ncclInvalidArgument);
  EXPECT_FALSE(encrypted);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "unsupported encryption mode 7"));
  ncclCryptApiMode = NCCL_ENCRYPTION_MODE_NONE;
}

// A stale installed key is dropped once the API is back to plaintext.
TEST_F(CryptMicrotest, ConnectionMode_PlaintextClearsAStaleMasterKey) {
  ncclCryptConfigKeyLen = 4;
  ncclCryptConfigKey = static_cast<unsigned char*>(std::calloc(4, 1));
  ncclCryptHaveKey = true;
  bool encrypted = true;
  ASSERT_EQ(ncclGetCryptConnectionMode(&encrypted), ncclSuccess);
  EXPECT_FALSE(encrypted);
  EXPECT_FALSE(ncclCryptHaveKey);
  EXPECT_EQ(ncclCryptConfigKey, nullptr);
  EXPECT_EQ(ncclCryptConfigKeyLen, 0u);
}

// A key that matches the API PSK is reused rather than re-derived. A no-TLS build never installs
// one itself, so these seed it to reach the key comparison.
class CryptInstalledKeyMicrotest : public CryptMicrotest {
 protected:
  void SetUp() override {
    CryptMicrotest::SetUp();
    ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_PSK, kPsk32);
    ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  }
  void TearDown() override {
    {
      std::lock_guard<std::mutex> lock(ncclCryptLock);
      cryptClearMasterKey();
    }
    CryptMicrotest::TearDown();
  }
  static void InstallKey(const char* key, size_t len) {
    ncclCryptConfigKey = static_cast<unsigned char*>(std::malloc(len));
    std::memcpy(ncclCryptConfigKey, key, len);
    ncclCryptConfigKeyLen = len;
    ncclCryptHaveKey = true;
  }
};

TEST_F(CryptInstalledKeyMicrotest, ConnectionMode_MatchingKeyIsEncrypted) {
  InstallKey(kPsk32, 32);
  bool encrypted = false;
  ASSERT_EQ(ncclGetCryptConnectionMode(&encrypted), ncclSuccess);
  EXPECT_TRUE(encrypted);
  EXPECT_FALSE(Logged(NCCL_LOG_WARN, "built without TLS support"));  // not re-derived
}

TEST_F(CryptInstalledKeyMicrotest, ConnectionMode_ADifferentKeyIsRederived) {
  std::string other(kPsk32);
  other[31] = 'X';  // same length, different bytes
  InstallKey(other.data(), 32);
  bool encrypted = false;
  EXPECT_EQ(ncclGetCryptConnectionMode(&encrypted), ncclInvalidUsage);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "built without TLS support"));

}

// The PSK plus one more byte: the first 32 bytes match, only the length differs.
TEST_F(CryptInstalledKeyMicrotest, ConnectionMode_ADifferentLengthIsRederived) {
  std::string longer = std::string(kPsk32) + "X";
  InstallKey(longer.data(), longer.size());
  bool encrypted = false;
  EXPECT_EQ(ncclGetCryptConnectionMode(&encrypted), ncclInvalidUsage);
}

TEST_F(CryptInstalledKeyMicrotest, StartConnectAndAcceptHello_EncryptedNeedTlsSupport) {
  InstallKey(kPsk32, 32);
  EXPECT_EQ(ncclCryptStartConnect(&sock), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "TLS connection requested without TLS support"));
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  EXPECT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclInternalError);
  EXPECT_TRUE(g_progressCalls.empty());
}

// Connect side

TEST_F(CryptMicrotest, StartConnect_PlaintextWritesMagicThenType) {
  ASSERT_EQ(ncclCryptStartConnect(&sock), ncclSuccess);
  uint64_t magic;
  ncclSocketType type;
  std::memcpy(&magic, sock.finalizeBuffer, sizeof(magic));
  std::memcpy(&type, sock.finalizeBuffer + sizeof(magic), sizeof(type));
  EXPECT_EQ(magic, kMagic);
  EXPECT_EQ(type, ncclSocketTypeBootstrap);
  EXPECT_EQ(sock.crypto, nullptr);
}

TEST_F(CryptMicrotest, StartConnect_PropagatesAModeError) {
  ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_PSK, kPsk32);
  ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  EXPECT_EQ(ncclCryptStartConnect(&sock), ncclInvalidUsage);
}

TEST_F(CryptMicrotest, ConnectHello_SendsTheWholeHelloInPlaintext) {
  ASSERT_EQ(ncclCryptConnectHello(&sock), ncclSuccess);
  ASSERT_EQ(g_progressCalls.size(), 1u);
  EXPECT_EQ(g_progressCalls[0].op, NCCL_SOCKET_SEND);
  EXPECT_EQ(g_progressCalls[0].size, kHelloBytes);
  EXPECT_EQ(sock.finalizeCounter, kHelloBytes);
  EXPECT_TRUE(g_polls.empty());
}

TEST_F(CryptMicrotest, ConnectHello_AlreadySentIsANoOp) {
  sock.finalizeCounter = kHelloBytes;
  EXPECT_EQ(ncclCryptConnectHello(&sock), ncclSuccess);
  EXPECT_TRUE(g_progressCalls.empty());
}

// A partial send is in progress; a blocking socket with a poll timeout waits for writability.
TEST_F(CryptMicrotest, ConnectHello_PartialSendIsInProgress) {
  g_progressBytes = 3;
  EXPECT_EQ(ncclCryptConnectHello(&sock), ncclInProgress);
  EXPECT_EQ(sock.finalizeCounter, 3);
  EXPECT_TRUE(g_polls.empty());  // no poll timeout configured

  g_pollTimeOut = 100;
  EXPECT_EQ(ncclCryptConnectHello(&sock), ncclInProgress);
  ASSERT_EQ(g_polls.size(), 1u);
  EXPECT_EQ(g_polls[0], NCCL_SOCKET_SEND);
  EXPECT_EQ(g_progressCalls.back().offsetBefore, 3);  // resumes where it left off

  sock.asyncFlag = 1;  // async sockets never block in poll
  g_polls.clear();
  EXPECT_EQ(ncclCryptConnectHello(&sock), ncclInProgress);
  EXPECT_TRUE(g_polls.empty());
}

TEST_F(CryptMicrotest, ConnectHello_PeerCloseIsARemoteError) {
  g_progressBytes = 0;
  g_progressClosed = 1;
  EXPECT_EQ(ncclCryptConnectHello(&sock), ncclRemoteError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "Connection closed by remote peer"));
}

TEST_F(CryptMicrotest, ConnectHello_PropagatesASocketError) {
  g_progressResult = ncclSystemError;
  EXPECT_EQ(ncclCryptConnectHello(&sock), ncclSystemError);
}

// A socket that somehow carries TLS state cannot handshake in this build.
TEST_F(CryptMicrotest, ConnectHello_TlsStateWithoutTlsSupportIsInternalError) {
  sock.crypto = reinterpret_cast<ncclSocketCrypto*>(std::calloc(1, sizeof(ncclSocketCrypto)));
  EXPECT_EQ(ncclCryptConnectHello(&sock), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "TLS handshake requested without TLS support"));
  EXPECT_TRUE(g_progressCalls.empty());
  ncclCryptFree(sock.crypto);
}

// Accept side

TEST_F(CryptMicrotest, AcceptHello_MatchingHelloIsReady) {
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
  ASSERT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclSuccess);
  EXPECT_EQ(verdict, NCCL_CRYPT_HELLO_VERDICT_READY);
  ASSERT_EQ(g_progressCalls.size(), 1u);
  EXPECT_EQ(g_progressCalls[0].op, NCCL_SOCKET_RECV);
}

TEST_F(CryptMicrotest, AcceptHello_AlreadyReceivedOnlyValidates) {
  std::memcpy(sock.finalizeBuffer, g_peerBytes.data(), kHelloBytes);
  sock.finalizeCounter = kHelloBytes;
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_RESET;
  ASSERT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclSuccess);
  EXPECT_EQ(verdict, NCCL_CRYPT_HELLO_VERDICT_READY);
  EXPECT_TRUE(g_progressCalls.empty());
}

TEST_F(CryptMicrotest, AcceptHello_WrongMagicResets) {
  SetPeerHello(kMagic ^ 1, ncclSocketTypeBootstrap);
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  ASSERT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclSuccess);
  EXPECT_EQ(verdict, NCCL_CRYPT_HELLO_VERDICT_RESET);
  EXPECT_TRUE(Logged(NCCL_LOG_ATTN, "Socket magic mismatch"));
}

TEST_F(CryptMicrotest, AcceptHello_WrongTypeResets) {
  SetPeerHello(kMagic, ncclSocketTypeProxy);
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  ASSERT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclSuccess);
  EXPECT_EQ(verdict, NCCL_CRYPT_HELLO_VERDICT_RESET);
  EXPECT_TRUE(Logged(NCCL_LOG_ATTN, "Wrong socket type"));
}

TEST_F(CryptMicrotest, AcceptHello_PartialReceiveIsAgain) {
  g_progressBytes = 5;
  g_pollTimeOut = 100;
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  EXPECT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclInProgress);
  EXPECT_EQ(verdict, NCCL_CRYPT_HELLO_VERDICT_AGAIN);
  EXPECT_EQ(sock.finalizeCounter, 5);
  ASSERT_EQ(g_polls.size(), 1u);
  EXPECT_EQ(g_polls[0], NCCL_SOCKET_RECV);

  g_progressBytes = kHelloBytes;  // the rest arrives
  ASSERT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclSuccess);
  EXPECT_EQ(verdict, NCCL_CRYPT_HELLO_VERDICT_READY);
}

// One bad connection must not fail the accept loop: a close or a remote error just resets.
TEST_F(CryptMicrotest, AcceptHello_PeerCloseResets) {
  g_progressBytes = 0;
  g_progressClosed = 1;
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  ASSERT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclSuccess);
  EXPECT_EQ(verdict, NCCL_CRYPT_HELLO_VERDICT_RESET);
  EXPECT_TRUE(Logged(NCCL_LOG_ATTN, "closed before socket hello"));
}

TEST_F(CryptMicrotest, AcceptHello_RemoteErrorResets) {
  g_progressResult = ncclRemoteError;
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  ASSERT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclSuccess);
  EXPECT_EQ(verdict, NCCL_CRYPT_HELLO_VERDICT_RESET);
  EXPECT_TRUE(Logged(NCCL_LOG_ATTN, "Handshake receive from peer failed"));
}

TEST_F(CryptMicrotest, AcceptHello_OtherSocketErrorsPropagate) {
  g_progressResult = ncclSystemError;
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  EXPECT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclSystemError);
  EXPECT_EQ(verdict, NCCL_CRYPT_HELLO_VERDICT_AGAIN);
}

TEST_F(CryptMicrotest, AcceptHello_PropagatesAModeError) {
  ncclEncryptionConfig_t config = Config(NCCL_ENCRYPTION_MODE_PSK, kPsk32);
  ASSERT_EQ(ncclSetEncryption(&config), ncclSuccess);
  ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_READY;
  EXPECT_EQ(ncclCryptAcceptHello(&sock, &verdict), ncclInvalidUsage);
  EXPECT_TRUE(g_progressCalls.empty());
}

// Encrypted I/O and state

TEST_F(CryptMicrotest, SocketSendAndRecv_WithoutTlsSupportAreInternalErrors) {
  char buf[4] = {};
  int offset = 0;
  bool closed = false;
  EXPECT_EQ(ncclCryptSocketSend(&sock, buf, sizeof(buf), &offset, &closed), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "ncclCryptSocketSend called without TLS support"));
  EXPECT_EQ(ncclCryptSocketRecv(&sock, buf, sizeof(buf), &offset, &closed), ncclInternalError);
  EXPECT_TRUE(Logged(NCCL_LOG_WARN, "ncclCryptSocketRecv called without TLS support"));
  EXPECT_TRUE(g_progressCalls.empty());
}

// Without TLS state there is nothing to free or rebind, so the socket is left byte-for-byte alone.
TEST_F(CryptMicrotest, FreeAndRebind_LeaveAPlainSocketUntouched) {
  ncclCryptFree(nullptr);
  ncclSocket before = sock;
  ncclCryptRebindSocket(&sock);
  EXPECT_EQ(0, std::memcmp(&before, &sock, sizeof(sock)));
  EXPECT_EQ(sock.crypto, nullptr);
}

}  // namespace
