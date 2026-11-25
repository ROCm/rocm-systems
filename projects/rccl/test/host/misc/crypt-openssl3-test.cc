/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for the TLS handshake in src/misc/crypt.cc.
//
// Contract: once ncclSetEncryption() sets a pre-shared key, a connecting socket
// finishes its handshake only with a peer that used that key. A server that
// ignores the PSK and presents a certificate instead is refused before the
// client sends it anything (NCCL 2.32.3, "crypt: require OpenSSL handshakes to
// use PSK path").
//
// The suite compiles the hipified crypt.cc directly (via CRYPT_CC_PATH) with the
// OpenSSL 3 backend, which librccl only gets from TLS_BACKEND=OPENSSL3. crypt.cc
// dlopens libssl.so.3, so building needs just the OpenSSL 3 headers. CMake builds
// this binary only when TLS_BACKEND=OPENSSL3. Both ends of a connection run in this process over a
// socketpair and are stepped in turn, non-blocking. No GPU, no librccl.so.

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#if !defined(NCCL_TLS_BACKEND_OPENSSL3)
#error "crypt-openssl3-test.cc needs NCCL_TLS_BACKEND_OPENSSL3 (configure with TLS_BACKEND=OPENSSL3)"
#endif

#include CRYPT_CC_PATH

// crypt.cc's calls into src/debug.cc, src/misc/socket.cc and src/os/linux.cc, which
// this binary does not link: logging (discarded), socket names, the default poll
// timeout (0: never poll), dl*, and a non-blocking send/recv with
// ncclOsSocketProgressOpt's closed/error contract.
uint32_t ncclDebugLevelMask = NCCL_DEBUG_LEVEL_MASK_UNINITIALIZED;
uint64_t ncclDebugMask = ~uint64_t(0);
thread_local int ncclDebugNoWarn = 0;
void ncclDebugLog(ncclDebugLogLevel, unsigned long, const char*, int, const char*, ...) {}
const char* ncclSocketToString(const union ncclSocketAddress*, char* buf, size_t bufLen, const int) {
  snprintf(buf, bufLen, "peer");
  return buf;
}
int64_t ncclParamPollTimeOut() { return 0; }
ncclOsLibraryHandle ncclOsDlopen(const char* name) { return dlopen(name, RTLD_NOW | RTLD_LOCAL); }
void* ncclOsDlsym(ncclOsLibraryHandle handle, const char* symbol) { return dlsym(handle, symbol); }
void ncclOsDlclose(ncclOsLibraryHandle handle) { if (handle) dlclose(handle); }
const char* ncclOsDlerror() { const char* err = dlerror(); return err ? err : ""; }
void ncclOsPollSocket(ncclSocketDescriptor, int) {}
ncclResult_t ncclOsSocketProgressOpt(int op, struct ncclSocket* sock, void* ptr, int size, int* offset, int,
                                     int* closed) {
  char* data = (char*)ptr + *offset;
  ssize_t n = op == NCCL_SOCKET_RECV ? recv(sock->socketDescriptor, data, size - *offset, MSG_DONTWAIT)
                                     : send(sock->socketDescriptor, data, size - *offset, MSG_DONTWAIT | MSG_NOSIGNAL);
  *closed = (n == 0 && op == NCCL_SOCKET_RECV) || (n < 0 && (errno == EPIPE || errno == ECONNRESET));
  if (n > 0) *offset += n;
  bool retry = n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR);
  return n >= 0 || *closed || retry ? ncclSuccess : ncclRemoteError;
}

namespace {

constexpr int kMaxSteps = 1000;  // a step only moves bytes already queued in the socketpair

void initSocket(struct ncclSocket* sock, int fd) {
  sock->socketDescriptor = fd;
  sock->asyncFlag = 1;
  sock->magic = NCCL_SOCKET_MAGIC;
  sock->type = ncclSocketTypeBootstrap;
}

class CryptPskHandshakeTest : public ::testing::Test {
 protected:
  int fds_[2] = {-1, -1};  // [0] the NCCL client, [1] its peer
  struct ncclSocket client_ = {};
  void* libssl_ = nullptr;

  void SetUp() override {
    libssl_ = dlopen("libssl.so.3", RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(nullptr, libssl_) << "TLS_BACKEND=OPENSSL3 but libssl.so.3 is not loadable: " << dlerror();
    ncclEncryptionConfig_t config = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
    config.mode = NCCL_ENCRYPTION_MODE_PSK;
    config.psk = "crypt-test pre-shared key, at least 32 bytes long";
    ASSERT_EQ(ncclSuccess, ncclSetEncryption(&config));
    ASSERT_EQ(0, socketpair(AF_UNIX, SOCK_STREAM, 0, fds_));
    initSocket(&client_, fds_[0]);
    ASSERT_EQ(ncclSuccess, ncclCryptStartConnect(&client_));
    ASSERT_NE(nullptr, client_.crypto) << "a PSK is set, so the connection must be encrypted";
  }

  void TearDown() override {
    ncclCryptFree(client_.crypto);
    for (int fd : fds_)
      if (fd >= 0) close(fd);
    ncclSetEncryption(nullptr);
    if (libssl_ != nullptr) dlclose(libssl_);
  }
};

// Both ends hold the PSK: the handshake completes and the server accepts the hello.
TEST_F(CryptPskHandshakeTest, PskPeersConnect) {
  struct ncclSocket server = {};
  initSocket(&server, fds_[1]);
  enum ncclCryptHelloVerdict verdict = NCCL_CRYPT_HELLO_VERDICT_AGAIN;
  ncclResult_t clientRes = ncclInProgress, serverRes = ncclInProgress;
  for (int i = 0; i < kMaxSteps && (clientRes == ncclInProgress || serverRes == ncclInProgress); i++) {
    if (clientRes == ncclInProgress) clientRes = ncclCryptConnectHello(&client_);
    if (serverRes == ncclInProgress) serverRes = ncclCryptAcceptHello(&server, &verdict);
  }
  ncclCryptFree(server.crypto);
  EXPECT_EQ(ncclSuccess, clientRes);
  EXPECT_EQ(ncclSuccess, serverRes);
  EXPECT_EQ(NCCL_CRYPT_HELLO_VERDICT_READY, verdict);
}

struct ImpostorOutcome {
  ncclResult_t clientRes = ncclInProgress;
  bool impostorHandshakeDone = false;
  int leaked = 0;
};

// A peer without the PSK ignores the client's PSK offer and completes a
// certificate handshake instead, which any client that skips verification accepts.
class CryptImpostorTest : public CryptPskHandshakeTest {
 protected:
  void ConnectToImpostor(bool clientSkipsVerification, ImpostorOutcome* out) {
#define CRYPT_TEST_SYMBOLS(X) \
  X(EVP_PKEY_Q_keygen) X(EVP_PKEY_free) X(EVP_sha256) X(X509_new) X(X509_free) X(X509_get_serialNumber) \
  X(ASN1_INTEGER_set) X(X509_getm_notBefore) X(X509_getm_notAfter) X(X509_gmtime_adj) X(X509_set_pubkey) \
  X(X509_sign) X(TLS_server_method) X(SSL_CTX_new) X(SSL_CTX_free) X(SSL_CTX_use_certificate) \
  X(SSL_CTX_use_PrivateKey) X(SSL_new) X(SSL_free) X(SSL_set_fd) X(SSL_accept) X(SSL_read) X(SSL_set_verify)
#define CRYPT_TEST_LOAD(name) \
  auto name = (decltype(&::name))dlsym(libssl_, #name); \
  ASSERT_TRUE(name != nullptr) << #name;
    CRYPT_TEST_SYMBOLS(CRYPT_TEST_LOAD)
#undef CRYPT_TEST_LOAD
#undef CRYPT_TEST_SYMBOLS
    if (clientSkipsVerification) SSL_set_verify(client_.crypto->ssl, SSL_VERIFY_NONE, nullptr);
    EVP_PKEY* key = EVP_PKEY_Q_keygen(nullptr, nullptr, "EC", "P-256");
    X509* cert = X509_new();
    SSL_CTX* ctx = SSL_CTX_new(TLS_server_method());
    ASSERT_TRUE(key && cert && ctx);
    // Without validity times the certificate does not parse and no client would accept it.
    ASN1_INTEGER_set(X509_get_serialNumber(cert), 1);
    X509_gmtime_adj(X509_getm_notBefore(cert), 0);
    X509_gmtime_adj(X509_getm_notAfter(cert), 3600);
    ASSERT_EQ(1, X509_set_pubkey(cert, key));
    ASSERT_GT(X509_sign(cert, key, EVP_sha256()), 0);
    ASSERT_EQ(1, SSL_CTX_use_certificate(ctx, cert));
    ASSERT_EQ(1, SSL_CTX_use_PrivateKey(ctx, key));
    SSL* impostor = SSL_new(ctx);
    ASSERT_EQ(0, fcntl(fds_[1], F_SETFL, O_NONBLOCK));
    ASSERT_EQ(1, SSL_set_fd(impostor, fds_[1]));

    for (int i = 0; i < kMaxSteps && !out->impostorHandshakeDone; i++) {
      if (out->clientRes == ncclInProgress) out->clientRes = ncclCryptConnectHello(&client_);
      out->impostorHandshakeDone = SSL_accept(impostor) == 1;
    }
    char hello[NCCL_SOCKET_PLAIN_HELLO_BYTES];
    out->leaked = SSL_read(impostor, hello, sizeof(hello));
    SSL_free(impostor);
    SSL_CTX_free(ctx);
    X509_free(cert);
    EVP_PKEY_free(key);
  }
};

// The client context's SSL_VERIFY_PEER refuses the impostor's certificate mid-handshake.
TEST_F(CryptImpostorTest, ClientRejectsServerWithoutPsk) {
  ImpostorOutcome out;
  ASSERT_NO_FATAL_FAILURE(ConnectToImpostor(false, &out));
  EXPECT_EQ(ncclRemoteError, out.clientRes) << "the client completed a TLS handshake that did not use the PSK";
  EXPECT_FALSE(out.impostorHandshakeDone) << "the client accepted the impostor's certificate";
  EXPECT_LE(out.leaked, 0) << "a server without the PSK decrypted the client's socket hello";
}

// Even when certificate verification is off, a completed handshake that did not use the PSK is refused.
TEST_F(CryptImpostorTest, ClientRejectsNonPskHandshakeWithoutVerification) {
  ImpostorOutcome out;
  ASSERT_NO_FATAL_FAILURE(ConnectToImpostor(true, &out));
  EXPECT_EQ(ncclRemoteError, out.clientRes) << "the client completed a TLS handshake that did not use the PSK";
  EXPECT_LE(out.leaked, 0) << "a server without the PSK decrypted the client's socket hello";
}

}  // namespace
