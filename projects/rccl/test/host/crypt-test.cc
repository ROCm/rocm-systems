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
// dlopens libssl.so.3, so building needs just the OpenSSL 3 headers; without
// them the tests skip. Both ends of a connection run in this process over a
// socketpair and are stepped in turn, non-blocking. No GPU, no librccl.so.

#include <gtest/gtest.h>

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#if __has_include(<openssl/opensslv.h>)
#include <openssl/opensslv.h>
#endif
#if defined(OPENSSL_VERSION_MAJOR) && OPENSSL_VERSION_MAJOR == 3 && !defined(NCCL_TLS_BACKEND_OPENSSL3)
#define NCCL_TLS_BACKEND_OPENSSL3 1
#endif

#include CRYPT_CC_PATH

// crypt.cc's calls into src/misc/socket.cc and src/os/linux.cc, which this
// binary does not link: the default poll timeout (0: never poll), dl*, and a
// non-blocking send/recv with ncclOsSocketProgressOpt's closed/error contract.
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
#if !defined(NCCL_TLS_BACKEND_OPENSSL3)
    GTEST_SKIP() << "built without the OpenSSL 3 headers, so crypt.cc has no TLS backend";
#endif
    libssl_ = dlopen("libssl.so.3", RTLD_NOW | RTLD_LOCAL);
    if (libssl_ == nullptr) GTEST_SKIP() << "libssl.so.3 is not loadable: " << dlerror();
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

// A peer without the PSK ignores the client's PSK offer and completes a
// certificate handshake instead, which any client that skips verification accepts.
// crypt.cc refuses it twice over: SSL_VERIFY_PEER on the client context, and
// SSL_session_reused() once the handshake completes.
TEST_F(CryptPskHandshakeTest, ClientRejectsServerWithoutPsk) {
#if defined(NCCL_TLS_BACKEND_OPENSSL3)
#define CRYPT_TEST_SYMBOLS(X) \
  X(EVP_PKEY_Q_keygen) X(EVP_PKEY_free) X(EVP_sha256) X(X509_new) X(X509_free) X(X509_get_serialNumber) \
  X(ASN1_INTEGER_set) X(X509_getm_notBefore) X(X509_getm_notAfter) X(X509_gmtime_adj) X(X509_set_pubkey) \
  X(X509_sign) X(TLS_server_method) X(SSL_CTX_new) X(SSL_CTX_free) X(SSL_CTX_use_certificate) \
  X(SSL_CTX_use_PrivateKey) X(SSL_new) X(SSL_free) X(SSL_set_fd) X(SSL_accept) X(SSL_read)
#define CRYPT_TEST_LOAD(name) \
  auto name = (decltype(&::name))dlsym(libssl_, #name); \
  ASSERT_TRUE(name != nullptr) << #name;
  CRYPT_TEST_SYMBOLS(CRYPT_TEST_LOAD)
#undef CRYPT_TEST_LOAD
#undef CRYPT_TEST_SYMBOLS
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

  ncclResult_t res = ncclInProgress;
  for (int i = 0; i < kMaxSteps && res == ncclInProgress; i++) {
    res = ncclCryptConnectHello(&client_);
    SSL_accept(impostor);
  }
  char hello[NCCL_SOCKET_PLAIN_HELLO_BYTES];
  int leaked = SSL_read(impostor, hello, sizeof(hello));
  SSL_free(impostor);
  SSL_CTX_free(ctx);
  X509_free(cert);
  EVP_PKEY_free(key);
  EXPECT_EQ(ncclRemoteError, res) << "the client completed a TLS handshake that did not use the PSK";
  EXPECT_LE(leaked, 0) << "a server without the PSK decrypted the client's socket hello";
#endif
}

}  // namespace
