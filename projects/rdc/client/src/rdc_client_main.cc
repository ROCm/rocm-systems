
/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <assert.h>
#include <grpcpp/grpcpp.h>

#include <string>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc_client_main.h"
#include "rdc/rdc_client.h"
#include "common/rdc_utils.h"

namespace amd {
namespace rdc {

#ifdef USE_PINNED_CERTS
// Pinned certificates
static const char *kDefaultRDCServerCertPinPath =
                                             "/etc/rdc/server/rdc_server.crt";
static const char *kDefaultRDCClientKeyPinPath =
                                     "/etc/rdc/client/private/rdc_client.key";
static const char *kDefaultRDCClientCertPinPath =
                                             "/etc/rdc/client/rdc_client.crt";
#endif  // USE_PINNED_CERTS

// PKI certificates
static const char * kDefaultRDCClientCertKeyPkiPath =
                                "/etc/rdc/client/private/rdc_client_cert.key";
static const char * kDefaultRDCClientCertPemPkiPath =
                                  "/etc/rdc/client/certs/rdc_client_cert.pem";
static const char * kDefaultRDCClientCACertPemPkiPath =
                                       "/etc/rdc/client/certs/rdc_cacert.pem";

RDCChannel::RDCChannel(std::string server_ip, std::string server_port,
            bool secure) : server_ip_(server_ip), server_port_(server_port),
                                                    secure_channel_(secure) {}

RDCChannel::~RDCChannel() {
}

#ifdef USE_PINNED_CERTS
static int ConstructSSLOptsPin(grpc::SslCredentialsOptions *ssl_opts) {
  assert(ssl_opts != nullptr);
  if (ssl_opts == nullptr) {
    return -EINVAL;
  }

  // Ensure the required paths exists before going forward
  // TODO(cfreehil): override these defaults with values read from config
  // file
  if (!amd::rdc::FileExists(kDefaultRDCClientKeyPinPath) ||
      !amd::rdc::FileExists(kDefaultRDCServerCertPinPath) ||
      !amd::rdc::FileExists(kDefaultRDCClientCertPinPath)) {
    return -ENOENT;
  }

  std::string cli_key;
  std::string ser_crt;
  std::string cli_crt;
  int ret;
  ret = amd::rdc::ReadFile(kDefaultRDCClientKeyPinPath, &cli_key);
  if (ret) {
    return ret;
  }
  ret = amd::rdc::ReadFile(kDefaultRDCServerCertPinPath, &ser_crt);
  if (ret) {
    return ret;
  }
  ret = amd::rdc::ReadFile(kDefaultRDCClientCertPinPath, &cli_crt);
  if (ret) {
    return ret;
  }

  ssl_opts->pem_root_certs = ser_crt;
  ssl_opts->pem_private_key = cli_key;
  ssl_opts->pem_cert_chain = cli_crt;

  return 0;
}
#endif  // USE_PINNED_CERTS

static int ConstructSSLOptsPKI(grpc::SslCredentialsOptions *ssl_opts) {
  assert(ssl_opts != nullptr);
  if (ssl_opts == nullptr) {
    return -EINVAL;
  }

  // Ensure the required paths exists before going forward
  // TODO(cfreehil): override these defaults with values read from config
  // file
  if (!amd::rdc::FileExists(kDefaultRDCClientCertKeyPkiPath) ||
      !amd::rdc::FileExists(kDefaultRDCClientCertPemPkiPath) ||
      !amd::rdc::FileExists(kDefaultRDCClientCACertPemPkiPath)) {
    return -ENOENT;
  }

  std::string pem_root_certs;
  std::string pem_private_key;
  std::string pem_cert_chain;
  int ret;
  ret = amd::rdc::ReadFile(kDefaultRDCClientCACertPemPkiPath, &pem_root_certs);
  if (ret) {
    return ret;
  }
  ret = amd::rdc::ReadFile(kDefaultRDCClientCertKeyPkiPath, &pem_private_key);
  if (ret) {
    return ret;
  }
  ret = amd::rdc::ReadFile(kDefaultRDCClientCertPemPkiPath, &pem_cert_chain);
  if (ret) {
    return ret;
  }

  ssl_opts->pem_root_certs = pem_root_certs;
  ssl_opts->pem_private_key = pem_private_key;
  ssl_opts->pem_cert_chain = pem_cert_chain;

  return 0;
}

rdc_status_t
RDCChannel::Initialize(void) {
  assert(!server_port_.empty());
  assert(!server_ip_.empty());

  int ret;
  std::string addr_str = server_ip() + ":";
  addr_str += server_port();

  if (secure_channel_) {
    grpc::SslCredentialsOptions ssl_opts;

#ifdef USE_PINNED_CERTS
    ret = ConstructSSLOptsPin(&ssl_opts);
#else
    ret = ConstructSSLOptsPKI(&ssl_opts);
#endif
    if (ret) {
      std::cerr << "Failed to process OpenSSL keys and certificates." <<
                                                                    std::endl;
      return RDC_STATUS_CLIENT_ERR_SSL;
    }

    channel_creds_ = grpc::SslCredentials(ssl_opts);
    channel_ = grpc::CreateChannel(addr_str, channel_creds_);
  } else {
    channel_ = ::grpc::CreateChannel(addr_str,
                                          grpc::InsecureChannelCredentials());
  }

  rsmi_stub_ = ::rdc::Rsmi::NewStub(channel_);
  if (rsmi_stub_ == nullptr) {
    return RDC_STATUS_GRPC_RESOURCE_EXHAUSTED;
  }

  rdc_admin_stub_ = ::rdc::RdcAdmin::NewStub(channel_);
  if (rdc_admin_stub_ == nullptr) {
    return RDC_STATUS_GRPC_RESOURCE_EXHAUSTED;
  }

  // Test to see if we can connect to server; if not, return err.
  return RDC_STATUS_SUCCESS;
}

}  // namespace rdc
}  // namespace amd
