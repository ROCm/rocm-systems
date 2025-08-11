/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#include "RdciSubSystem.h"

#include <sstream>

#include "common/rdc_fields_supported.h"
#include "common/rdc_utils.h"
#include "rdc_lib/RdcException.h"

namespace amd {
namespace rdc {

RdciSubSystem::RdciSubSystem()
    : rdc_handle_(nullptr),
      ip_port_("localhost:50051")  // default host
      ,
      use_auth_(true),
      root_ca_("/etc/rdc/client/certs/rdc_cacert.pem"),
      client_cert_("/etc/rdc/client/certs/rdc_client_cert.pem"),
      client_key_("/etc/rdc/client/private/rdc_client_cert.key"),
      is_json_output_(false) {
  rdc_status_t status = rdc_init(0);
  if (status != RDC_ST_OK) {
    throw RdcException(status, "RDC initialize fail");
  }
}

bool RdciSubSystem::is_json_output() const { return is_json_output_; }

std::vector<std::string> RdciSubSystem::split_string(const std::string& s, char delimiter) const {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

void RdciSubSystem::connect() {
  rdc_status_t status;

  if (use_auth_) {
    std::string ca_pem;
    std::string client_cert_pem;
    std::string client_key_pem;

    if (!FileExists(root_ca_.c_str())) {
      std::cout << "In order to use the SSL mutual authentication, the "
                << "root CA must be copied to " << root_ca_ << std::endl;
      throw RdcException(RDC_ST_BAD_PARAMETER, "root CA not found");
    }
    int ret = ReadFile(root_ca_, &ca_pem);
    if (ret) {
      throw RdcException(RDC_ST_BAD_PARAMETER, std::string("Fail to read root CA at") + root_ca_);
    }
    if (!FileExists(client_cert_.c_str())) {
      std::cout << "In order to use the SSL mutual authentication, the "
                << "client certificate must be copied to " << client_cert_ << std::endl;
      throw RdcException(RDC_ST_BAD_PARAMETER, "client cert not found");
    }
    ret = ReadFile(client_cert_, &client_cert_pem);
    if (ret) {
      throw RdcException(RDC_ST_BAD_PARAMETER,
                         std::string("Fail to read client certificate at") + client_cert_);
    }
    if (!FileExists(client_key_.c_str())) {
      std::cout << "In order to use the SSL mutual authentication, the "
                << "client private key must be copied to " << client_key_ << std::endl;
      throw RdcException(RDC_ST_BAD_PARAMETER, "client key not found");
    }
    ret = ReadFile(client_key_, &client_key_pem);
    if (ret) {
      throw RdcException(RDC_ST_BAD_PARAMETER,
                         std::string("Fail to read client key at ") + client_key_);
    }

    status = rdc_connect(ip_port_.c_str(), &rdc_handle_, ca_pem.c_str(), client_cert_pem.c_str(),
                         client_key_pem.c_str());
  } else {  // Not use the SSL mutual authentication
    status = rdc_connect(ip_port_.c_str(), &rdc_handle_, nullptr, nullptr, nullptr);
  }

  if (status != RDC_ST_OK) {
    throw RdcException(status,
                       "Fail to setup the connection. Please check all "
                       "libraries in right folder");
  }
}

void RdciSubSystem::show_common_usage() const {
  std::cout << "  --host       <IP/FQDN>:port    Connects to "
            << "specified IP or fully-qualified domain name.\n";
  std::cout << "                                 The port "
            << "must be specified.\n";
  std::cout << "                                 Default: localhost:50051\n";
  std::cout << "  -u  --unauth                   Do not use the SSL mutual"
            << " authentication to encrypt the communication\n"
            << "                                 Default: SSL mutual will be"
            << " used. You must copy the root CA to " << root_ca_ << "\n"
            << "                                 Client certificate to " << client_cert_ << "\n"
            << "                                 Client key to " << client_key_ << "\n";
  std::cout << "  -h  --help                     Displays usage "
            << "information and exits.\n";
}

void RdciSubSystem::set_json_output(bool is_json) {
  is_json_output_ = is_json;
  std::cout << "{";
}

RdciSubSystem::~RdciSubSystem() {
  if (rdc_handle_) {
    rdc_disconnect(rdc_handle_);
    rdc_handle_ = nullptr;
  }

  rdc_shutdown();

  if (is_json_output_) {
    std::cout << "}" << std::endl;
  }
}

}  // namespace rdc
}  // namespace amd
