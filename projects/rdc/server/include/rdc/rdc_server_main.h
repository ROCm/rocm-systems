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
#ifndef SERVER_INCLUDE_RDC_RDC_SERVER_MAIN_H_
#define SERVER_INCLUDE_RDC_RDC_SERVER_MAIN_H_

#include <grpcpp/grpcpp.h>

#include <memory>
#include <string>

#include "rdc/rdc_admin_service.h"
#include "rdc/rdc_api_service.h"

#define RDC_SERVER_VERSION_MAJOR 1
#define RDC_SERVER_VERSION_MINOR 1
#define RDC_SERVER_VERSION_RELEASE 0

#define RDC_SERVER_VERSION_CREATE_STRING(MAJOR, MINOR, RELEASE) (#MAJOR "." #MINOR "." #RELEASE)
#define RDC_SERVER_VERSION_EXPAND_PARTS(MAJOR_STR, MINOR_STR, RELEASE_STR) RDC_SERVER_VERSION_CREATE_STRING(MAJOR_STR, MINOR_STR, RELEASE_STR)
#define RDC_SERVER_VERSION_STRING RDC_SERVER_VERSION_EXPAND_PARTS(RDC_SERVER_VERSION_MAJOR, RDC_SERVER_VERSION_MINOR, RDC_SERVER_VERSION_RELEASE)

#define Q(x) #x
#define QUOTE(x) Q(x)

typedef struct {
  std::string listen_address;
  std::string listen_port;
  bool no_authentication;
  bool use_pinned_certs;
  bool log_dbg;
} RdcdCmdLineOpts;

class RDCServer {
 public:
  RDCServer();
  ~RDCServer();

  void Initialize(RdcdCmdLineOpts* cl);

  void Run(void);
  void ShutDown(void);

  bool start_rdc_admin_service(void) const { return start_rdc_admin_service_; }
  void set_start_rdc_admin_service(bool s) { start_rdc_admin_service_ = s; }

  bool start_api_service(void) const { return start_api_service_; }
  void set_start_api_service(bool s) { start_api_service_ = s; }

  bool secure_creds(void) const { return secure_creds_; }
  void set_secure_creds(bool s) { secure_creds_ = s; }

 private:
  void HandleSignal(int sig);
  std::string server_address_;
  std::unique_ptr<::grpc::Server> server_;
  bool secure_creds_;
  bool use_pinned_certs_;
  bool log_debug_;
  RdcdCmdLineOpts* cmd_line_;

  bool start_rdc_admin_service_;
  amd::rdc::RDCAdminServiceImpl* rdc_admin_service_;

  bool start_api_service_;
  amd::rdc::RdcAPIServiceImpl* api_service_;
};

#endif  // SERVER_INCLUDE_RDC_RDC_SERVER_MAIN_H_
