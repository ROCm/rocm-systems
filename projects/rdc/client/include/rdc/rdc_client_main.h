
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

#ifndef CLIENT_INCLUDE_RDC_RDC_CLIENT_MAIN_H_
#define CLIENT_INCLUDE_RDC_RDC_CLIENT_MAIN_H_

#include <grpcpp/grpcpp.h>

#include <string>
#include <memory>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc_client.h"

namespace amd {
namespace rdc {

class RDCChannel {
 public:
  explicit RDCChannel(std::string server_ip, std::string server_port,
                                                         bool secure_channel);
  ~RDCChannel();

  rdc_status_t Initialize(void);

  // Getters and Setters

  // Don't have setter for server ip and ports; we don't want to change those
  // after construction
  std::string server_ip(void) const {return server_ip_;}
  std::string server_port(void) const {return server_port_;}
  bool secure_channel(void) const {return secure_channel_;}
  std::shared_ptr<::rdc::Rsmi::Stub> rsmi_stub(void) const {return rsmi_stub_;}
  std::shared_ptr<::rdc::RdcAdmin::Stub> rdc_admin_stub(void) const {
                                                      return rdc_admin_stub_;}
  std::shared_ptr<grpc::Channel> const channel(void) {return channel_;}

 private:
  std::string server_ip_;
  std::string server_port_;
  bool secure_channel_;
  std::shared_ptr<::rdc::Rsmi::Stub> rsmi_stub_;
  std::shared_ptr<::rdc::RdcAdmin::Stub> rdc_admin_stub_;
  std::shared_ptr<grpc::Channel> channel_;
  std::shared_ptr<grpc::ChannelCredentials> channel_creds_;
};

}  // namespace rdc
}  // namespace amd

#endif  // CLIENT_INCLUDE_RDC_RDC_CLIENT_MAIN_H_
