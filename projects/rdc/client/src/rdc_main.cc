
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
#include "rdc/rdc_main.h"
#include "rdc/rdc_client.h"

namespace amd {
namespace rdc {

RDCChannel::RDCChannel(std::string server_ip, std::string server_port,
            bool secure) : server_ip_(server_ip), server_port_(server_port),
                                                    secure_channel_(secure) {}

RDCChannel::~RDCChannel() {
}

rdc_status_t
RDCChannel::Initialize(void) {
  assert(!server_port_.empty());
  assert(!server_ip_.empty());

  std::string addr_str = server_ip() + ":";
  addr_str += server_port();

  std::shared_ptr<grpc::Channel> channel;

  if (secure_channel_) {
    // Not yet supported
    return RDC_STATUS_GRPC_UNIMPLEMENTED;
  } else {
    channel = ::grpc::CreateChannel(addr_str,
                                          grpc::InsecureChannelCredentials());
  }

  stub_ = ::rdc::Rsmi::NewStub(channel);

  if (stub_ == nullptr) {
    return RDC_STATUS_GRPC_RESOURCE_EXHAUSTED;
  }
  return RDC_STATUS_SUCCESS;
}

}  // namespace rdc
}  // namespace amd
