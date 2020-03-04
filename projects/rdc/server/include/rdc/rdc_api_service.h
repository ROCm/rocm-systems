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
#ifndef SERVER_INCLUDE_RDC_RDC_API_SERVICE_H_
#define SERVER_INCLUDE_RDC_RDC_API_SERVICE_H_

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc.h"

namespace amd {
namespace rdc {

class RdcAPIServiceImpl final : public ::rdc::RdcAPI::Service {
 public:
    RdcAPIServiceImpl();
    ~RdcAPIServiceImpl();

    rdc_status_t Initialize(uint64_t rdcd_init_flags = 0);

    ::grpc::Status GetAllDevices(::grpc::ServerContext* context,
                  const ::rdc::Empty* request,
                  ::rdc::GetAllDevicesResponse* reply) override;

    ::grpc::Status GetDeviceAttributes(::grpc::ServerContext* context,
                  const ::rdc::GetDeviceAttributesRequest* request,
                  ::rdc::GetDeviceAttributesResponse* reply) override;

 private:
    rdc_handle_t rdc_handle_;
};

}  // namespace rdc
}  // namespace amd

#endif  // SERVER_INCLUDE_RDC_RDC_API_SERVICE_H_
