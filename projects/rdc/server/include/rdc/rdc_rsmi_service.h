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
#ifndef SERVER_INCLUDE_RDC_RDC_RSMI_SERVICE_H_
#define SERVER_INCLUDE_RDC_RDC_RSMI_SERVICE_H_

#include "rdc.grpc.pb.h"  // NOLINT
#include "rocm_smi/rocm_smi.h"
#include "rdc/rdc_rsmi_service.h"

class RsmiServiceImpl final : public ::rdc::Rsmi::Service {
 public:
    RsmiServiceImpl();
    ~RsmiServiceImpl();

    rsmi_status_t Initialize(uint64_t rsmi_init_flags = 0);

    ::grpc::Status VerifyConnection(::grpc::ServerContext* context,
                                const rdc::VerifyConnectionRequest* request,
                              rdc::VerifyConnectionResponse* reply) override;

    ::grpc::Status
    GetNumDevices(::grpc::ServerContext* context,
                                   const ::rdc::GetNumDevicesRequest* request,
                                 ::rdc::GetNumDevicesResponse* reply) override;

    ::grpc::Status
    GetTemperature(::grpc::ServerContext* context,
           const ::rdc::GetTemperatureRequest* request,
                             ::rdc::GetTemperatureResponse* response) override;

 private:
    bool rsmi_initialized_;
};

#endif  // SERVER_INCLUDE_RDC_RDC_RSMI_SERVICE_H_
