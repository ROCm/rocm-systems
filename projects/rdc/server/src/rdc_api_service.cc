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
#include <assert.h>
#include <grpcpp/grpcpp.h>

#include <iostream>
#include <memory>
#include <string>
#include <csignal>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc_api_service.h"
#include "rdc/rdc.h"

namespace amd {
namespace rdc {

RdcAPIServiceImpl::RdcAPIServiceImpl():rdc_handle_(nullptr) {
}

rdc_status_t RdcAPIServiceImpl::Initialize(uint64_t rdcd_init_flags) {
    rdc_status_t result = rdc_init(rdcd_init_flags);
    if (result != RDC_ST_OK) {
        std::cout << "RDC API initialize fail\n";
        return result;
    }

    result = rdc_start_embedded(RDC_OPERATION_MODE_AUTO, &rdc_handle_);
    if (result != RDC_ST_OK) {
        std::cout << "RDC API start embedded fail\n";
    }

    return result;
}

RdcAPIServiceImpl::~RdcAPIServiceImpl() {
  if (rdc_handle_) {
    rdc_stop_embedded(rdc_handle_);
    rdc_handle_ = nullptr;
  }
  rdc_shutdown();
}

::grpc::Status RdcAPIServiceImpl::GetAllDevices(::grpc::ServerContext* context,
                  const ::rdc::Empty* request,
                  ::rdc::GetAllDevicesResponse* reply) {
    (void)(context);
    (void)(request);
    if (!reply) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty reply");
    }
    uint32_t  gpu_index_list[RDC_MAX_NUM_DEVICES];
    uint32_t count = 0;
    rdc_status_t result =  rdc_get_all_devices(rdc_handle_,
                  gpu_index_list, &count);
    reply->set_status(result);
    if (result != RDC_ST_OK) {
        return ::grpc::Status::OK;
    }
    for (uint32_t i = 0; i < count; i++) {
         reply->add_gpus(gpu_index_list[i]);
    }

    return ::grpc::Status::OK;
}

::grpc::Status RdcAPIServiceImpl::GetDeviceAttributes(
      ::grpc::ServerContext* context,
      const ::rdc::GetDeviceAttributesRequest* request,
      ::rdc::GetDeviceAttributesResponse* reply) {
    (void)(context);
    if (!reply || !request) {
      return ::grpc::Status(::grpc::StatusCode::INTERNAL, "Empty contents");
    }
    uint32_t gpu_index = request->gpu_index();
    rdc_device_attributes_t attribute;
    rdc_status_t result = rdc_get_device_attributes(rdc_handle_,
            gpu_index, &attribute);

    ::rdc::DeviceAttributes* attr = reply->mutable_attributes();
    attr->set_device_name(attribute.device_name);

    reply->set_status(result);

    return ::grpc::Status::OK;
}

}  // namespace rdc
}  // namespace amd


