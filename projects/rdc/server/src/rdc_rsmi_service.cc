
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

#include <iostream>
#include <memory>
#include <string>
#include <csignal>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rocm_smi/rocm_smi.h"
#include "rdc/rdc_rsmi_service.h"

RsmiServiceImpl::RsmiServiceImpl():rsmi_initialized_(false) {
}

RsmiServiceImpl::~RsmiServiceImpl() {
  if (rsmi_initialized_) {
    rsmi_status_t rsmi_ret = rsmi_shut_down();
    rsmi_initialized_ = false;
    assert(rsmi_ret == RSMI_STATUS_SUCCESS);
  }
}

// rsmi and rdc currently happen to have a 1-to-1 mapping, but
// have this function in case that changes
static rsmi_temperature_metric_t
    rdc_temp2rsmi_temp(::rdc::GetTemperatureRequest_TemperatureMetric
                                                                   rdc_temp) {
  return static_cast<rsmi_temperature_metric_t>(rdc_temp);
}

rsmi_status_t
RsmiServiceImpl::Initialize(uint64_t rsmi_init_flags) {
  rsmi_status_t rsmi_ret = rsmi_init(rsmi_init_flags);
  if (rsmi_ret != RSMI_STATUS_SUCCESS) {
    std::cout << "rsmi_init() returned error" << std::endl;
  } else {
    rsmi_initialized_ = true;
  }
  return rsmi_ret;
}

::grpc::Status
RsmiServiceImpl::GetNumDevices(::grpc::ServerContext* context,
                               const ::rdc::GetNumDevicesRequest* request,
                               ::rdc::GetNumDevicesResponse* reply) {
  assert(reply != nullptr);
  uint32_t num_devices;

  (void)context;  // Quiet warning for now;
  (void)request;

  rsmi_status_t ret = rsmi_num_monitor_devices(&num_devices);

  // TODO(cfreehil) replace below with macro
  if (ret != RSMI_STATUS_SUCCESS) {
    std::cout << "rsmi_num_monitor_devices() returned error" << std::endl;
  }
  reply->set_val(num_devices);
  reply->set_ret_val(ret);

  return ::grpc::Status::OK;
}

::grpc::Status
RsmiServiceImpl::GetTemperature(::grpc::ServerContext* context,
       const ::rdc::GetTemperatureRequest* request,
                         ::rdc::GetTemperatureResponse* response) {
  (void)context;  // Quiet warning for now;

  int64_t temperature;
  rsmi_status_t ret = rsmi_dev_temp_metric_get(request->dv_ind(),
      request->sensor_type(), rdc_temp2rsmi_temp(request->metric()),
                                                              &temperature);

  response->set_temperature(temperature);
  response->set_ret_val(ret);
  return ::grpc::Status::OK;
}

// TODO(cfreehil): read server config from YAML file. Config can include things
// like server address, Secure/Insecure creds, rsmi_init flags, etc.
void RunServer() {
  std::string server_address("0.0.0.0:50051");
  RsmiServiceImpl service;

  ::grpc::ServerBuilder builder;
  // Listen on the given address without any authentication mechanism.
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  // Register "service" as the instance through which we'll communicate with
  // clients. In this case it corresponds to an *synchronous* service.
  builder.RegisterService(&service);
  // Finally assemble the server.
  std::unique_ptr<::grpc::Server> server(builder.BuildAndStart());
  std::cout << "Server listening on " << server_address << std::endl;

  uint64_t flags = 0;  // TODO(cfreehil) Read this from config file
  rsmi_status_t rsmi_ret = rsmi_init(flags);
  // TODO(cfreehil): check rsmi return code
  // Wait for the server to shutdown. Note that some other thread must be
  // responsible for shutting down the server for this call to ever return.
  if (rsmi_ret != RSMI_STATUS_SUCCESS) {
    std::cout << "rsmi_init() returned error. Exiting" << std::endl;
    return;
  }
  server->Wait();
}
