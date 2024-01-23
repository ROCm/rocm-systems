
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

#include "rdc/rdc_client.h"

#include <grpcpp/grpcpp.h>
#include <unistd.h>

#include <iostream>

#include "amd_smi/amdsmi.h"

#define CHK_RET_STATUS(RET)                                                           \
  if ((RET) != RDC_STATUS_SUCCESS) {                                                  \
    const char* err_msg_str;                                                          \
    (void)rdc_status_string((RET), &err_msg_str);                                     \
    std::cout << "rdc call returned error: " << (RET) << ":\"" << err_msg_str << "\"" \
              << std::endl;                                                           \
  }

#define CHK_RET_STATUS_CONT(RET)                                    \
  if ((RET) != RDC_STATUS_SUCCESS) {                                \
    std::cout << "rdc call returned error: " << (RET) << std::endl; \
    continue;                                                       \
  }

int main(int argc, char** argv) {
  rdc_status_t ret;
  rdc_channel_t server_ch;
  uint64_t num_gpu;
  int64_t temperature;
  std::string serv_host("localhost");
  std::string serv_port("50051");

  if (argc > 1) {
    serv_host = argv[1];
  }
  if (argc > 2) {
    serv_port = argv[2];
  }

  std::cout << "Attempting to create channel to " << serv_host << ":" << serv_port << std::endl;

  ret = rdc_channel_create(&server_ch, serv_host.c_str(), serv_port.c_str(), false);
  CHK_RET_STATUS(ret)
  std::cout << "Successfully created channel" << std::endl;

  grpc_connectivity_state ch_state;
  ret = rdc_channel_state_get(server_ch, true, &ch_state);
  CHK_RET_STATUS(ret)
  std::cout << "Current channel state is " << ch_state << std::endl;

  std::cout << "Verifying connection to server..." << std::endl;
  ret = rdc_channel_connection_verify(server_ch);
  CHK_RET_STATUS(ret)
  if (ret == RDC_STATUS_SUCCESS) {
    std::cout << "Verified connection to server." << std::endl;
  }
  std::cout << "Getting number of gpus at server..." << std::endl;
  ret = rdc_num_gpus_get(server_ch, &num_gpu);
  CHK_RET_STATUS(ret)
  std::cout << "Number of GPUs at server is " << server_ch << num_gpu << std::endl;

  for (uint32_t dv_ind = 0; dv_ind < num_gpu; ++dv_ind) {
    std::cout << "Info for Device " << dv_ind << ":" << std::endl;
    std::cout << "\tGetting temperature..." << std::endl;
    ret = rdc_dev_temp_metric_get(server_ch, dv_ind, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_CURRENT,
                                  &temperature);
    CHK_RET_STATUS_CONT(ret)
    std::cout << "\t GPU " << dv_ind << " has a temperature of " << temperature << std::endl;
  }

  ret = rdc_channel_destroy(server_ch);
  CHK_RET_STATUS(ret)
  std::cout << "Successfully destroyed channel to " << serv_host << ":" << serv_port << std::endl;

  return 0;
}
