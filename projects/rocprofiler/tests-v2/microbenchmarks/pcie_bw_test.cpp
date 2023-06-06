/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include <stdint.h>
#include <stddef.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <chrono>
#include <thread>
#include <future>

#include "rocm_smi/rocm_smi.h"


/**
 * @brief Test SMI APIs for PCIe
 *
 * rsmi_status_t 	rsmi_dev_pci_bandwidth_get (uint32_t dev, rsmi_pcie_bandwidth_t* bw)
        Retrieve PCIe link speeds for given device

 * rsmi_status_t 	rsmi_dev_pci_throughput_get (uint32_t dev, u64* sent, u64* received, u64*
 max_packet_size) Retrieve number of packets sent via PCIe to/from device, and the max packet size
 in bytes.
*/


#define DISPLAY_RSMI_ERR(RET)                                                                      \
  {                                                                                                \
    if (RET != RSMI_STATUS_SUCCESS) {                                                              \
      const char* err_str;                                                                         \
      std::cout << "\t===> ERROR: RSMI call returned " << (RET) << std::endl;                      \
      rsmi_status_string((RET), &err_str);                                                         \
      std::cout << "\t===> (" << err_str << ")" << std::endl;                                      \
      std::cout << "\t===> at " << __FILE__ << ":" << std::dec << __LINE__ << std::endl;           \
    }                                                                                              \
  }

#define CHK_ERR_ASRT(RET)                                                                          \
  {                                                                                                \
    if (((RET) != RSMI_STATUS_SUCCESS)) {                                                          \
      std::cout << std::endl << "\t===> TEST FAILURE." << std::endl;                               \
      DISPLAY_RSMI_ERR(RET);                                                                       \
      std::cout << "\t===> Abort is over-ridden due to dont_fail command line option."             \
                << std::endl;                                                                      \
    }                                                                                              \
  }

#define HANDLE_ERROR CHK_ERR_ASRT(ret);
#define HIP_ASSERT(x) (assert((x) == hipSuccess))

#define SEND_DATA()                                                                                \
  HIP_ASSERT(hipMemcpyAsync(dst, src, SIZE * sizeof(int), hipMemcpyDefault, stream));

static float burn_hip(int dev, int* dst, int* src, size_t SIZE,
                      std::atomic<bool>* transfer_started) {
  hipSetDevice(dev);
  hipStream_t stream;
  hipStreamCreate(&stream);
  hipEvent_t events[3];

  for (int i = 0; i < 3; i++) {
    hipEventCreate(events + i);
    SEND_DATA();
    hipEventRecord(events[i], stream);
  }
  SEND_DATA();
  hipEventSynchronize(events[0]);
  transfer_started->store(true);

  float elapsed = 0;
  uint64_t counter = 0;
  while (elapsed < 1500.0f) {  // Transfer data for 1.5 seconds = 1500 ms
    float out;

    hipEventSynchronize(events[(counter + 1) % 3]);
    hipEventElapsedTime(&out, events[counter % 3], events[(counter + 1) % 3]);
    elapsed += out;

    hipEventRecord(events[counter % 3], stream);
    SEND_DATA();
    counter += 1;
  }
  hipStreamSynchronize(stream);

  for (int i = 0; i < 3; i++) hipEventDestroy(events[i]);
  hipStreamDestroy(stream);

  return float(SIZE * sizeof(int) * counter) / elapsed / 1E6;
}

int main() {
  const size_t SIZE = 3 << 28;
  rsmi_status_t ret;
  uint16_t dev_id;

  int* h_ptr = new int[SIZE];
  hipHostRegister(h_ptr, SIZE * sizeof(int), 0);
  for (size_t i = 0; i < SIZE; i++) h_ptr[i] = i;

  ret = rsmi_init(0);
  HANDLE_ERROR;
  uint32_t num_devices;
  ret = rsmi_num_monitor_devices(&num_devices);
  HANDLE_ERROR;
  std::cout << "Num devices: " << num_devices << std::endl;

  for (uint32_t dev = 0; dev < num_devices; dev++) {
    hipSetDevice(dev);
    int* d_ptr;
    HIP_ASSERT(hipMalloc((void**)&d_ptr, SIZE * sizeof(int)));

    std::cout << ">>> Device " << dev << std::endl;
    ret = rsmi_dev_id_get(dev, &dev_id);
    HANDLE_ERROR;

    rsmi_pcie_bandwidth_t bandwidth;
    ret = rsmi_dev_pci_bandwidth_get(dev, &bandwidth);
    HANDLE_ERROR;

    for (uint32_t i = 0; i < bandwidth.transfer_rate.num_supported; i++) {
      std::cout << "State " << i << ": " << bandwidth.transfer_rate.frequency[i] << " at "
                << bandwidth.lanes[i] << " lanes.\n";
    }
    std::cout << "Current: " << bandwidth.transfer_rate.frequency[bandwidth.transfer_rate.current]
              << '\n';

    uint64_t sent = 0, received = 0, max_pkt_sz = 0;
    std::atomic<bool> transfer_started;
    transfer_started.store(false);
    auto thread =
        std::async(std::launch::async, burn_hip, dev, d_ptr, h_ptr, SIZE, &transfer_started);

    while (!transfer_started.load()) usleep(1);

    ret = rsmi_dev_pci_throughput_get(dev, &sent, &received, &max_pkt_sz);
    HANDLE_ERROR;

    std::cout << "Data sent: " << sent << std::endl;
    std::cout << "Data received: " << received << std::endl;
    std::cout << "Max packet size: " << max_pkt_sz << std::endl;

    std::cout << "HtD BW: " << 0.1 * int(10 * thread.get() + 0.5f) << " GB/s" << std::endl;

    transfer_started.store(false);
    thread = std::async(std::launch::async, burn_hip, dev, h_ptr, d_ptr, SIZE, &transfer_started);

    while (!transfer_started.load()) usleep(1);

    ret = rsmi_dev_pci_throughput_get(dev, &sent, &received, &max_pkt_sz);
    HANDLE_ERROR;

    std::cout << "Data sent: " << sent << std::endl;
    std::cout << "Data received: " << received << std::endl;
    std::cout << "Max packet size: " << max_pkt_sz << std::endl;

    std::cout << "DtH BW: " << 0.1 * int(10 * thread.get() + 0.5f) << " GB/s" << std::endl;
    HIP_ASSERT(hipFree(d_ptr));
  }

  hipHostUnregister(h_ptr);
  delete[] h_ptr;
  ret = rsmi_shut_down();
  return 0;
}