//
// Copyright (c) 2017 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

/*
 * Modifications Copyright (C)2023 Advanced
 * Micro Devices, Inc. All rights reserved.
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

// Each device will write it's id into the bytes that it "owns", ownership is based on round robin
// (global_id % num_id) num_id is equal to number of SVM devices in the system plus one (for the
// host code). id is the index (id) of the device that this kernel is executing on. For example, if
// there are 2 SVM devices and the host; the buffer should look like this after each device and the
// host write their id's: 0, 1, 2, 0, 1, 2, 0, 1, 2...
__global__ void write_owned_locations(char* a, unsigned int num_id, unsigned int id) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  int owner = i % num_id;
  if (id == owner) a[i] = id;  // modify location if it belongs to this device, write id
}

// Verify that a device can see the byte sized updates from the other devices, sum up the device
// id's and see if they match expected value. Note: this must be called with a reduced NDRange so
// that neighbor acesses don't go past end of buffer. For example if there are two SVM devices and
// the host (3 total devices) the buffer should look like this: 0,1,2,0,1,2... and the expected sum
// at each point is 0+1+2 = 3.
__global__ void sum_neighbor_locations(char* a, unsigned int num_devices,
                                       unsigned int* error_count) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int expected_sum = (num_devices * (num_devices - 1)) / 2;
  unsigned int sum = 0;
  for (unsigned int j = 0; j < num_devices; j++) {
    sum += a[i + j];  // add my neighbors to the right
  }
  if (sum != expected_sum) atomicAdd_system(error_count, 1u);  // like opencl atomic_inc()
}

/**
* Test Description
* ------------------------
* - The suite will test the following functions,
      hipHostMalloc() with following flags,
        hipHostMallocCoherent(CL_MEM_SVM_FINE_GRAIN_BUFFER + CL_MEM_SVM_ATOMICS)
        hipHostMallocNonCoherent(CL_MEM_SVM_FINE_GRAIN_BUFFER)
      atomicAdd_system()(in kernel)
      hipStreamCreate()
      hipStreamSynchronize()
*   It will demonstrate use of SVM's atomics to do fine grain synchronization among
*   devices with each stream on each device. The result will be verified on the host.
* Test source
* ------------------------
* - catch/unit/memory/hipSVMTestByteGranularity.cpp
* Test requirements
* ------------------------
*  - Host specific (WINDOWS and LINUX)
*  - Fine grain access and atomics supported on devices
*  - HIP_VERSION >= 5.7
*/
TEST_CASE("test_svm_byte_granularity") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    fprintf(stderr, "Device doesn't support pcie atomic, Skipped\n");
    REQUIRE(true);
    return;
  }
  const int num_elements = 2048;
  int num_devices = 0;
  HIP_CHECK(hipGetDeviceCount(&num_devices));
  int num_devices_plus_host = num_devices + 1;
  std::vector<hipStream_t> streams(num_devices);

  for (int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipSetDevice(d));
    HIP_CHECK(hipStreamCreate(&streams[d]));
  }
  HIP_CHECK(hipSetDevice(0));
  char* pA = nullptr;
  // hipHostMallocNonCoherent means CL_MEM_SVM_FINE_GRAIN_BUFFER
  HIP_CHECK(hipHostMalloc(&pA, sizeof(char) * num_elements, hipHostMallocNonCoherent));
  unsigned int** error_counts = (unsigned int**)malloc(sizeof(void*) * num_devices);

  for (unsigned int i = 0; i < num_devices; i++) {
    // hipHostMallocCoherent means CL_MEM_SVM_FINE_GRAIN_BUFFER + CL_MEM_SVM_ATOMICS
    // We need atomic inc among different GPUs
    HIP_CHECK(hipHostMalloc(&error_counts[i], sizeof(unsigned int) * num_elements,
                            hipHostMallocCoherent));
    *error_counts[i] = 0;
  }
  for (int i = 0; i < num_elements; i++) pA[i] = -1;

  // get all the devices going simultaneously
  for (unsigned int d = 0; d < num_devices; d++)  // device ids starting at 1.
  {
    HIP_CHECK(hipSetDevice(d));
    write_owned_locations<<<num_elements, 1, 0, streams[d]>>>(pA, num_devices_plus_host, d);
    HIP_CHECK(hipGetLastError());
  }
  unsigned int host_id = num_devices;  // host code will take the id above the devices.
  for (unsigned int i = num_devices; i < num_elements; i += num_devices_plus_host) pA[i] = host_id;

  for (unsigned int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipStreamSynchronize(streams[d]));
  }

  // now check that each device can see the byte writes made by the other devices.
  // adjusted so sum_neighbor_locations doesn't read past end of buffer
  size_t adjusted_num_elements = num_elements - num_devices;
  for (unsigned int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipSetDevice(d));
    sum_neighbor_locations<<<adjusted_num_elements, 1, 0, streams[d]>>>(pA, num_devices_plus_host,
                                                                        error_counts[d]);
    HIP_CHECK(hipGetLastError());
  }

  for (unsigned int d = 0; d < num_devices; d++) {
    HIP_CHECK(hipStreamSynchronize(streams[d]));
  }
  // see if any of the devices found errors
  for (unsigned int d = 0; d < num_devices; d++) {
    if (*error_counts[d] > 0) {
      fprintf(stderr, "*error_counts[%u] = %u\n", d, *error_counts[d]);
      REQUIRE(false);
    }
  }
  unsigned int expected = (num_devices_plus_host * (num_devices_plus_host - 1)) / 2;
  // check that host can see the byte writes made by the devices.
  for (unsigned int i = 0; i < num_elements - num_devices_plus_host; i++) {
    unsigned int sum = 0;
    for (unsigned int j = 0; j < num_devices_plus_host; j++) sum += pA[i + j];
    if (sum != expected) {
      fprintf(stderr, "[%u]: sum %u != expected %u", i, sum, expected);
      REQUIRE(false);
    }
  }
  for (unsigned int i = 0; i < num_devices; i++) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
    HIP_CHECK(hipHostFree(error_counts[i]));
  }
  free(error_counts);
  HIP_CHECK(hipHostFree(pA));
  REQUIRE(true);
}
