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

#include <chrono>
#include <thread>
#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>
#include "hipSVMCommon.h"
#define MAX_TARGETS 1024

__global__ void find_targets(unsigned int* image, unsigned int target,
                             unsigned int* numTargetsFound, unsigned int* targetLocations) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int index = 0;
  if (image[i] == target) {
    index = atomicAdd((unsigned int*)numTargetsFound, 1u);
    if (index < MAX_TARGETS) {
      atomicExch_system((unsigned int*)&targetLocations[index], (unsigned int)i);
    }
  }
}

void spawnAnalysisTask(int location) { printf("found target at location %d\n", location); }

/**
* Test Description
* ------------------------
* - The suite will test the following functions,
      hipHostMalloc() with following flags,
        hipHostMallocCoherent(CL_MEM_SVM_FINE_GRAIN_BUFFER + CL_MEM_SVM_ATOMICS)
        hipHostMallocNonCoherent(CL_MEM_SVM_FINE_GRAIN_BUFFER)
      atomicAdd()(in kernel)
      atomicExch_system()(in kernel)
      InterlockedExchangeAdd()(in WINDOWS host)
      __sync_add_and_fetch()(in LINUX host)
      hipStreamCreate()
      hipEventCreate()
      hipEventRecord()
      hipEventQuery()
*   It will demonstrate use of SVM's atomics to do fine grain synchronization between
*   a device and the host. The result will be verified on the host.
*   Concept: a device kernel is used to search an input image for regions that match a
*   target pattern. The device immediately notifies the host when it finds a target
*   (via an atomic operation that works across host and devices). The host is then able
*   to spawn a task that further analyzes the target while the device continues searching
*   for more targets.
* Test source
* ------------------------
* - catch/unit/memory/hipSVMTestFineGrainSyncBuffers.cpp
* Test requirements
* ------------------------
*  - Host specific (WINDOWS and LINUX)
*  - Fine grain access and atomics supported on device and host
*  - HIP_VERSION >= 5.7
*/
TEST_CASE("test_svm_fine_grain_sync_buffers") {
  int pcieAtomic = 0;
  HIP_CHECK(hipDeviceGetAttribute(&pcieAtomic, hipDeviceAttributeHostNativeAtomicSupported, 0));
  if (!pcieAtomic) {
    fprintf(stderr, "Device doesn't support pcie atomic, Skipped\n");
    REQUIRE(true);
    return;
  }
  size_t num_pixels = 1024 * 1024 * 2;
  hipStream_t stream;
  HIP_CHECK(hipSetDevice(0));
  HIP_CHECK(hipStreamCreate(&stream));
  hipEvent_t event;
  HIP_CHECK(hipEventCreate(&event));
  unsigned int *pInputImage, *pNumTargetsFound, *pTargetLocations;
  HIP_CHECK(
      hipHostMalloc(&pInputImage, sizeof(unsigned int) * num_pixels, hipHostMallocNonCoherent));
  HIP_CHECK(hipHostMalloc(&pNumTargetsFound, sizeof(unsigned int), hipHostMallocCoherent));
  HIP_CHECK(hipHostMalloc(&pTargetLocations, sizeof(int) * MAX_TARGETS, hipHostMallocCoherent));
  unsigned int targetDescriptor = 777;
  *pNumTargetsFound = 0;

  unsigned int i;
  for (i = 0; i < MAX_TARGETS; i++) pTargetLocations[i] = -1;
  for (i = 0; i < num_pixels; i++) pInputImage[i] = 0;
  pInputImage[0] = targetDescriptor;
  pInputImage[3] = targetDescriptor;
  pInputImage[num_pixels - 1] = targetDescriptor;

  find_targets<<<(num_pixels + 255) / 256, 256, 0, stream>>>(pInputImage, targetDescriptor,
                                                             pNumTargetsFound, pTargetLocations);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipEventRecord(event, stream));

  i = 0;
  hipError_t status = hipSuccess;
  unsigned int loc = 0;
  // check for new targets, if found spawn a task to analyze target.
  do {
    status = hipEventQuery(event);
    if (status != hipErrorNotReady && status != hipSuccess) {
      fprintf(stderr, "Unexpected status = %d\n", status);
      REQUIRE(false);
    }
    loc = AtomicLoad32(&pTargetLocations[i]);
    if (loc != -1)  // -1 indicates slot not used yet.
    {
      spawnAnalysisTask(loc);  // Do something...
      i++;
    }
  } while (status == hipErrorNotReady || AtomicLoad32(&pTargetLocations[i]) != -1);

  HIP_CHECK(hipHostFree(pInputImage));
  HIP_CHECK(hipHostFree(pNumTargetsFound));
  HIP_CHECK(hipHostFree(pTargetLocations));
  HIP_CHECK(hipEventDestroy(event));
  HIP_CHECK(hipStreamDestroy(stream));
  REQUIRE(i == 3);
}
