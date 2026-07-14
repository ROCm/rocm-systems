/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "pod_detection.hpp"

#include <hip/hip_runtime.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)

#ifdef HAVE_AMDSMI_GPU_FABRIC_INFO
#include "amdsmi_loader.hpp"
#endif

namespace rocshmem {

PodIds detectLocalPodIds() {
#ifdef HAVE_AMDSMI_GPU_FABRIC_INFO
  PodIds podIds = {};  // Zero-initialize the structure

  // Get the current HIP device
  int device;
  hipError_t err = hipGetDevice(&device);
  if (err != hipSuccess) {
    return podIds;
  }

  // Get the BDF ID (PCI Bus ID) of the current device
  char bdfId[64];
  err = hipDeviceGetPCIBusId(bdfId, sizeof(bdfId), device);
  if (err != hipSuccess) {
    return podIds;
  }

  // Load AMD SMI library dynamically
  AmdsmiLoader amdsmi;
  if (!amdsmi.isLoaded()) {
    return podIds;
  }

  // Initialize AMD SMI library
  amdsmi_status_t status = amdsmi.init(AMDSMI_INIT_AMD_GPUS);
  if (status != AMDSMI_STATUS_SUCCESS) {
    return podIds;
  }

  // Get processor handle from BDF ID
  amdsmi_processor_handle gpuHandle;
  status = amdsmi.get_processor_handle_from_bdf(bdfId, &gpuHandle);
  if (status != AMDSMI_STATUS_SUCCESS) {
    amdsmi.shut_down();
    return podIds;
  }

  // Get fabric information for the GPU
  amdsmi_fabric_info_t fabricInfo;
  status = amdsmi.get_gpu_fabric_info(gpuHandle, &fabricInfo);
  if (status != AMDSMI_STATUS_SUCCESS) {
    amdsmi.shut_down();
    return podIds;
  }

  // Extract pod IDs from fabric info
  memcpy(podIds.physicalPodId, fabricInfo.info.v1.ppod_id, 16);
  podIds.virtualPodId = fabricInfo.info.v1.vpod_id;

  // Cleanup AMD SMI
  amdsmi.shut_down();

  return podIds;
#else
  // Stub implementation when fabric support is not available
  PodIds podIds = {};  // Zero-initialize the structure
  return podIds;
#endif
}

std::vector<int> matchIpcCapableRanks(int rank, const std::vector<PodIds>& allPodIds) {
  std::vector<int> ipcCapableRanks;

  if (rank < 0 || rank >= static_cast<int>(allPodIds.size())) {
    return ipcCapableRanks;
  }

  PodIds myPodIds = allPodIds[rank];

  // Find all ranks with matching pod IDs
  for (int i = 0; i < static_cast<int>(allPodIds.size()); i++) {
    if (memcmp(allPodIds[i].physicalPodId, myPodIds.physicalPodId, 16) == 0 &&
        allPodIds[i].virtualPodId == myPodIds.virtualPodId) {
      ipcCapableRanks.push_back(i);
    }
  }

  return ipcCapableRanks;
}

}  // namespace rocshmem
