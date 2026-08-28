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

#include <cstdio>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "log.hpp"

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
    LOG_WARN("Pod detection: hipGetDevice failed (%s)", hipGetErrorString(err));
    return podIds;
  }

  // Get the BDF ID (PCI Bus ID) of the current device
  char bdfId[64];
  err = hipDeviceGetPCIBusId(bdfId, sizeof(bdfId), device);
  if (err != hipSuccess) {
    LOG_WARN("Pod detection: hipDeviceGetPCIBusId failed for device %d (%s)", device,
             hipGetErrorString(err));
    return podIds;
  }

  // Load AMD SMI library dynamically
  AmdsmiLoader amdsmi;
  if (!amdsmi.isLoaded()) {
    LOG_WARN("Pod detection: libamd_smi.so could not be loaded, or it does not export "
             "amdsmi_get_gpu_fabric_info (requires amd-smi 26.4 or newer)");
    return podIds;
  }

  // Initialize AMD SMI library
  amdsmi_status_t status = amdsmi.init(AMDSMI_INIT_AMD_GPUS);
  if (status != AMDSMI_STATUS_SUCCESS) {
    LOG_WARN("Pod detection: amdsmi_init failed with status %d", static_cast<int>(status));
    return podIds;
  }

  /*
   * Resolve the processor handle from the BDF. amdsmi_get_processor_handle_from_bdf()
   * takes a packed amdsmi_bdf_t by value, so the "domain:bus:device.function" string
   * that HIP hands back has to be parsed into the bitfields first.
   */
  unsigned int domain{0}, bus{0}, dev{0}, func{0};
  if (sscanf(bdfId, "%x:%x:%x.%x", &domain, &bus, &dev, &func) != 4) {
    LOG_WARN("Pod detection: could not parse PCI bus id '%s'", bdfId);
    amdsmi.shut_down();
    return podIds;
  }

  amdsmi_bdf_t bdf{};
  bdf.domain_number = domain;
  bdf.bus_number = bus;
  bdf.device_number = dev;
  bdf.function_number = func;

  amdsmi_processor_handle gpuHandle;
  status = amdsmi.get_processor_handle_from_bdf(bdf, &gpuHandle);
  if (status != AMDSMI_STATUS_SUCCESS) {
    LOG_WARN("Pod detection: amdsmi_get_processor_handle_from_bdf(%s) failed with status %d",
             bdfId, static_cast<int>(status));
    amdsmi.shut_down();
    return podIds;
  }

  // Get fabric information for the GPU
  amdsmi_fabric_info_t fabricInfo{};
  status = amdsmi.get_gpu_fabric_info(gpuHandle, &fabricInfo);
  if (status != AMDSMI_STATUS_SUCCESS) {
    LOG_WARN("Pod detection: amdsmi_get_gpu_fabric_info(%s) failed with status %d", bdfId,
             static_cast<int>(status));
    amdsmi.shut_down();
    return podIds;
  }

  // Extract pod IDs from fabric info
  memcpy(podIds.physicalPodId, fabricInfo.info.v1.ppod_id, 16);
  podIds.virtualPodId = fabricInfo.info.v1.vpod_id;

  if (IS_PODIDS_ZERO(podIds)) {
    LOG_WARN("Pod detection: %s reports an all-zero pod id (vpod_id=%u, accel_state=%u); "
             "the driver is not exposing fabric pod membership on this device",
             bdfId, fabricInfo.info.v1.vpod_id,
             static_cast<unsigned>(fabricInfo.info.v1.accel_state));
  }

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
