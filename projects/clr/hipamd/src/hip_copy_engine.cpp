/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_ext_copy_engine.h>

#include <algorithm>
#include <array>

#include "device/rocm/rocabce.hpp"
#include "device/rocm/rocdevice.hpp"
#include "hip_internal.hpp"
#include "hip_platform.hpp"

namespace hip {
namespace {

amd::roc::Device* RocDevice(int device) {
  return static_cast<amd::roc::Device*>(g_devices[device]->devices()[0]);
}

bool IsDeviceOrdinal(int device) {
  return device >= 0 && static_cast<size_t>(device) < g_devices.size();
}

//! The copy endpoint for a HIP device ordinal, or host memory for hipCpuDeviceId.
bool EndpointFor(int device, abce_copy_endpoint_t& endpoint) {
  if (device == hipCpuDeviceId) {
    endpoint = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_HOST, ABCE_ANY_DEVICE);
    return true;
  }
  if (!IsDeviceOrdinal(device)) {
    return false;
  }
  endpoint = abce_copy_endpoint_make(ABCE_ENDPOINT_KIND_DEVICE,
                                     RocDevice(device)->info().driverNodeId_);
  return true;
}

}  // namespace

hipError_t hipExtCopyEngineGetDeviceHandles(int device, int srcDevice, int dstDevice,
                                            uint64_t engineMask, int maxHandles,
                                            hipExtCopyEngineDeviceHandle* handles,
                                            int* numHandles) {
  HIP_INIT_API(hipExtCopyEngineGetDeviceHandles, device, srcDevice, dstDevice, engineMask,
               maxHandles, handles, numHandles);
  if (handles == nullptr || numHandles == nullptr || maxHandles < 1 || engineMask == 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  abce_copy_endpoint_t src{};
  abce_copy_endpoint_t dst{};
  if (!IsDeviceOrdinal(device) || !EndpointFor(srcDevice, src) || !EndpointFor(dstDevice, dst)) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  *numHandles = 0;
  amd::roc::AbceEngineSet* engines = RocDevice(device)->getOrCreateAbceEngineSet();
  if (engines == nullptr) {
    HIP_RETURN(hipErrorNotSupported);
  }
  std::array<abce_device_handle_t, amd::roc::AbceEngineSet::kMaxEngines> internal{};
  std::array<uint32_t, amd::roc::AbceEngineSet::kMaxEngines> engine_ids{};
  const uint32_t num_internal =
      engines->GetDeviceHandles(src, dst, engineMask, std::min<uint32_t>(maxHandles, internal.size()),
                                internal.data(), engine_ids.data());
  if (num_internal == 0) {
    HIP_RETURN(hipErrorNotSupported);
  }
  for (uint32_t handle_idx = 0; handle_idx < num_internal; ++handle_idx) {
    handles[handle_idx].stateAddress = reinterpret_cast<uint64_t>(internal[handle_idx].engine);
    handles[handle_idx].deviceAbiVersion = internal[handle_idx].abi_version;
    handles[handle_idx].engineId = engine_ids[handle_idx];
  }
  *numHandles = static_cast<int>(num_internal);
  HIP_RETURN(hipSuccess);
}

}  // namespace hip
