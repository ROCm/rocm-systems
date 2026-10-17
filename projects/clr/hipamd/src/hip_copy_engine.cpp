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
hipError_t hipExtCopyEngineGetDeviceHandle(int device, hipExtCopyEngineDeviceHandle* handle) {
  HIP_INIT_API(hipExtCopyEngineGetDeviceHandle, device, handle);
  if (handle == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  if (device < 0 || static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  auto* roc_device = static_cast<amd::roc::Device*>(g_devices[device]->devices()[0]);
  amd::roc::AbceEngineSet* engines = roc_device->getOrCreateAbceEngineSet();
  if (engines == nullptr) {
    HIP_RETURN(hipErrorNotSupported);
  }

  // Publish the whole process's device set, not just this one: a kernel on this
  // device tags peer endpoints with peer agent ids, so it needs every ordinal it
  // could name. Ordinals beyond the map's reach stay unknown, which still submits
  // and simply places the copy by ring availability.
  const uint32_t num_devices =
      static_cast<uint32_t>(std::min(g_devices.size(), static_cast<size_t>(ABCE_DEVICE_MAX_PEERS)));
  std::array<uint32_t, ABCE_DEVICE_MAX_PEERS> driver_node_ids{};
  for (uint32_t peer = 0; peer < num_devices; ++peer) {
    auto* peer_device = static_cast<amd::roc::Device*>(g_devices[peer]->devices()[0]);
    driver_node_ids[peer] = peer_device->info().driverNodeId_;
  }
  engines->ConfigureDevicePeers(driver_node_ids.data(), num_devices);

  abce_device_handle_t internal{};
  if (!engines->GetDeviceHandle(internal)) {
    HIP_RETURN(hipErrorNotSupported);
  }
  handle->stateAddress = reinterpret_cast<uint64_t>(internal.engine_set);
  handle->deviceAbiVersion = internal.abi_version;
  HIP_RETURN(hipSuccess);
}

}  // namespace hip
