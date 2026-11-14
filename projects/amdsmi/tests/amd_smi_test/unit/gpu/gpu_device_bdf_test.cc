// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// A GPU device built without DRM data (libdrm missing) must not report stale
// heap bytes as its BDF. No GPU required.

#include <gtest/gtest.h>

#include <cstring>
#include <new>

#include "amd_smi/impl/amd_smi_gpu_device.h"

namespace {

TEST(GpuUnit, DeviceWithoutDrmDataHasZeroBdf) {
  amd::smi::AMDSmiDrm drm;  // never initialized, so it holds no DRM data
  alignas(amd::smi::AMDSmiGPUDevice) unsigned char storage[sizeof(amd::smi::AMDSmiGPUDevice)];
  std::memset(storage, 0xA5, sizeof(storage));

  auto* device = new (storage) amd::smi::AMDSmiGPUDevice(0, drm);
  EXPECT_EQ(device->get_bdf().as_uint, 0U);
  EXPECT_EQ(device->get_vendor_id(), 0U);
  device->~AMDSmiGPUDevice();
}

}  // namespace
