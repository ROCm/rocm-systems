// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// amdsmi_get_gpu_driver_info_v2() through the public API, with a synthetic GPU that has no DRM
// device behind it, and the layout of both driver info structs. No GPU required.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"
#include "test_fixture_utils.h"

namespace {

using amd::smi::AMDSmiDrm;
using amd::smi::AMDSmiGPUDevice;
using amd::smi::testing::ScopedProcessorRegistration;

constexpr auto kCard = "card0";

// Existing binaries pass a 768-byte amdsmi_driver_info_t, so it must never grow again.
TEST(GpuUnit, DriverInfoV1LayoutIsUnchanged) {
  EXPECT_EQ(sizeof(amdsmi_driver_info_t), 768U);
  EXPECT_EQ(offsetof(amdsmi_driver_info_t, driver_date), 256U);
  EXPECT_EQ(offsetof(amdsmi_driver_info_t, driver_name), 512U);
}

TEST(GpuUnit, DriverInfoV2ReservedTailFillsStruct) {
  EXPECT_EQ(sizeof(amdsmi_driver_info_v2_t), 4096U);
  EXPECT_EQ(offsetof(amdsmi_driver_info_v2_t, driver_kernel_version), 768U);
  EXPECT_EQ(offsetof(amdsmi_driver_info_v2_t, driver_package), 1792U);
  EXPECT_EQ(offsetof(amdsmi_driver_info_v2_t, reserved), 1796U);
  EXPECT_EQ(offsetof(amdsmi_driver_info_v2_t, reserved) + sizeof(amdsmi_driver_info_v2_t::reserved),
            sizeof(amdsmi_driver_info_v2_t));
}

TEST(GpuUnit, DriverInfoV2BeforeInitIsNotInit) {
  auto info = amdsmi_driver_info_v2_t{};

  EXPECT_EQ(amdsmi_get_gpu_driver_info_v2(nullptr, &info), AMDSMI_STATUS_NOT_INIT);
}

TEST(GpuUnit, DriverInfoV2NullOutputIsInval) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + kCard);
  AMDSmiDrm drm;
  AMDSmiGPUDevice gpu_device(rocm_device.index(), kCard, amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  EXPECT_EQ(amdsmi_get_gpu_driver_info_v2(registration.handle(), nullptr), AMDSMI_STATUS_INVAL);
}

TEST(GpuUnit, DriverInfoV2FailsLikeV1AndZeroesOutput) {
  ScopedAmdsmiInit smi_init;
  FakeSysfsTree tree;
  ScopedRocmSmiDevice rocm_device(tree.root() + "/" + kCard);
  AMDSmiDrm drm;  // never initialized, so the v1 call cannot read the DRM version
  AMDSmiGPUDevice gpu_device(rocm_device.index(), kCard, amdsmi_bdf_t{}, drm);
  ScopedProcessorRegistration registration(&gpu_device);

  auto v1 = amdsmi_driver_info_t{};
  ASSERT_EQ(amdsmi_get_gpu_driver_info(registration.handle(), &v1), AMDSMI_STATUS_NOT_SUPPORTED);

  amdsmi_driver_info_v2_t v2;
  std::memset(&v2, 0xA5, sizeof(v2));
  EXPECT_EQ(amdsmi_get_gpu_driver_info_v2(registration.handle(), &v2), AMDSMI_STATUS_NOT_SUPPORTED);
  const auto* bytes = reinterpret_cast<const unsigned char*>(&v2);
  EXPECT_TRUE(std::all_of(bytes, bytes + sizeof(v2), [](unsigned char byte) { return byte == 0; }));
}

}  // namespace
