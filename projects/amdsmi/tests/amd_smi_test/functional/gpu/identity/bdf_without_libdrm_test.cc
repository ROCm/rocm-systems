// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"

namespace {

std::vector<uint64_t> PublicGpuBdfs() {
  std::vector<uint64_t> bdfs;
  uint32_t socket_count = 0;
  if (amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS) return bdfs;
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS) {
    return bdfs;
  }
  for (auto socket : sockets) {
    uint32_t count = 0;
    if (amdsmi_get_processor_handles(socket, &count, nullptr) != AMDSMI_STATUS_SUCCESS) continue;
    std::vector<amdsmi_processor_handle> handles(count);
    if (amdsmi_get_processor_handles(socket, &count, handles.data()) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    for (auto handle : handles) {
      amdsmi_processor_type_t type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      amdsmi_bdf_t bdf = {};
      if (amdsmi_get_processor_type(handle, &type) == AMDSMI_STATUS_SUCCESS &&
          type == AMDSMI_PROCESSOR_TYPE_AMD_GPU &&
          amdsmi_get_gpu_device_bdf(handle, &bdf) == AMDSMI_STATUS_SUCCESS) {
        bdfs.push_back(bdf.as_uint);
      }
    }
  }
  return bdfs;
}

// The render node and BDF come from sysfs, so they must not depend on libdrm loading.
TEST(GpuFunctionalReadOnly, DrmInitWithoutLibdrmKeepsBdfAndRenderNode) {
  amdsmi_status_t status = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  if (status == AMDSMI_STATUS_DRIVER_NOT_LOADED) GTEST_SKIP() << "amdgpu driver not loaded";
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  std::vector<uint64_t> expected = PublicGpuBdfs();

  amd::smi::AMDSmiDrm drm;
  EXPECT_EQ(drm.init({"libdrm_amdgpu_absent_for_test.so.1"}), AMDSMI_STATUS_FAIL_LOAD_MODULE);
  EXPECT_FALSE(drm.has_libdrm());
  std::vector<uint64_t> actual;
  for (const auto& bdf : drm.get_bdfs()) {
    EXPECT_NE(bdf.as_uint, 0U);
    actual.push_back(bdf.as_uint);
  }
  for (const auto& path : drm.get_drm_paths()) {
    EXPECT_EQ(path.rfind("renderD", 0), 0U) << path;
  }
  drm.cleanup();
  ASSERT_EQ(amdsmi_shut_down(), AMDSMI_STATUS_SUCCESS);

  ASSERT_FALSE(expected.empty());
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(actual, expected);
}

}  // namespace
