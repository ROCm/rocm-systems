// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "unit/cuid_gpu_test.h"

#include <gtest/gtest.h>

#include <cstring>
#include <string>
#include <vector>

#include "src/cuid_gpu.h"
#include "src/cuid_util.h"

TestCuidGpuRenderNode::TestCuidGpuRenderNode() {
  SetTitle("CuidGpu Render Node");
  SetDescription(
      "Verify CuidGpu::normalize_render_node preserves GIM PCI-device paths, "
      "strips the trailing /device from DRM paths, and returns arbitrary "
      "paths verbatim.");
}

void TestCuidGpuRenderNode::SetUp() {}

void TestCuidGpuRenderNode::Run() {
  // An unavailable GIM-style PCI path retains its spelling. A live PCI object
  // with DRM nodes is covered by the namespace path-resolution tests.
  {
    const std::string pci_path = "/sys/bus/pci/devices/ffff:ff:1f.7";
    EXPECT_EQ(CuidGpu::normalize_render_node(pci_path), pci_path)
        << "GIM-style PCI device paths must not be trimmed";
  }

  // DRM enumeration passes "/sys/class/drm/<card>/device"; the trailing
  // "/device" must be stripped. card999 is unlikely to exist, so no renderD
  // node is resolved and the trimmed card path is returned unchanged.
  {
    const std::string drm_path = "/sys/class/drm/card999/device";
    EXPECT_EQ(CuidGpu::normalize_render_node(drm_path), "/sys/class/drm/card999")
        << "DRM enumeration paths must have the trailing /device stripped";
  }

  // A path that is neither form must be returned verbatim.
  {
    const std::string path = "/some/custom/path";
    EXPECT_EQ(CuidGpu::normalize_render_node(path), path);
  }
}

TEST(cuidtstUnprivileged, GpuPropagatesInvalidUnitId) {
  // Without a driver CUID a whole GPU is temporary even with a readable
  // serial. A temporary identity needs the host's machine identity; without
  // one the in-range case has no identity to report, and must leave the
  // output alone.
  uint64_t probe = 0;
  const bool have_machine_id =
      CuidUtilities::make_fallback_fingerprint(CuidUtilities::AuxiliaryInput{}, probe) ==
      AMDCUID_STATUS_SUCCESS;
  for (const uint16_t unit_id : {0x1FFF, 0x2000, 0x2001, 0xFFFF}) {
    SCOPED_TRACE(::testing::Message() << "unit=" << unit_id);
    amdcuid_gpu_info info = {};
    info.header.fields.gpu.unit_id = unit_id;
    info.header.fields.gpu.device_id = 0x73A3;
    info.header.fields.gpu.vendor_id = 0x1002;
    info.bdf = "ffff:ff:1f.7";
    info.render_node = "/nonexistent/cuid-vector-gpu";
    CuidGpu gpu(info);
    amdcuid_primary_id id;
    std::memset(&id, 0xA5, sizeof(id));
    const amdcuid_primary_id original = id;
    const amdcuid_status_t status = gpu.get_primary_cuid(id);
    if (unit_id > 0x1FFF) {
      EXPECT_EQ(status, AMDCUID_STATUS_INVALID_ARGUMENT);
      EXPECT_EQ(std::memcmp(&id, &original, sizeof(id)), 0);
    } else if (!have_machine_id) {
      EXPECT_EQ(status, AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND);
      EXPECT_EQ(std::memcmp(&id, &original, sizeof(id)), 0);
    } else {
      ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);
      EXPECT_EQ(id.raw_bits[8], 0xFF);
      EXPECT_EQ(id.raw_bits[14] & 0x1F, 0x1F);
      EXPECT_NE(id.raw_bits[14] & 0x20, 0);
    }
  }
}

// Another vendor's display device, a BMC's for instance, is not a GPU
// component and must not be given a temporary CUID.
TEST(cuidtstUnprivileged, OnlyAmdGpusAreListed) {
  uint32_t count = 0;
  if (amdcuid_get_all_handles(nullptr, &count) != AMDCUID_STATUS_INSUFFICIENT_SIZE)
    GTEST_SKIP() << "no components";
  std::vector<amdcuid_id_t> handles(count);
  ASSERT_EQ(amdcuid_get_all_handles(handles.data(), &count), AMDCUID_STATUS_SUCCESS);
  for (const auto& handle : handles) {
    amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
    uint32_t length = sizeof(type);
    ASSERT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE, &type, &length),
              AMDCUID_STATUS_SUCCESS);
    if (type != AMDCUID_DEVICE_TYPE_GPU) continue;
    uint16_t vendor = 0;
    length = sizeof(vendor);
    const auto status =
        amdcuid_query_device_property(handle, AMDCUID_QUERY_VENDOR_ID, &vendor, &length);
    // A partition whose driver metadata this caller cannot read has no vendor.
    if (status == AMDCUID_STATUS_UNSUPPORTED) continue;
    ASSERT_EQ(status, AMDCUID_STATUS_SUCCESS);
    EXPECT_EQ(vendor, 0x1002) << amdcuid_id_to_string(handle);
  }
}
