// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// A VF is named by its own BDF with UnitID 0. It has no serial of its own:
// any reachable DSN is the physical card's.
//
// The host-VF flag is set on a device whose serial is readable, so the test
// fails if the refusal is removed; it skips when no such device exists.
//
// The lifecycle GPU-path fixture checks primary/derived refusal with fabricated
// driver attributes, including the auxiliary whole-VF path.

#include <dirent.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <string>

#include "src/cuid_gpu.h"
#include "src/cuid_util.h"

namespace {

amdcuid_gpu_info GpuAt(const std::string& render_node, bool host_vf) {
  amdcuid_gpu_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
  info.header.fields.gpu.vendor_id = 0x1002;
  info.render_node = render_node;
  info.bdf = CuidUtilities::readlink_bdf(render_node + "/device");
  info.host_vf = host_vf;
  return info;
}

// A render node whose serial this process can read.
std::string RenderNodeWithAReachableSerial() {
  DIR* dir = opendir("/sys/class/drm");
  if (!dir) return "";

  std::string found;
  while (const dirent* entry = readdir(dir)) {
    if (std::strncmp(entry->d_name, "renderD", 7) != 0) continue;

    const std::string node = std::string("/sys/class/drm/") + entry->d_name;
    CuidGpu gpu(GpuAt(node, /*host_vf=*/false));
    uint64_t fingerprint = 0;
    if (gpu.get_hardware_fingerprint(fingerprint) == AMDCUID_STATUS_SUCCESS && fingerprint != 0) {
      found = node;
      break;
    }
  }
  closedir(dir);
  return found;
}

}  // namespace

TEST(cuidtst, AVfWeCannotNameHasNoSerialOfItsOwn) {
  if (geteuid() != 0) {
    GTEST_SKIP() << "get_hardware_fingerprint() is privileged; run with sudo to enable.";
  }
  const std::string node = RenderNodeWithAReachableSerial();
  if (node.empty()) {
    GTEST_SKIP() << "no GPU here answers with a serial, so refusing one proves nothing";
  }

  CuidGpu vf(GpuAt(node, /*host_vf=*/true));
  uint64_t fingerprint = 0xDEADBEEF;
  EXPECT_EQ(vf.get_hardware_fingerprint(fingerprint), AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND)
      << "a guest's view of a VF answered with " << node << "'s serial, which is the card's";
  EXPECT_EQ(fingerprint, 0u);
}

// A host VF is told by its physfn link, from the PCI function directory itself
// or through a link to it, so a VF bound to vfio, with no DRM node, is told
// alike. Its temporary input is its own address with UnitID 0.
TEST(cuidtst, HostVfIsNamedByItsOwnAddress) {
  char base[] = "/tmp/cuid_vf_XXXXXX";
  ASSERT_NE(mkdtemp(base), nullptr);
  const std::string root(base);
  const std::string pf = root + "/0000:03:00.0";
  const std::string vf = root + "/0000:03:00.2";
  ASSERT_EQ(mkdir(pf.c_str(), 0755), 0);
  ASSERT_EQ(mkdir(vf.c_str(), 0755), 0);
  ASSERT_EQ(symlink("../0000:03:00.2", (pf + "/virtfn1").c_str()), 0);
  ASSERT_EQ(symlink("../0000:03:00.0", (vf + "/physfn").c_str()), 0);
  ASSERT_EQ(symlink(vf.c_str(), (root + "/device").c_str()), 0);

  EXPECT_TRUE(CuidUtilities::is_host_vf(vf));
  EXPECT_TRUE(CuidUtilities::is_host_vf(root + "/device"));
  EXPECT_FALSE(CuidUtilities::is_host_vf(pf));

  // A physfn link to a PF this namespace does not show is still a VF.
  const std::string hidden = root + "/0000:03:00.3";
  ASSERT_EQ(mkdir(hidden.c_str(), 0755), 0);
  ASSERT_EQ(symlink("../0000:04:00.0", (hidden + "/physfn").c_str()), 0);
  EXPECT_TRUE(CuidUtilities::is_host_vf(hidden));
  unlink((hidden + "/physfn").c_str());
  rmdir(hidden.c_str());

  amdcuid_gpu_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
  info.header.fields.gpu.vendor_id = 0x1002;
  info.header.fields.gpu.device_id = 0x73a3;
  info.render_node = vf;
  info.bdf = "0000:03:00.2";
  info.host_vf = true;
  CuidUtilities::AuxiliaryInput input{};
  ASSERT_EQ(CuidGpu(info).get_temporary_input(input), AMDCUID_STATUS_SUCCESS);
  EXPECT_EQ(input.unit_id, 0);
  EXPECT_EQ(input.routing_id, CuidUtilities::routing_id_from_bdf("0000:03:00.2"));

  for (const char* link : {"/device", "/0000:03:00.0/virtfn1", "/0000:03:00.2/physfn"})
    unlink((root + link).c_str());
  rmdir(vf.c_str());
  rmdir(pf.c_str());
  rmdir(root.c_str());
}
