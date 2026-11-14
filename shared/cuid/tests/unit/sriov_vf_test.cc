// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// On a host a VF's physfn link gives its index, so it is named as a sub-unit
// of the card. A VF whose index cannot be found has no serial of its own: any
// reachable DSN is the physical card's.
//
// The unnamed-VF flag is set on a device whose serial is readable, so the test
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

amdcuid_gpu_info GpuAt(const std::string& render_node, bool unnamed_vf) {
  amdcuid_gpu_info info = {};
  info.header.device_type = AMDCUID_DEVICE_TYPE_GPU;
  info.header.fields.gpu.vendor_id = 0x1002;
  info.render_node = render_node;
  info.bdf = CuidUtilities::readlink_bdf(render_node + "/device");
  info.unnamed_vf = unnamed_vf;
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
    CuidGpu gpu(GpuAt(node, /*unnamed_vf=*/false));
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

  CuidGpu vf(GpuAt(node, /*unnamed_vf=*/true));
  uint64_t fingerprint = 0xDEADBEEF;
  EXPECT_EQ(vf.get_hardware_fingerprint(fingerprint), AMDCUID_STATUS_HW_FINGERPRINT_NOT_FOUND)
      << "a guest's view of a VF answered with " << node << "'s serial, which is the card's";
  EXPECT_EQ(fingerprint, 0u);
}

// A host VF is told by its physfn link, from the PCI function directory itself
// or through a link to it, so a VF bound to vfio, with no DRM node, has the
// same UnitID as when amdgpu drives it.
TEST(cuidtst, HostVfIndexComesFromSysfsWhateverNamesIt) {
  char base[] = "/tmp/cuid_vf_XXXXXX";
  ASSERT_NE(mkdtemp(base), nullptr);
  const std::string root(base);
  const std::string pf = root + "/0000:03:00.0";
  const std::string vf = root + "/0000:03:00.2";
  ASSERT_EQ(mkdir(pf.c_str(), 0755), 0);
  ASSERT_EQ(mkdir(vf.c_str(), 0755), 0);
  ASSERT_EQ(symlink("../0000:03:00.1", (pf + "/virtfn0").c_str()), 0);
  ASSERT_EQ(symlink("../0000:03:00.2", (pf + "/virtfn1").c_str()), 0);
  ASSERT_EQ(symlink("../0000:03:00.0", (vf + "/physfn").c_str()), 0);
  ASSERT_EQ(symlink(vf.c_str(), (root + "/device").c_str()), 0);

  for (const std::string& path : {vf, root + "/device"}) {
    SCOPED_TRACE(path);
    const auto id = CuidUtilities::get_gpu_vf_identity(path);
    EXPECT_TRUE(id.is_vf);
    EXPECT_TRUE(id.index_known);
    EXPECT_EQ(id.unit_id, 2);
  }
  EXPECT_FALSE(CuidUtilities::get_gpu_vf_identity(pf).is_vf);

  // A physfn link to a PF this namespace does not show is still a VF, of
  // unknown index.
  const std::string hidden = root + "/0000:03:00.3";
  ASSERT_EQ(mkdir(hidden.c_str(), 0755), 0);
  ASSERT_EQ(symlink("../0000:04:00.0", (hidden + "/physfn").c_str()), 0);
  const auto id = CuidUtilities::get_gpu_vf_identity(hidden);
  EXPECT_TRUE(id.is_vf);
  EXPECT_FALSE(id.index_known);
  unlink((hidden + "/physfn").c_str());
  rmdir(hidden.c_str());

  for (const char* link :
       {"/device", "/0000:03:00.0/virtfn0", "/0000:03:00.0/virtfn1", "/0000:03:00.2/physfn"})
    unlink((root + link).c_str());
  rmdir(vf.c_str());
  rmdir(pf.c_str());
  rmdir(root.c_str());
}
