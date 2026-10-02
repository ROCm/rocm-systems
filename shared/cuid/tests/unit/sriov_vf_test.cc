// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// On a host a VF's physfn link gives its index, so it is named as a sub-unit
// of the card. In a guest there is no physfn link, and any reachable DSN is the
// physical card's, so an unnamed VF has no serial of its own and takes the
// auxiliary path (the kernel publishes no CUID attributes on a VF).
//
// The unnamed-VF flag is set on a device whose serial is readable, so the test
// fails if the refusal is removed; it skips when no such device exists.
//
// The lifecycle GPU-path fixture checks primary/derived refusal with fabricated
// driver attributes, including the auxiliary whole-VF path.

#include <dirent.h>
#include <gtest/gtest.h>
#include <unistd.h>

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
