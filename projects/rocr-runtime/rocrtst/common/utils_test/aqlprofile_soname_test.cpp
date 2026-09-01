// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Regression test for the aqlprofile SONAME probing in Runtime::Load(). The
// runtime prefers the version-suffixed name that matches the ABI it is built
// against, i.e. kAqlProfileLib + "." + hsa_ven_amd_aqlprofile_VERSION_MAJOR
// ("libhsa-amd-aqlprofile64.so.1"), and only falls back to the unversioned dev
// symlink ("libhsa-amd-aqlprofile64.so") if the versioned file is absent. The
// versioned name is what TheRock/rocm-sdk core wheels actually ship. This test
// guards that the constructed versioned name matches that shipped file.

#include <gtest/gtest.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <string>

TEST(aqlprofile_soname, versioned_preferred_name) {
  const std::string base = kAqlProfileLib;
  const std::string versioned =
      base + "." + std::to_string(hsa_ven_amd_aqlprofile_VERSION_MAJOR);

  // The runtime is built against aqlprofile extension major version 1.
  EXPECT_EQ(hsa_ven_amd_aqlprofile_VERSION_MAJOR, 1);

  // The preferred name appends the expected extension major version.
  EXPECT_EQ(versioned, base + ".1");

  // It must reference a real, version-suffixed shared object, i.e. it ends in
  // ".so.<major>" rather than the bare ".so" dev-symlink name.
  EXPECT_NE(versioned.rfind(".so.1"), std::string::npos);
  EXPECT_EQ(base.rfind(".so"), base.size() - 3);
}
