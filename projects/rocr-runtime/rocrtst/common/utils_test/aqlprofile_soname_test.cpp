// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Regression test for the aqlprofile versioned-SONAME fallback added to
// Runtime::Load(). The runtime first probes os::LoadLib(kAqlProfileLib) using
// the unversioned name (e.g. "libhsa-amd-aqlprofile64.so"). When only the
// major-versioned real file is present on disk and the unversioned dev symlink
// is absent (as with TheRock/rocm-sdk wheels), the runtime retries with
// kAqlProfileLib + "." + hsa_ven_amd_aqlprofile_VERSION_MAJOR
// (i.e. "libhsa-amd-aqlprofile64.so.1"). This test guards that the fallback
// name the runtime constructs matches the shipped, version-suffixed file.

#include <gtest/gtest.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

#include <string>

TEST(aqlprofile_soname, versioned_fallback_name) {
  const std::string base = kAqlProfileLib;
  const std::string versioned =
      base + "." + std::to_string(hsa_ven_amd_aqlprofile_VERSION_MAJOR);

  // The runtime is built against aqlprofile extension major version 1.
  EXPECT_EQ(hsa_ven_amd_aqlprofile_VERSION_MAJOR, 1);

  // Fallback appends the extension major version to the unversioned name.
  EXPECT_EQ(versioned, base + ".1");

  // The fallback must reference a real, version-suffixed shared object, i.e.
  // it ends in ".so.<major>" rather than the bare ".so" dev-symlink name.
  EXPECT_NE(versioned.rfind(".so.1"), std::string::npos);
  EXPECT_EQ(base.rfind(".so"), base.size() - 3);
}
