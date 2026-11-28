/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * The generated nccl4py bindings record the NCCL version they were generated for, both in
 * each file's header and in nccl.pyx's _version_span (which becomes nccl.__version__).
 * That has to match ncclGetVersion() of the librccl built from makefiles/version.mk.
 */

#include <gtest/gtest.h>
#include <rccl/rccl.h>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace RcclUnitTesting
{
  namespace
  {
    // First version whose bindings this test accepts: NCCL 2.32.6.
    constexpr int kMinVersionCode = 23206;

    // Finds "<marker>X.Y.Z" in the file and returns X*10000 + Y*100 + Z (NCCL_VERSION), or -1.
    int RecordedVersion(const std::string& path, const std::string& marker)
    {
      std::ifstream in(path);
      if (!in) return -1;
      std::stringstream ss;
      ss << in.rdbuf();
      std::string text = ss.str();
      size_t pos = text.find(marker);
      if (pos == std::string::npos) return -1;
      int major = 0, minor = 0, patch = 0;
      if (sscanf(text.c_str() + pos + marker.size(), "%d.%d.%d", &major, &minor, &patch) != 3) return -1;
      return major * 10000 + minor * 100 + patch;
    }

    // Every version the generated bindings record, labelled by where it came from.
    bool CollectBindingVersions(std::vector<std::pair<std::string, int>>& out)
    {
      const std::string dir = RCCL_NCCL4PY_BINDINGS_DIR;
      if (!std::ifstream(dir + "/nccl.pyx")) return false;
      const char* files[] = {"nccl.pyx", "nccl.pxd", "cynccl.pyx", "cynccl.pxd",
                             "_internal/nccl.pxd", "_internal/nccl_linux.pyx"};
      for (const char* f : files)
        out.emplace_back(f, RecordedVersion(dir + "/" + f, "automatically generated with version "));
      out.emplace_back("nccl.pyx _version_span",
                       RecordedVersion(dir + "/nccl.pyx", "_version_span = \"with version "));
      return true;
    }
  }

  // Consistency only: the bindings record the version of the library they ship with.
  TEST(Nccl4pyVersion, BindingsMatchLibrary)
  {
    std::vector<std::pair<std::string, int>> recorded;
    if (!CollectBindingVersions(recorded))
      GTEST_SKIP() << "nccl4py bindings not found at " << RCCL_NCCL4PY_BINDINGS_DIR << " (installed run?)";

    int libVersion = 0;
    ASSERT_EQ(ncclGetVersion(&libVersion), ncclSuccess);
    for (const auto& r : recorded)
      EXPECT_EQ(r.second, libVersion) << r.first << " records a version other than ncclGetVersion()";
  }

  // The 2.32.6 release versioning: version.mk and every generated binding name 2.32.6 or later.
  TEST(Nccl4pyVersion, RecordsAtLeast_2_32_6)
  {
    std::vector<std::pair<std::string, int>> recorded;
    if (!CollectBindingVersions(recorded))
      GTEST_SKIP() << "nccl4py bindings not found at " << RCCL_NCCL4PY_BINDINGS_DIR << " (installed run?)";

    int libVersion = 0;
    ASSERT_EQ(ncclGetVersion(&libVersion), ncclSuccess);
    EXPECT_GE(libVersion, kMinVersionCode) << "librccl reports a version older than NCCL 2.32.6";
    for (const auto& r : recorded) {
      EXPECT_GE(r.second, kMinVersionCode) << r.first << " records a version older than NCCL 2.32.6";
      EXPECT_EQ(r.second, libVersion) << r.first << " records a version other than ncclGetVersion()";
    }
  }
}
