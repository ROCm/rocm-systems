/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef ISOLATED_CHILD_SKIP_HPP
#define ISOLATED_CHILD_SKIP_HPP

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

namespace RcclUnitTesting
{
  // Skips here, or writes the reason to the file named by envVar, since an isolated child's skip scores as a pass.
  inline void SkipOrHandToParent(const char* envVar, const std::string& reason)
  {
    const char* path = std::getenv(envVar);
    if (path == nullptr)
    {
      GTEST_SKIP() << reason;
    }
    std::ofstream out(path);
    out << reason;
    EXPECT_TRUE(out.good()) << "cannot hand the skip to the parent via " << path << ": " << reason;
  }

  // Reason a child left in path via SkipOrHandToParent, or "" if it ran; removes the file.
  inline std::string TakeHandedSkip(const std::string& path)
  {
    std::string reason;
    std::getline(std::ifstream(path), reason);
    std::remove(path.c_str());
    return reason;
  }
}  // namespace RcclUnitTesting

#endif  // ISOLATED_CHILD_SKIP_HPP
