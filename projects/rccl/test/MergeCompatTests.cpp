/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gtest/gtest.h"

#include <dlfcn.h>
#include <limits.h>
#include <string>
#include <unistd.h>

namespace RcclUnitTesting {

TEST(MergeCompat, LibrcclDlopenSucceeds) {
  char exePath[PATH_MAX] = {};
  ASSERT_GT(readlink("/proc/self/exe", exePath, sizeof(exePath) - 1), 0);

  std::string libPath(exePath);
  auto slash = libPath.rfind('/');
  ASSERT_NE(slash, std::string::npos);
  libPath.erase(slash);
  libPath += "/../librccl.so.1.0";

  void* handle = dlopen(libPath.c_str(), RTLD_NOW | RTLD_LOCAL);
  ASSERT_NE(handle, nullptr) << dlerror();

  void* symbol = dlsym(handle, "ncclGetVersion");
  ASSERT_NE(symbol, nullptr) << dlerror();

  ASSERT_EQ(dlclose(handle), 0);
}

} // namespace RcclUnitTesting
