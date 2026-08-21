/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>

#if defined(__linux__)
#include <cstdlib>
#include <dirent.h>

static int CountOpenFds() {
  DIR* d = opendir("/proc/self/fd");
  REQUIRE(d != nullptr);
  int dir_fd = dirfd(d);
  int count = 0;
  while (struct dirent* ent = readdir(d)) {
    if (ent->d_name[0] == '.') continue;
    int fd = atoi(ent->d_name);
    if (fd == dir_fd) continue;
    ++count;
  }
  closedir(d);
  return count;
}

HIP_TEST_CASE(Unit_hipModuleLoad_Positive_NoFdLeak) {
  HIP_CHECK(hipFree(nullptr));

  {
    hipModule_t warmup = nullptr;
    HIP_CHECK(hipModuleLoad(&warmup, "empty_module.code"));
    HIP_CHECK(hipModuleUnload(warmup));
  }

  const int baseline = CountOpenFds();

  {
    hipModule_t module = nullptr;
    HIP_CHECK(hipModuleLoad(&module, "empty_module.code"));
    REQUIRE(module != nullptr);
    const int after_load = CountOpenFds();
    INFO("Open FDs before load: " << baseline
         << ", after load (before unload): " << after_load);
    REQUIRE(after_load == baseline);
    HIP_CHECK(hipModuleUnload(module));
  }

  constexpr int kIterations = 32;
  for (int i = 0; i < kIterations; ++i) {
    hipModule_t module = nullptr;
    HIP_CHECK(hipModuleLoad(&module, "empty_module.code"));
    REQUIRE(module != nullptr);
    HIP_CHECK(hipModuleUnload(module));
  }
  const int after_cycles = CountOpenFds();
  INFO("Open FDs before: " << baseline << ", after " << kIterations
       << " load/unload cycles: " << after_cycles);
  REQUIRE(after_cycles == baseline);
}
#endif

HIP_TEST_CASE(Unit_hipModuleLoad_Positive_Basic) {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module = nullptr;
  HIP_CHECK(hipModuleLoad(&module, "empty_module.code"));
  REQUIRE(module != nullptr);
  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipModuleLoad_Negative_Parameters) {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module;

  SECTION("module == nullptr") {
    HIP_CHECK_ERROR(hipModuleLoad(nullptr, "empty_module.code"), hipErrorInvalidValue);
  }

  SECTION("fname == nullptr") {
    HIP_CHECK_ERROR(hipModuleLoad(&module, nullptr), hipErrorInvalidValue);
  }

  SECTION("fname == empty string") {
    HIP_CHECK_ERROR(hipModuleLoad(&module, ""), hipErrorInvalidValue);
  }

  SECTION("fname == non existent file") {
    HIP_CHECK_ERROR(hipModuleLoad(&module, "non existent file"), hipErrorFileNotFound);
  }
}

HIP_TEST_CASE(Unit_hipModuleLoad_Negative_Load_From_A_File_That_Is_Not_A_Module) {
  HIP_CHECK(hipFree(nullptr));
  hipModule_t module;

  HIP_CHECK_ERROR(hipModuleLoad(&module, "not_a_module.txt"), hipErrorInvalidImage);
  HIP_CHECK_ERROR(hipModuleLoad(&module, "empty_file.txt"), hipErrorInvalidImage);
}
