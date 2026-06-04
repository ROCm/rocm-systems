// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <dlfcn.h>

#ifndef ROCJITSU_AFL_PRELOAD_PATH
#error "ROCJITSU_AFL_PRELOAD_PATH must point to librocjitsu_afl_preload.so"
#endif

namespace {

class DlHandle {
public:
  explicit DlHandle(const char *path) : handle_(dlopen(path, RTLD_LAZY | RTLD_LOCAL)) {}
  DlHandle(const DlHandle &) = delete;
  DlHandle &operator=(const DlHandle &) = delete;
  ~DlHandle() {
    if (handle_ != nullptr)
      dlclose(handle_);
  }

  void *get() const { return handle_; }

private:
  void *handle_ = nullptr;
};

} // namespace

TEST(RocjitsuAflPreloadSymbolsTest, ExportsVectorAddInterceptSurface) {
  DlHandle preload(ROCJITSU_AFL_PRELOAD_PATH);
  ASSERT_NE(preload.get(), nullptr) << dlerror();

  const char *symbols[] = {
      "hipModuleLoadData",
      "hipModuleUnload",
      "hipModuleGetFunction",
      "hipModuleLaunchKernel",
      "hipDeviceSynchronize",
      "hipMemcpy",
      "hsa_code_object_reader_create_from_memory",
      "hsa_code_object_reader_create_from_file",
      "hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size",
      "hsa_code_object_reader_destroy",
      "rocjitsu_afl_persistent_begin",
      "rocjitsu_afl_persistent_end",
  };

  for (const char *symbol : symbols) {
    dlerror();
    void *address = dlsym(preload.get(), symbol);
    EXPECT_NE(address, nullptr) << symbol << ": " << dlerror();
  }
}
