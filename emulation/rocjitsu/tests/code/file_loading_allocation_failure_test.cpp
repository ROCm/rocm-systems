// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/rj_code.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <new>
#include <string>

namespace {
thread_local bool fail_allocation = false;
thread_local size_t selected_size = 0;
thread_local bool injected = false;

class ScopedAllocationFailure {
public:
  explicit ScopedAllocationFailure(size_t size) {
    selected_size = size;
    injected = false;
    fail_allocation = true;
  }
  ~ScopedAllocationFailure() { fail_allocation = false; }
  ScopedAllocationFailure(const ScopedAllocationFailure &) = delete;
  ScopedAllocationFailure &operator=(const ScopedAllocationFailure &) = delete;
};
} // namespace

// This executable owns the replacement so other test binaries are unaffected.
void *operator new(std::size_t size) {
  if (fail_allocation && (selected_size == 0 || selected_size == size)) {
    fail_allocation = false;
    injected = true;
    throw std::bad_alloc();
  }
  for (;;) {
    if (void *memory = std::malloc(size == 0 ? 1 : size))
      return memory;
    const std::new_handler handler = std::get_new_handler();
    if (!handler)
      throw std::bad_alloc();
    handler();
  }
}

// Inlining free into new/delete callers triggers GCC's mismatched-new-delete warning.
RJ_NOINLINE void operator delete(void *memory) noexcept { std::free(memory); }
RJ_NOINLINE void operator delete(void *memory, std::size_t) noexcept { std::free(memory); }

// Keep nothrow allocations paired with our free-based delete under ASan.
// GoogleTest's std::stable_sort buffer would otherwise use ASan's nothrow new.
void *operator new(std::size_t size, const std::nothrow_t &) noexcept {
  try {
    return ::operator new(size);
  } catch (...) {
    return nullptr;
  }
}

RJ_NOINLINE void operator delete(void *memory, const std::nothrow_t &) noexcept {
  ::operator delete(memory);
}

TEST(FileLoadingAllocationFailureTest, TranslatesImageAllocationFailure) {
  constexpr size_t kImageSize = 65536;
  const rocjitsu::test::ScopedTempFile file("rocjitsu_allocation_");
  file.write(std::string(kImageSize, '\0'));
  char sentinel;
  auto *const original = reinterpret_cast<rj_code_executable_t *>(&sentinel);
  rj_code_executable_t *executable = original;

  // The same bytes are rejected normally when allocation succeeds.
  ASSERT_EQ(rj_code_executable_create(file.path().c_str(), &executable),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  ASSERT_EQ(executable, original);

  rj_status_t status;
  {
    const ScopedAllocationFailure failure(kImageSize);
    status = rj_code_executable_create(file.path().c_str(), &executable);
  }
  EXPECT_TRUE(injected);
  EXPECT_EQ(status, ROCJITSU_STATUS_OUT_OF_RESOURCES);
  EXPECT_EQ(executable, original);
}

TEST(FileLoadingAllocationFailureTest, TranslatesHandleAllocationFailure) {
  char sentinel;
  auto *const original = reinterpret_cast<rj_code_executable_t *>(&sentinel);
  rj_code_executable_t *executable = original;
  rj_status_t status;
  {
    // The first allocation creates the handle, before the path is inspected.
    const ScopedAllocationFailure failure(0);
    status = rj_code_executable_create("unused", &executable);
  }
  EXPECT_TRUE(injected);
  EXPECT_EQ(status, ROCJITSU_STATUS_OUT_OF_RESOURCES);
  EXPECT_EQ(executable, original);
}
