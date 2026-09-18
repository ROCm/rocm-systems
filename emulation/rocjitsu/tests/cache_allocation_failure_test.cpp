// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cache_allocation_failure_test.cpp
/// @brief Isolated calloc-failure coverage for Cache byte storage.
/// @details Uses the ELF linker's calloc wrapper so fault injection cannot affect
/// the main simulator test executable.
#include "simdojo/components/cache.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <new>

namespace {
using TestCache = simdojo::Cache<6, 4, 2>;
thread_local uint32_t failures_remaining = 0;
thread_local uint32_t handler_calls = 0;

class CacheAllocationFailureTest : public ::testing::Test {
protected:
  void SetUp() override {
    previous_handler_ = std::set_new_handler(nullptr);
    failures_remaining = 0;
    handler_calls = 0;
  }
  void TearDown() override {
    failures_remaining = 0;
    std::set_new_handler(previous_handler_);
  }

private:
  std::new_handler previous_handler_ = nullptr;
};
} // namespace

extern "C" void *__real_calloc(std::size_t count, std::size_t size);
extern "C" void *__wrap_calloc(std::size_t count, std::size_t size) {
  const bool cache_bytes = (count == TestCache::TOTAL_SIZE && size == sizeof(uint8_t)) ||
                           (count == sizeof(uint8_t) && size == TestCache::TOTAL_SIZE);
  if (cache_bytes && failures_remaining) {
    --failures_remaining;
    return nullptr;
  }
  return __real_calloc(count, size);
}

TEST_F(CacheAllocationFailureTest, NoHandlerThrowsBadAlloc) {
  failures_remaining = 1;
  EXPECT_THROW({ TestCache store; }, std::bad_alloc);
  EXPECT_EQ(failures_remaining, 0u);
}

TEST_F(CacheAllocationFailureTest, ReturningHandlerRetriesWithZeroContents) {
  std::set_new_handler([] { ++handler_calls; });
  failures_remaining = 2;
  TestCache store;
  EXPECT_EQ(handler_calls, 2u);
  EXPECT_EQ(failures_remaining, 0u);
  TestCache::Allocation allocation = store.allocate_with_data(0);
  for (uint32_t i = 0; i < TestCache::LINE_SIZE; ++i)
    EXPECT_EQ(allocation.data[i], 0);
}

TEST_F(CacheAllocationFailureTest, ThrowingHandlerPropagates) {
  struct HandlerFailure {};
  std::set_new_handler([] {
    ++handler_calls;
    throw HandlerFailure{};
  });
  failures_remaining = 1;
  EXPECT_THROW({ TestCache store; }, HandlerFailure);
  EXPECT_EQ(handler_calls, 1u);
  EXPECT_EQ(failures_remaining, 0u);
}

TEST_F(CacheAllocationFailureTest, CopyAssignmentReusesExistingBacking) {
  constexpr uint64_t kAddr = 0;
  TestCache source;
  TestCache::Allocation source_line = source.allocate_with_data(kAddr, /*vmid=*/7);
  source_line.data[0] = 42;
  TestCache destination;
  uint8_t *previous_bytes = destination.allocate_with_data(kAddr, /*vmid=*/7).data;
  failures_remaining = 1;
  destination = source;
  EXPECT_EQ(failures_remaining, 1u);
  EXPECT_EQ(destination.line_data_for_read(kAddr, /*vmid=*/7), previous_bytes);
  EXPECT_EQ(previous_bytes[0], 42);
}
