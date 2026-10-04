// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file cache_allocation_failure_test.cpp
/// @brief Isolated calloc-failure coverage for Cache tag and byte storage.
/// @details Uses the ELF linker's calloc wrapper so fault injection cannot affect
/// the main simulator test executable.
#include "simdojo/components/cache.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <new>

namespace {
using TestCache = simdojo::Cache<6, 4, 2>;
constexpr size_t kLineCount = TestCache::TOTAL_SIZE / TestCache::LINE_SIZE;
enum class Storage { Data, Tags };
thread_local Storage failing_storage = Storage::Data;
thread_local uint32_t failures_remaining = 0;
thread_local uint32_t handler_calls = 0;

class CacheAllocationFailureTest : public ::testing::TestWithParam<Storage> {
protected:
  void SetUp() override {
    failing_storage = GetParam();
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
  const size_t expected_count =
      failing_storage == Storage::Data ? TestCache::TOTAL_SIZE : kLineCount;
  const size_t expected_size =
      failing_storage == Storage::Data ? sizeof(uint8_t) : sizeof(simdojo::CacheTag);
  const bool selected_storage = (count == expected_count && size == expected_size) ||
                                (count == expected_size && size == expected_count);
  if (selected_storage && failures_remaining) {
    --failures_remaining;
    return nullptr;
  }
  return __real_calloc(count, size);
}

TEST_P(CacheAllocationFailureTest, NoHandlerThrowsBadAlloc) {
  failures_remaining = 1;
  EXPECT_THROW({ TestCache store; }, std::bad_alloc);
  EXPECT_EQ(failures_remaining, 0u);
}

TEST_P(CacheAllocationFailureTest, ReturningHandlerRetriesWithZeroContents) {
  std::set_new_handler([] { ++handler_calls; });
  failures_remaining = 2;
  TestCache store;
  EXPECT_EQ(handler_calls, 2u);
  EXPECT_EQ(failures_remaining, 0u);
  for (size_t line = 0; line < kLineCount; ++line) {
    const uint64_t addr = static_cast<uint64_t>(line) * TestCache::LINE_SIZE;
    EXPECT_FALSE(store.lookup(addr));
    simdojo::CacheTag evicted;
    evicted.valid = true;
    TestCache::Allocation allocation = store.allocate_with_data(addr, /*vmid=*/0, &evicted);
    EXPECT_FALSE(evicted.valid);
    for (uint32_t i = 0; i < TestCache::LINE_SIZE; ++i)
      EXPECT_EQ(allocation.data[i], 0);
  }
}

TEST_P(CacheAllocationFailureTest, ThrowingHandlerPropagates) {
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

TEST_P(CacheAllocationFailureTest, CopyAssignmentReusesExistingBacking) {
  constexpr uint64_t kAddr = 0;
  TestCache source;
  TestCache::Allocation source_line = source.allocate_with_data(kAddr, /*vmid=*/7);
  source_line.data[0] = 42;
  TestCache destination;
  TestCache::Allocation previous_line = destination.allocate_with_data(kAddr, /*vmid=*/7);
  uint8_t *previous_bytes = previous_line.data;
  failures_remaining = 1;
  destination = source;
  EXPECT_EQ(failures_remaining, 1u);
  EXPECT_EQ(destination.line_data_for_read(kAddr, /*vmid=*/7), previous_bytes);
  EXPECT_EQ(previous_bytes[0], 42);
  simdojo::CacheTag *assigned_tag = nullptr;
  ASSERT_TRUE(destination.lookup(kAddr, &assigned_tag, /*vmid=*/7));
  EXPECT_EQ(assigned_tag, previous_line.tag);
}

INSTANTIATE_TEST_SUITE_P(CacheStorage, CacheAllocationFailureTest,
                         ::testing::Values(Storage::Data, Storage::Tags),
                         [](const ::testing::TestParamInfo<Storage> &info) {
                           return info.param == Storage::Data ? "Data" : "Tags";
                         });
