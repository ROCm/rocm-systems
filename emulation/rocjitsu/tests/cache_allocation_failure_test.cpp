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
#include <utility>

namespace {
using TestCache = simdojo::Cache<6, 4, 2>;
using LazyCache = simdojo::Cache<6, 4, 2, simdojo::LRUPolicy<4, 2>, /*LazyStorage=*/true>;
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

TEST_P(CacheAllocationFailureTest, LazyConstructionCopyAndFreshInvalidationDeferAllocation) {
  failures_remaining = 1;
  LazyCache source;
  source.invalidate_all();
  source.invalidate(0, /*vmid=*/7);
  source.invalidate_all_vmids(0);
  EXPECT_FALSE(source.lookup(0, nullptr, /*vmid=*/7));
  EXPECT_EQ(std::as_const(source).peek(0, /*vmid=*/7), nullptr);
  LazyCache copied(source);
  LazyCache assigned;
  assigned = copied;
  assigned.invalidate_all();
  EXPECT_EQ(failures_remaining, 1u);
}

TEST_P(CacheAllocationFailureTest, LazyDataQueriesOnMissDoNotAllocate) {
  LazyCache store;
  failures_remaining = 1;
  EXPECT_EQ(store.line_data_for_read(0, /*vmid=*/7), nullptr);
  EXPECT_EQ(store.line_data_for_write(0, /*vmid=*/7), nullptr);
  EXPECT_EQ(failures_remaining, 1u);
  EXPECT_THROW(store.allocate_with_data(0, /*vmid=*/7), std::bad_alloc);

  // A failed first fill can leave only the tag store materialized. Queries
  // must still report a miss without allocating the missing byte store.
  failures_remaining = 1;
  EXPECT_EQ(store.line_data_for_read(0, /*vmid=*/7), nullptr);
  EXPECT_EQ(store.line_data_for_write(0, /*vmid=*/7), nullptr);
  EXPECT_EQ(failures_remaining, 1u);
  failures_remaining = 0;
  auto line = store.allocate_with_data(0, /*vmid=*/7);
  ASSERT_NE(line.data, nullptr);
  EXPECT_EQ(store.line_data_for_read(0, /*vmid=*/7), line.data);
  EXPECT_EQ(store.line_data_for_write(0, /*vmid=*/7), line.data);
}

TEST_P(CacheAllocationFailureTest, LazyAllocationFailureLeavesNoResidentLine) {
  LazyCache store;
  failures_remaining = 1;
  EXPECT_THROW(store.allocate_with_data(0, /*vmid=*/7), std::bad_alloc);
  EXPECT_EQ(failures_remaining, 0u);
  EXPECT_FALSE(store.lookup(0, nullptr, /*vmid=*/7));
  auto line = store.allocate_with_data(0, /*vmid=*/7);
  ASSERT_NE(line.data, nullptr);
  for (uint32_t byte = 0; byte < LazyCache::LINE_SIZE; ++byte)
    EXPECT_EQ(line.data[byte], 0);
}

TEST_P(CacheAllocationFailureTest, LazyReturningHandlerRetriesWithZeroContents) {
  LazyCache store;
  std::set_new_handler([] { ++handler_calls; });
  failures_remaining = 2;
  auto line = store.allocate_with_data(0, /*vmid=*/7);
  EXPECT_EQ(handler_calls, 2u);
  EXPECT_EQ(failures_remaining, 0u);
  for (uint32_t byte = 0; byte < LazyCache::LINE_SIZE; ++byte)
    EXPECT_EQ(line.data[byte], 0);
}

TEST_P(CacheAllocationFailureTest, LazyResidentCopyAndInvalidationPreserveIndependentContents) {
  LazyCache source;
  auto line = source.allocate_with_data(0, /*vmid=*/7);
  line.data[0] = 42;
  LazyCache copied(source);
  LazyCache assigned;
  assigned = source;
  source.invalidate_all();
  EXPECT_FALSE(source.lookup(0, nullptr, /*vmid=*/7));
  ASSERT_TRUE(copied.lookup(0, nullptr, /*vmid=*/7));
  ASSERT_TRUE(assigned.lookup(0, nullptr, /*vmid=*/7));
  EXPECT_EQ(copied.line_data_for_read(0, /*vmid=*/7)[0], 42);
  EXPECT_EQ(assigned.line_data_for_read(0, /*vmid=*/7)[0], 42);
  copied.invalidate(0, /*vmid=*/7);
  EXPECT_FALSE(copied.lookup(0, nullptr, /*vmid=*/7));
  EXPECT_TRUE(assigned.lookup(0, nullptr, /*vmid=*/7));
}

TEST_P(CacheAllocationFailureTest, LazyCopyFailurePreservesEmptyDestination) {
  LazyCache source;
  source.allocate_with_data(0, /*vmid=*/7).data[0] = 42;
  LazyCache destination;
  failures_remaining = 1;
  EXPECT_THROW(destination = source, std::bad_alloc);
  EXPECT_EQ(failures_remaining, 0u);
  EXPECT_FALSE(destination.lookup(0, nullptr, /*vmid=*/7));
  destination.allocate_with_data(256, /*vmid=*/7);
  EXPECT_FALSE(destination.lookup(0, nullptr, /*vmid=*/7));
  EXPECT_EQ(source.line_data_for_read(0, /*vmid=*/7)[0], 42);
}

TEST_P(CacheAllocationFailureTest, LazyCopyFailurePreservesResidentDestination) {
  LazyCache source;
  source.allocate_with_data(0, /*vmid=*/7).data[0] = 42;
  LazyCache destination;
  destination.allocate_with_data(256, /*vmid=*/7).data[0] = 19;
  failures_remaining = 1;
  EXPECT_THROW(destination = source, std::bad_alloc);
  EXPECT_EQ(failures_remaining, 0u);
  EXPECT_FALSE(destination.lookup(0, nullptr, /*vmid=*/7));
  ASSERT_TRUE(destination.lookup(256, nullptr, /*vmid=*/7));
  EXPECT_EQ(destination.line_data_for_read(256, /*vmid=*/7)[0], 19);
  destination.allocate_with_data(512, /*vmid=*/7);
  EXPECT_FALSE(destination.lookup(0, nullptr, /*vmid=*/7));
  EXPECT_EQ(source.line_data_for_read(0, /*vmid=*/7)[0], 42);
}

INSTANTIATE_TEST_SUITE_P(CacheStorage, CacheAllocationFailureTest,
                         ::testing::Values(Storage::Data, Storage::Tags),
                         [](const ::testing::TestParamInfo<Storage> &info) {
                           return info.param == Storage::Data ? "Data" : "Tags";
                         });
