// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file util_unique_handle_test.cpp
/// @brief Unit tests for platform and trait-based handle ownership.

#include "util/unique_handle.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <type_traits>

#ifdef _WIN32
#include <windows.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace {

struct CountingHandleTraits {
  using handle_type = int;

  inline static int closed = -1;

  static handle_type invalid() noexcept { return -1; }
  static bool is_valid(handle_type handle) noexcept { return handle >= 0; }
  static void close(handle_type handle) noexcept { closed = handle; }
};

using CountingHandle = util::BasicUniqueHandle<CountingHandleTraits>;

static_assert(!std::is_copy_constructible_v<CountingHandle>);
static_assert(!std::is_copy_assignable_v<CountingHandle>);
static_assert(std::is_nothrow_move_constructible_v<CountingHandle>);
static_assert(std::is_nothrow_move_assignable_v<CountingHandle>);

TEST(UniqueHandle, MovesAndReleasesOwnership) {
  CountingHandleTraits::closed = -1;
  CountingHandle source(7);
  CountingHandle destination(std::move(source));

  EXPECT_FALSE(source);
  EXPECT_EQ(source.get(), CountingHandleTraits::invalid());
  ASSERT_TRUE(destination);
  EXPECT_EQ(destination.release(), 7);
  EXPECT_FALSE(destination);
  EXPECT_EQ(CountingHandleTraits::closed, -1);
}

TEST(UniqueHandle, MoveAssignmentClosesReplacedHandle) {
  CountingHandleTraits::closed = -1;
  CountingHandle source(7);
  CountingHandle destination(3);

  destination = std::move(source);

  EXPECT_EQ(CountingHandleTraits::closed, 3);
  EXPECT_FALSE(source);
  EXPECT_EQ(source.get(), CountingHandleTraits::invalid());
  ASSERT_TRUE(destination);
  EXPECT_EQ(destination.get(), 7);
}

TEST(UniqueHandle, ResetClosesPreviousHandle) {
  CountingHandleTraits::closed = -1;
  CountingHandle handle(3);
  handle.reset(5);

  EXPECT_EQ(CountingHandleTraits::closed, 3);
  EXPECT_EQ(handle.get(), 5);

  CountingHandleTraits::closed = -1;
  handle.reset(5);
  EXPECT_EQ(CountingHandleTraits::closed, -1);
}

TEST(UniqueHandle, DestructorClosesOwnedHandle) {
  CountingHandleTraits::closed = -1;
  { CountingHandle handle(11); }
  EXPECT_EQ(CountingHandleTraits::closed, 11);
}

TEST(UniqueHandle, ClosesNativePlatformHandle) {
#ifdef _WIN32
  HANDLE event = ::CreateEventW(nullptr, FALSE, FALSE, nullptr);
  ASSERT_NE(event, nullptr);
  ASSERT_NE(event, INVALID_HANDLE_VALUE);
  {
    util::UniqueHandle handle(event);
    ASSERT_TRUE(handle);
  }
  DWORD flags = 0;
  EXPECT_EQ(::GetHandleInformation(event, &flags), FALSE);
  EXPECT_EQ(::GetLastError(), ERROR_INVALID_HANDLE);
#elif defined(__linux__)
  int descriptors[2] = {-1, -1};
  ASSERT_EQ(::pipe(descriptors), 0);
  {
    util::UniqueHandle handle(descriptors[0]);
    ASSERT_TRUE(handle);
  }
  EXPECT_EQ(::fcntl(descriptors[0], F_GETFD), -1);
  EXPECT_EQ(errno, EBADF);
  EXPECT_EQ(::close(descriptors[1]), 0);
#endif
}

#ifdef _WIN32
TEST(UniqueHandle, NormalizesInvalidHandleValue) {
  util::UniqueHandle handle(INVALID_HANDLE_VALUE);
  EXPECT_FALSE(handle);
  EXPECT_EQ(handle.get(), nullptr);
}
#endif

} // namespace
