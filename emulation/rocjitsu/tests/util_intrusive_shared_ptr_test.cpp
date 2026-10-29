// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/intrusive_shared_ptr.h"

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

struct Tracked {
  explicit Tracked(std::atomic<int> &destroyed) : destroyed(destroyed) {}
  ~Tracked() { destroyed.fetch_add(1, std::memory_order_relaxed); }
  std::atomic<int> &destroyed;
};

using TrackedPtr = util::IntrusiveSharedPtr<util::RefCounted<Tracked>>;
using IntPtr = util::IntrusiveSharedPtr<util::RefCounted<int>>;
using ConstIntPtr = util::IntrusiveSharedPtr<const util::RefCounted<int>>;

static_assert(sizeof(IntPtr) == sizeof(void *));
static_assert(std::is_nothrow_copy_constructible_v<IntPtr>);
static_assert(std::is_nothrow_move_constructible_v<IntPtr>);
static_assert(std::is_nothrow_copy_assignable_v<IntPtr>);
static_assert(std::is_nothrow_move_assignable_v<IntPtr>);
static_assert(!std::is_copy_constructible_v<util::RefCounted<int>>);
static_assert(!std::is_move_constructible_v<util::RefCounted<int>>);
static_assert(!std::is_copy_constructible_v<util::RefCountedCacheAligned<int>>);
static_assert(!std::is_move_constructible_v<util::RefCountedCacheAligned<int>>);
static_assert(!std::is_constructible_v<IntPtr, ConstIntPtr>);
// Dereferencing a const-element handle must expose a const value.
static_assert(std::is_same_v<decltype((std::declval<ConstIntPtr>()->value)), const int &>);

TEST(IntrusiveSharedPtr, EmptyHandlesCanBeCopiedMovedAndReset) {
  IntPtr empty;
  IntPtr null(nullptr);
  IntPtr copy(empty);
  IntPtr moved(std::move(copy));
  empty = moved;
  moved = std::move(null);
  empty.reset();
  EXPECT_FALSE(empty);
  EXPECT_FALSE(null);
  EXPECT_FALSE(copy);
  EXPECT_FALSE(moved);
  EXPECT_EQ(empty.get(), nullptr);
}

TEST(IntrusiveSharedPtr, CopiesRetainUntilLastOwner) {
  std::atomic<int> destroyed{0};
  auto owner = TrackedPtr::make(destroyed);
  ASSERT_TRUE(owner);
  EXPECT_EQ(owner->ref_count().load(), 1u);
  {
    auto copy = owner;
    EXPECT_EQ(copy.get(), owner.get());
    EXPECT_EQ(&*copy, owner.get());
    EXPECT_EQ(owner->ref_count().load(), 2u);
    owner.reset();
    EXPECT_EQ(destroyed.load(), 0);
    EXPECT_EQ(copy->ref_count().load(), 1u);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(IntrusiveSharedPtr, MovesTransferWithoutIncrementing) {
  std::atomic<int> destroyed{0};
  {
    auto source = TrackedPtr::make(destroyed);
    auto *address = source.get();
    auto moved = std::move(source);
    EXPECT_FALSE(source);
    EXPECT_EQ(moved.get(), address);
    EXPECT_EQ(moved->ref_count().load(), 1u);
    TrackedPtr destination;
    destination = std::move(moved);
    EXPECT_FALSE(moved);
    EXPECT_EQ(destination.get(), address);
    EXPECT_EQ(destination->ref_count().load(), 1u);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(IntrusiveSharedPtr, AssignmentDestroysReplacedObject) {
  std::atomic<int> destroyed{0};
  auto first = TrackedPtr::make(destroyed);
  auto second = TrackedPtr::make(destroyed);
  first = second;
  EXPECT_EQ(destroyed.load(), 1);
  EXPECT_EQ(first.get(), second.get());
  second = nullptr;
  EXPECT_EQ(destroyed.load(), 1);
  first.reset();
  EXPECT_EQ(destroyed.load(), 2);
}

TEST(IntrusiveSharedPtr, SelfAssignmentPreservesOwnership) {
  std::atomic<int> destroyed{0};
  auto owner = TrackedPtr::make(destroyed);
  auto &alias = owner;
  owner = alias;
  owner = std::move(alias);
  ASSERT_TRUE(owner);
  EXPECT_EQ(owner->ref_count().load(), 1u);
  EXPECT_EQ(destroyed.load(), 0);
  owner.reset();
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(IntrusiveSharedPtr, SwapExchangesOwnership) {
  auto first = IntPtr::make(3);
  auto second = IntPtr::make(7);
  using std::swap;
  swap(first, second);
  EXPECT_EQ(first->value, 7);
  EXPECT_EQ(second->value, 3);
  first.swap(second);
  EXPECT_EQ(first->value, 3);
  EXPECT_EQ(second->value, 7);
}

TEST(IntrusiveSharedPtr, ConstConversionRetainsOrTransfersOwnership) {
  auto owner = IntPtr::make(42);
  ConstIntPtr copy(owner);
  EXPECT_EQ(owner->ref_count().load(), 2u);
  ConstIntPtr moved(std::move(owner));
  EXPECT_FALSE(owner);
  EXPECT_EQ(moved.get(), copy.get());
  EXPECT_EQ(moved->value, 42);
  EXPECT_EQ(moved->ref_count().load(), 2u);
  copy.reset();
  EXPECT_EQ(moved->ref_count().load(), 1u);

  ConstIntPtr assigned;
  auto replacement = IntPtr::make(17);
  assigned = replacement;
  EXPECT_EQ(assigned->value, 17);
  assigned = std::move(replacement);
  EXPECT_FALSE(replacement);
  EXPECT_EQ(assigned->ref_count().load(), 1u);
}

struct CustomCounted final {
  explicit CustomCounted(std::atomic<int> &destroyed) : destroyed_(destroyed) {}

private:
  template <typename> friend class util::IntrusiveSharedPtr;
  ~CustomCounted() { destroyed_.fetch_add(1, std::memory_order_relaxed); }
  void add_ref() const noexcept { ++refs_; }
  void release_ref() const noexcept {
    if (--refs_ == 0)
      delete this;
  }
  std::atomic<int> &destroyed_;
  mutable uint32_t refs_ = 1;
};

TEST(IntrusiveSharedPtr, SupportsPrivateOwnershipHooksWithNonAtomicCounter) {
  std::atomic<int> destroyed{0};
  {
    auto owner = util::IntrusiveSharedPtr<CustomCounted>::make(destroyed);
    util::IntrusiveSharedPtr<const CustomCounted> copy = owner;
    owner.reset();
    EXPECT_EQ(destroyed.load(), 0);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(IntrusiveSharedPtr, IndependentHandlesCanCopyAndReleaseConcurrently) {
  std::atomic<int> destroyed{0};
  constexpr int kThreads = 8;
  std::barrier start(kThreads + 1);
  std::vector<std::jthread> workers;
  auto owner = TrackedPtr::make(destroyed);
  for (int i = 0; i < kThreads; ++i) {
    workers.emplace_back([local = owner, &start]() mutable {
      start.arrive_and_wait();
      for (int j = 0; j < 2000; ++j) {
        auto copy = local;
        auto moved = std::move(copy);
        copy = moved;
      }
      local.reset();
    });
  }
  owner.reset();
  EXPECT_EQ(destroyed.load(), 0);
  start.arrive_and_wait();
  workers.clear();
  EXPECT_EQ(destroyed.load(), 1);
}

struct Throws {
  explicit Throws(std::atomic<int> &destroyed) : member(destroyed) {
    throw std::runtime_error("construction failed");
  }
  Tracked member;
};

TEST(IntrusiveSharedPtr, ConstructionFailureUnwindsMembers) {
  std::atomic<int> destroyed{0};
  EXPECT_THROW(
      static_cast<void>(util::IntrusiveSharedPtr<util::RefCounted<Throws>>::make(destroyed)),
      std::runtime_error);
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RefCounted, ForwardsMoveOnlyValuesAndValueInitializesScalars) {
  using Pointer = util::IntrusiveSharedPtr<util::RefCounted<std::unique_ptr<int>>>;
  auto value = Pointer::make(std::make_unique<int>(17));
  EXPECT_EQ(*value->value, 17);
  auto zero = IntPtr::make();
  EXPECT_EQ(zero->value, 0);
}

template <typename T> void expect_isolated(const util::RefCountedCacheAligned<T> &object) {
  constexpr auto size = util::detail::kDestructiveInterferenceSize;
  const auto counter = reinterpret_cast<std::uintptr_t>(&object.ref_count());
  const auto value = reinterpret_cast<std::uintptr_t>(&object.value);
  EXPECT_EQ(value % size, 0u);
  EXPECT_GE(value, counter + sizeof(object.ref_count()));
  EXPECT_NE(counter / size, value / size);
}

struct alignas(256) OverAligned {
  int value = 0;
};

static_assert(alignof(util::RefCountedCacheAligned<OverAligned>) >= alignof(OverAligned));
static_assert(alignof(util::RefCountedCacheAligned<int>) >=
              util::detail::kDestructiveInterferenceSize);

TEST(RefCounted, AlignedStorageSeparatesCounterAndValue) {
  using Pointer = util::IntrusiveSharedPtr<util::RefCountedCacheAligned<int>>;
  auto value = Pointer::make(42);
  EXPECT_EQ(value->value, 42);
  expect_isolated(*value);

  util::RefCountedCacheAligned<int> array[3];
  for (const auto &element : array)
    expect_isolated(element);

  auto over_aligned = util::IntrusiveSharedPtr<util::RefCountedCacheAligned<OverAligned>>::make();
  expect_isolated(*over_aligned);
  EXPECT_EQ(reinterpret_cast<std::uintptr_t>(&over_aligned->value) % alignof(OverAligned), 0u);
}

TEST(RefCounted, AlignedStorageCanHoldAtomicEpoch) {
  using Pointer = util::IntrusiveSharedPtr<util::RefCountedCacheAligned<std::atomic<uint64_t>>>;
  auto epoch = Pointer::make(1);
  auto retained = epoch;
  epoch->value.fetch_add(1, std::memory_order_release);
  epoch.reset();
  EXPECT_EQ(retained->value.load(std::memory_order_acquire), 2u);
  expect_isolated(*retained);
}

} // namespace
