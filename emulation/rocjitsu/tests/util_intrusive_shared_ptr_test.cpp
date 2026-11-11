// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "util/intrusive_shared_ptr.h"

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <cstdint>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace util {

class IntrusiveSharedPtrTestAccess {
public:
  template <typename T>
  static const std::atomic<uint32_t> &ref_count_storage(const RefCountedCacheAligned<T> &object) {
    return object.refs_;
  }
};

} // namespace util

namespace {

class Tracked {
public:
  explicit Tracked(std::atomic<int> &destroyed) : destroyed(destroyed) {}
  ~Tracked() { destroyed.fetch_add(1, std::memory_order_relaxed); }
  std::atomic<int> &destroyed;
};

using TrackedPtr = util::IntrusiveSharedPtr<util::RefCounted<Tracked>>;
using IntPtr = util::IntrusiveSharedPtr<util::RefCounted<int>>;
using ConstIntPtr = util::IntrusiveSharedPtr<const util::RefCounted<int>>;

template <typename T>
concept HasPublicAddRef = requires(const T &object) { object.add_ref(); };
template <typename T>
concept HasPublicReleaseRef = requires(const T &object) { object.release_ref(); };

template <typename T> constexpr bool has_read_only_ownership() {
  static_assert(!HasPublicAddRef<T>);
  static_assert(!HasPublicReleaseRef<T>);
  static_assert(std::is_same_v<decltype(std::declval<T &>().ref_count()), uint32_t>);
  static_assert(std::is_same_v<decltype(std::declval<const T &>().ref_count()), uint32_t>);
  return true;
}

static_assert(has_read_only_ownership<util::RefCounted<int>>());
static_assert(has_read_only_ownership<util::RefCountedCacheAligned<int>>());
static_assert(has_read_only_ownership<util::RefCountedNonAtomic<int>>());
template <typename T>
concept HasPublicCounterStorage = requires(const T &object) { object.ref_count_storage(); };

static_assert(!HasPublicCounterStorage<util::RefCounted<int>>);
static_assert(!HasPublicCounterStorage<util::RefCountedCacheAligned<int>>);
static_assert(!HasPublicCounterStorage<util::RefCountedNonAtomic<int>>);

class CatchAllValue {
public:
  template <typename T> explicit CatchAllValue(T &&) {}
};

template <template <typename> typename Storage>
constexpr bool has_constrained_forwarding_constructor() {
  using IntStorage = Storage<int>;
  using GenericStorage = Storage<CatchAllValue>;
  static_assert(std::is_constructible_v<IntStorage, int>);
  static_assert(!std::is_constructible_v<IntStorage, const char *>);
  static_assert(!std::is_constructible_v<IntStorage, IntStorage &>);
  static_assert(!std::is_constructible_v<IntStorage, const IntStorage &&>);
  static_assert(std::is_constructible_v<GenericStorage, int>);
  static_assert(!std::is_constructible_v<GenericStorage, GenericStorage &>);
  static_assert(!std::is_constructible_v<GenericStorage, const GenericStorage &&>);
  return true;
}

static_assert(has_constrained_forwarding_constructor<util::RefCounted>());
static_assert(has_constrained_forwarding_constructor<util::RefCountedNonAtomic>());
static_assert(has_constrained_forwarding_constructor<util::RefCountedCacheAligned>());

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

TEST(IntrusiveSharedPtr, EqualityComparesPointerIdentity) {
  IntPtr empty;
  ConstIntPtr const_empty;
  auto owner = IntPtr::make(42);
  auto copy = owner;
  ConstIntPtr const_owner(owner);
  auto distinct = IntPtr::make(42);

  EXPECT_EQ(empty, IntPtr{});
  EXPECT_EQ(empty, const_empty);
  EXPECT_EQ(const_empty, empty);
  EXPECT_EQ(owner, copy);
  EXPECT_EQ(owner, const_owner);
  EXPECT_EQ(const_owner, owner);
  EXPECT_NE(owner, distinct);
  EXPECT_NE(const_owner, distinct);
  EXPECT_NE(distinct, const_owner);
  EXPECT_NE(owner, empty);
  EXPECT_NE(const_empty, owner);
  EXPECT_EQ(owner->ref_count(), 3u);
  EXPECT_EQ(distinct->ref_count(), 1u);
}

TEST(IntrusiveSharedPtr, CopiesRetainUntilLastOwner) {
  std::atomic<int> destroyed{0};
  auto owner = TrackedPtr::make(destroyed);
  ASSERT_TRUE(owner);
  EXPECT_EQ(owner->ref_count(), 1u);
  {
    auto copy = owner;
    EXPECT_EQ(copy.get(), owner.get());
    EXPECT_EQ(&*copy, owner.get());
    EXPECT_EQ(owner->ref_count(), 2u);
    owner.reset();
    EXPECT_EQ(destroyed.load(), 0);
    EXPECT_EQ(copy->ref_count(), 1u);
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
    EXPECT_EQ(moved->ref_count(), 1u);
    TrackedPtr destination;
    destination = std::move(moved);
    EXPECT_FALSE(moved);
    EXPECT_EQ(destination.get(), address);
    EXPECT_EQ(destination->ref_count(), 1u);
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
  EXPECT_EQ(owner->ref_count(), 1u);
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
  EXPECT_EQ(owner->ref_count(), 2u);
  ConstIntPtr moved(std::move(owner));
  EXPECT_FALSE(owner);
  EXPECT_EQ(moved.get(), copy.get());
  EXPECT_EQ(moved->value, 42);
  EXPECT_EQ(moved->ref_count(), 2u);
  copy.reset();
  EXPECT_EQ(moved->ref_count(), 1u);

  ConstIntPtr assigned;
  auto replacement = IntPtr::make(17);
  assigned = replacement;
  EXPECT_EQ(assigned->value, 17);
  assigned = std::move(replacement);
  EXPECT_FALSE(replacement);
  EXPECT_EQ(assigned->ref_count(), 1u);
}

class CustomCounted final {
public:
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
  for (int thread_index = 0; thread_index < kThreads; ++thread_index) {
    workers.emplace_back([local = owner, &start]() mutable {
      start.arrive_and_wait();
      for (int iteration = 0; iteration < 2000; ++iteration) {
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

TEST(RefCounted, ForwardsMoveOnlyValuesAndValueInitializesScalars) {
  using Pointer = util::IntrusiveSharedPtr<util::RefCounted<std::unique_ptr<int>>>;
  auto value = Pointer::make(std::make_unique<int>(17));
  EXPECT_EQ(*value->value, 17);
  auto zero = IntPtr::make();
  EXPECT_EQ(zero->value, 0);
}

using NonAtomicIntPtr = util::IntrusiveSharedPtr<util::RefCountedNonAtomic<int>>;
using ConstNonAtomicIntPtr = util::IntrusiveSharedPtr<const util::RefCountedNonAtomic<int>>;

static_assert(!std::is_copy_constructible_v<util::RefCountedNonAtomic<int>>);
static_assert(!std::is_move_constructible_v<util::RefCountedNonAtomic<int>>);
static_assert(!std::is_copy_assignable_v<util::RefCountedNonAtomic<int>>);
static_assert(!std::is_move_assignable_v<util::RefCountedNonAtomic<int>>);
static_assert(
    std::is_same_v<decltype(std::declval<ConstNonAtomicIntPtr>()->ref_count()), uint32_t>);
static_assert(std::is_same_v<decltype((std::declval<ConstNonAtomicIntPtr>()->value)), const int &>);

TEST(RefCountedNonAtomic, CopiesRetainAndMovesTransferUntilFinalRelease) {
  using Pointer = util::IntrusiveSharedPtr<util::RefCountedNonAtomic<Tracked>>;
  std::atomic<int> destroyed{0};
  {
    auto owner = Pointer::make(destroyed);
    EXPECT_EQ(owner->ref_count(), 1u);
    auto copy = owner;
    EXPECT_EQ(owner->ref_count(), 2u);
    auto moved = std::move(copy);
    EXPECT_FALSE(copy);
    EXPECT_EQ(moved.get(), owner.get());
    EXPECT_EQ(moved->ref_count(), 2u);
    owner.reset();
    EXPECT_EQ(moved->ref_count(), 1u);
    EXPECT_EQ(destroyed.load(), 0);
  }
  EXPECT_EQ(destroyed.load(), 1);
}

TEST(RefCountedNonAtomic, ConstConversionRetainsOrTransfersOwnership) {
  auto owner = NonAtomicIntPtr::make(42);
  ConstNonAtomicIntPtr copy = owner;
  EXPECT_EQ(owner->ref_count(), 2u);
  ConstNonAtomicIntPtr moved = std::move(owner);
  EXPECT_FALSE(owner);
  EXPECT_EQ(copy.get(), moved.get());
  EXPECT_EQ(moved->ref_count(), 2u);
  copy.reset();
  EXPECT_EQ(moved->ref_count(), 1u);
  EXPECT_EQ(moved->value, 42);
}

TEST(RefCountedNonAtomic, ForwardsMoveOnlyValuesAndValueInitializesScalars) {
  using Pointer = util::IntrusiveSharedPtr<util::RefCountedNonAtomic<std::unique_ptr<int>>>;
  auto value = Pointer::make(std::make_unique<int>(17));
  EXPECT_EQ(*value->value, 17);
  auto zero = NonAtomicIntPtr::make();
  EXPECT_EQ(zero->value, 0);
}

TEST(RefCountedNonAtomic, DetachedVectorPreservesOtherOwners) {
  using Pointer = util::IntrusiveSharedPtr<util::RefCountedNonAtomic<std::vector<int>>>;
  auto owner = Pointer::make(std::vector<int>{1, 2});
  auto copy = owner;
  EXPECT_EQ(owner->ref_count(), 2u);
  copy = Pointer::make(copy->value);
  copy->value[0] = 3;
  copy->value.push_back(4);
  EXPECT_EQ(owner->value, (std::vector<int>{1, 2}));
  EXPECT_EQ(copy->value, (std::vector<int>{3, 2, 4}));
  EXPECT_EQ(owner->ref_count(), 1u);
  EXPECT_EQ(copy->ref_count(), 1u);
}

template <typename T> void expect_isolated(const util::RefCountedCacheAligned<T> &object) {
  constexpr auto size = util::detail::kDestructiveInterferenceSize;
  const std::atomic<uint32_t> &counter_storage =
      util::IntrusiveSharedPtrTestAccess::ref_count_storage(object);
  const auto counter = reinterpret_cast<std::uintptr_t>(&counter_storage);
  const auto value = reinterpret_cast<std::uintptr_t>(&object.value);
  EXPECT_EQ(value % size, 0u);
  EXPECT_GE(value, counter + sizeof(counter_storage));
  EXPECT_NE(counter / size, value / size);
}

struct alignas(256) OverAligned {
  int value = 0;
};

static_assert(alignof(util::RefCountedCacheAligned<OverAligned>) >= alignof(OverAligned));
static_assert(alignof(util::RefCountedCacheAligned<int>) >=
              util::detail::kDestructiveInterferenceSize);

TEST(RefCountedCacheAligned, AlignedStorageSeparatesCounterAndValue) {
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

TEST(RefCountedCacheAligned, AlignedStorageCanHoldAtomicEpoch) {
  using Pointer = util::IntrusiveSharedPtr<util::RefCountedCacheAligned<std::atomic<uint64_t>>>;
  auto epoch = Pointer::make(1);
  auto retained = epoch;
  epoch->value.fetch_add(1, std::memory_order_release);
  epoch.reset();
  EXPECT_EQ(retained->value.load(std::memory_order_acquire), 2u);
  expect_isolated(*retained);
}

} // namespace
