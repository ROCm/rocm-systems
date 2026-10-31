// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace util {

/// @brief Strong-only shared ownership using object-provided ownership hooks.
/// @details T must provide noexcept add_ref() and release_ref() hooks accessible
/// to this class. make() adopts the initial reference of the newly constructed
/// object; release_ref() handles destruction on the final release. The hooks
/// determine thread safety. Concurrent mutation of the same handle requires
/// external synchronization. Does not provide weak ownership.
template <typename T> class IntrusiveSharedPtr {
  template <typename> friend class IntrusiveSharedPtr;

public:
  IntrusiveSharedPtr() noexcept = default;
  IntrusiveSharedPtr(std::nullptr_t) noexcept {}

  template <typename... Args> [[nodiscard]] static IntrusiveSharedPtr make(Args &&...args) {
    return IntrusiveSharedPtr(new T(std::forward<Args>(args)...));
  }

  IntrusiveSharedPtr(const IntrusiveSharedPtr &other) noexcept : ptr_(other.ptr_) { retain(); }
  IntrusiveSharedPtr(IntrusiveSharedPtr &&other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)) {}

  // Allow adding const without permitting base/derived conversions whose deletion
  // and counter lookup could differ from the originally allocated type.
  template <typename U>
    requires(std::is_same_v<T, const U> && !std::is_const_v<U>)
  IntrusiveSharedPtr(const IntrusiveSharedPtr<U> &other) noexcept : ptr_(other.ptr_) {
    retain();
  }
  template <typename U>
    requires(std::is_same_v<T, const U> && !std::is_const_v<U>)
  IntrusiveSharedPtr(IntrusiveSharedPtr<U> &&other) noexcept
      : ptr_(std::exchange(other.ptr_, nullptr)) {}

  IntrusiveSharedPtr &operator=(IntrusiveSharedPtr other) noexcept {
    swap(other);
    return *this;
  }

  ~IntrusiveSharedPtr() {
    if (ptr_)
      ptr_->release_ref();
  }

  [[nodiscard]] T *get() const noexcept { return ptr_; }
  [[nodiscard]] T *operator->() const noexcept { return ptr_; }
  [[nodiscard]] T &operator*() const noexcept { return *ptr_; }
  [[nodiscard]] explicit operator bool() const noexcept { return ptr_ != nullptr; }

  void reset() noexcept { IntrusiveSharedPtr{}.swap(*this); }
  void swap(IntrusiveSharedPtr &other) noexcept { std::swap(ptr_, other.ptr_); }
  friend void swap(IntrusiveSharedPtr &a, IntrusiveSharedPtr &b) noexcept { a.swap(b); }

private:
  explicit IntrusiveSharedPtr(T *ptr) noexcept : ptr_(ptr) {}

  void retain() noexcept {
    if (ptr_)
      ptr_->add_ref();
  }

  T *ptr_ = nullptr;
};

namespace detail {

template <typename Derived> class ReferenceCount {
public:
  void add_ref() const noexcept { refs_.fetch_add(1, std::memory_order_relaxed); }
  void release_ref() const noexcept {
    if (refs_.fetch_sub(1, std::memory_order_acq_rel) == 1)
      delete static_cast<const Derived *>(this);
  }

  [[nodiscard]] std::atomic<uint32_t> &ref_count() const noexcept { return refs_; }

private:
  mutable std::atomic<uint32_t> refs_{1};
};

#ifdef __cpp_lib_hardware_interference_size
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
// These internal storage types use consistent target settings across translation
// units; their layout is not a stable ABI across different builds.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winterference-size"
#endif
inline constexpr std::size_t kDestructiveInterferenceSize =
    std::hardware_destructive_interference_size;
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 12
#pragma GCC diagnostic pop
#endif
#else
// Some Clang/standard-library combinations omit the interference-size constant.
// Use 128 bytes to also cover hosts with cache lines larger than x86's 64 bytes.
inline constexpr std::size_t kDestructiveInterferenceSize = 128;
#endif

} // namespace detail

/// @brief Compact atomic counter and value storage for IntrusiveSharedPtr.
/// @details This is the owned object, not a pointer. Access the wrapped T via value.
template <typename T> struct RefCounted final : detail::ReferenceCount<RefCounted<T>> {
  template <typename... Args>
  explicit RefCounted(Args &&...args) : value(std::forward<Args>(args)...) {}

  RefCounted(const RefCounted &) = delete;
  RefCounted(RefCounted &&) = delete;
  RefCounted &operator=(const RefCounted &) = delete;
  RefCounted &operator=(RefCounted &&) = delete;

  T value;
};

/// @brief Compact non-atomic counter and value storage for IntrusiveSharedPtr.
/// @details All ownership operations, including handle destruction and count
/// inspection, must be confined to one thread or externally synchronized across
/// every handle sharing the object. Access the wrapped T via value.
template <typename T> struct RefCountedNonAtomic final {
  template <typename... Args>
  explicit RefCountedNonAtomic(Args &&...args) : value(std::forward<Args>(args)...) {}

  RefCountedNonAtomic(const RefCountedNonAtomic &) = delete;
  RefCountedNonAtomic(RefCountedNonAtomic &&) = delete;
  RefCountedNonAtomic &operator=(const RefCountedNonAtomic &) = delete;
  RefCountedNonAtomic &operator=(RefCountedNonAtomic &&) = delete;

  void add_ref() const noexcept { ++refs_; }
  void release_ref() const noexcept {
    if (--refs_ == 0)
      delete this;
  }

  [[nodiscard]] uint32_t ref_count() const noexcept { return refs_; }

  T value;

private:
  mutable uint32_t refs_{1};
};

/// @brief Counter and value storage with the value isolated from the counter.
/// @details Padding separates the value from the counter by the compile-time
/// destructive interference size (128 bytes if the standard constant is absent).
/// Stronger alignment required by T is preserved. All translation units sharing
/// this type must use consistent target/compiler settings so its layout agrees.
template <typename T>
struct RefCountedCacheAligned final : detail::ReferenceCount<RefCountedCacheAligned<T>> {
  template <typename... Args>
  explicit RefCountedCacheAligned(Args &&...args) : value(std::forward<Args>(args)...) {}

  RefCountedCacheAligned(const RefCountedCacheAligned &) = delete;
  RefCountedCacheAligned(RefCountedCacheAligned &&) = delete;
  RefCountedCacheAligned &operator=(const RefCountedCacheAligned &) = delete;
  RefCountedCacheAligned &operator=(RefCountedCacheAligned &&) = delete;

  alignas(detail::kDestructiveInterferenceSize) alignas(T) T value;
};

} // namespace util
