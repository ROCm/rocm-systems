// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cassert>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace util {

/// A success/failure result for operations that do not return a value.
class [[nodiscard]] Result {
public:
  constexpr Result() noexcept = default;

  static constexpr Result success() noexcept {
    Result result;
    result.succeeded_ = true;
    return result;
  }
  static constexpr Result failure() noexcept { return {}; }

  [[nodiscard]] constexpr bool succeeded() const noexcept { return succeeded_; }
  [[nodiscard]] constexpr bool failed() const noexcept { return !succeeded_; }

private:
  bool succeeded_ = false;
};

/// Customization point for value types with a reserved failure representation.
/// Niche specializations provide failure_value() and is_failure(). The niche must
/// be safe to copy, move, assign, and destroy as a T. Program-defined
/// specializations must be declared with T and be visible before every use of
/// FailureOr<T> so that all translation units select the same storage.
template <typename T> struct failure_or_storage_traits {
  static constexpr bool has_niche = false;
};

/// Null is the canonical pointer failure representation.
template <typename T> struct failure_or_storage_traits<T *> {
  static constexpr bool has_niche = true;

  [[nodiscard]] static constexpr T *failure_value() noexcept { return nullptr; }

  [[nodiscard]] static constexpr bool is_failure(T *value) noexcept { return value == nullptr; }
};

/// A unique pointer with raw-pointer storage can use the null niche while retaining value
/// references. Fancy pointers use the generic optional-backed storage.
template <typename T, typename Deleter>
  requires(std::is_same_v<typename std::unique_ptr<T, Deleter>::pointer,
                          typename std::unique_ptr<T, Deleter>::element_type *> &&
           failure_or_storage_traits<typename std::unique_ptr<T, Deleter>::pointer>::has_niche &&
           std::is_constructible_v<std::unique_ptr<T, Deleter>,
                                   typename std::unique_ptr<T, Deleter>::pointer>)
struct failure_or_storage_traits<std::unique_ptr<T, Deleter>> {
  using Value = std::unique_ptr<T, Deleter>;
  using Pointer = typename Value::pointer;

  static constexpr bool has_niche = true;

  [[nodiscard]] static constexpr Value failure_value() noexcept { return Value(Pointer{}); }

  [[nodiscard]] static constexpr bool is_failure(const Value &value) noexcept {
    return failure_or_storage_traits<Pointer>::is_failure(value.get());
  }
};

/// Storage for FailureOr. Specialize failure_or_storage_traits to use a niche.
template <typename T, bool HasNiche = failure_or_storage_traits<T>::has_niche>
class FailureOrStorage {
public:
  constexpr FailureOrStorage() noexcept = default;

  template <typename U>
    requires std::is_constructible_v<T, U &&>
  constexpr FailureOrStorage(U &&value) noexcept : value_(std::in_place, std::forward<U>(value)) {}

  FailureOrStorage(const FailureOrStorage &) noexcept = default;
  FailureOrStorage(FailureOrStorage &&) noexcept = default;
  FailureOrStorage &operator=(const FailureOrStorage &) noexcept = default;
  FailureOrStorage &operator=(FailureOrStorage &&) noexcept = default;
  constexpr ~FailureOrStorage() noexcept = default;

  [[nodiscard]] constexpr bool has_value() const noexcept { return value_.has_value(); }

  [[nodiscard]] constexpr T &value() & noexcept { return *value_; }
  [[nodiscard]] constexpr const T &value() const & noexcept { return *value_; }
  [[nodiscard]] constexpr T &&value() && noexcept { return std::move(*value_); }

private:
  std::optional<T> value_;
};

/// Niche-backed storage that still contains a T and can therefore return references.
template <typename T> class FailureOrStorage<T, true> {
  using Traits = failure_or_storage_traits<T>;

public:
  constexpr FailureOrStorage() noexcept : value_(Traits::failure_value()) {}

  template <typename U>
    requires std::is_constructible_v<T, U &&>
  constexpr FailureOrStorage(U &&value) noexcept : value_(std::forward<U>(value)) {
    assert(!Traits::is_failure(value_) &&
           "cannot construct a successful FailureOr from its failure representation");
  }

  FailureOrStorage(const FailureOrStorage &) noexcept = default;
  FailureOrStorage(FailureOrStorage &&) noexcept = default;
  FailureOrStorage &operator=(const FailureOrStorage &) noexcept = default;
  FailureOrStorage &operator=(FailureOrStorage &&) noexcept = default;
  constexpr ~FailureOrStorage() noexcept = default;

  [[nodiscard]] constexpr bool has_value() const noexcept { return !Traits::is_failure(value_); }

  [[nodiscard]] constexpr T &value() & noexcept { return value_; }
  [[nodiscard]] constexpr const T &value() const & noexcept { return value_; }
  [[nodiscard]] constexpr T &&value() && noexcept { return std::move(value_); }

private:
  T value_;
};

/// A result that contains either a value or failure. A niche's failure
/// representation is not a valid success value. After a move, the source is
/// valid but whether it contains a value is unspecified.
/// Every operation is noexcept; an exception from a value operation terminates the process.
template <typename T> class [[nodiscard]] FailureOr {
public:
  /// Construct a failure; the supplied Result must itself represent failure.
  constexpr FailureOr([[maybe_unused]] Result result) noexcept {
    assert(result.failed() && "cannot construct a FailureOr from a successful Result");
  }

  template <typename U = T>
    requires(!std::is_same_v<std::remove_cvref_t<U>, FailureOr> &&
             !std::is_same_v<std::remove_cvref_t<U>, Result> && std::is_constructible_v<T, U &&>)
  constexpr FailureOr(U &&value) noexcept : storage_(std::forward<U>(value)) {}

  FailureOr(const FailureOr &) noexcept = default;
  FailureOr(FailureOr &&) noexcept = default;
  FailureOr &operator=(const FailureOr &) noexcept = default;
  FailureOr &operator=(FailureOr &&) noexcept = default;
  constexpr ~FailureOr() noexcept = default;

  [[nodiscard]] constexpr bool succeeded() const noexcept { return storage_.has_value(); }
  [[nodiscard]] constexpr bool failed() const noexcept { return !storage_.has_value(); }

  [[nodiscard]] constexpr T &value() & noexcept {
    assert(succeeded() && "cannot access the value of a failed FailureOr");
    return storage_.value();
  }
  [[nodiscard]] constexpr const T &value() const & noexcept {
    assert(succeeded() && "cannot access the value of a failed FailureOr");
    return storage_.value();
  }
  [[nodiscard]] constexpr T &&value() && noexcept {
    assert(succeeded() && "cannot access the value of a failed FailureOr");
    return std::move(storage_).value();
  }

private:
  FailureOrStorage<T> storage_;
};

template <typename T> FailureOr(T) -> FailureOr<T>;

} // namespace util
