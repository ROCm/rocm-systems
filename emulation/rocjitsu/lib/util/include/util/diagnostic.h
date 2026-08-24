// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "util/result.h"

#include <cassert>
#include <concepts>
#include <functional>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace util {

/// A non-owning destination for one completed diagnostic message.
///
/// The referenced callable must outlive the sink and every diagnostic created
/// from it. It must be a non-volatile object; wrap a free function in an object
/// when needed. A default-constructed sink intentionally discards messages.
/// Explicit emission propagates callback failures, while destructor-triggered
/// emission suppresses them.
class DiagnosticSink {
public:
  DiagnosticSink() = default;

  template <typename Callable>
    requires(!std::same_as<std::remove_cvref_t<Callable>, DiagnosticSink> &&
             std::is_object_v<Callable> && !std::is_volatile_v<Callable> &&
             std::invocable<Callable &, std::string_view>)
  explicit DiagnosticSink(Callable &callable)
      : context_(std::addressof(callable)),
        callback_([](const void *context, std::string_view message) {
          using Object = std::remove_const_t<Callable>;
          if constexpr (std::is_const_v<Callable>)
            std::invoke(*static_cast<const Object *>(context), message);
          else
            std::invoke(*const_cast<Object *>(static_cast<const Object *>(context)), message);
        }) {}

  void emit(std::string_view message) const {
    if (callback_ != nullptr)
      callback_(context_, message);
  }

  [[nodiscard]] bool ignores_messages() const { return callback_ == nullptr; }

private:
  using Callback = void (*)(const void *, std::string_view);

  const void *context_ = nullptr;
  Callback callback_ = nullptr;
};

/// A move-only error builder that commits its message at most once.
/// Non-fatal diagnostics and structured severity are intentionally outside this API.
class InFlightDiagnostic {
public:
  explicit InFlightDiagnostic(DiagnosticSink sink) : sink_(sink) {
    if (!sink_.ignores_messages())
      stream_.emplace();
  }

  InFlightDiagnostic(const InFlightDiagnostic &) = delete;
  InFlightDiagnostic &operator=(const InFlightDiagnostic &) = delete;

  InFlightDiagnostic(InFlightDiagnostic &&other) noexcept
      : sink_(other.sink_), stream_(std::move(other.stream_)), active_(other.active_) {
    other.active_ = false;
  }

  InFlightDiagnostic &operator=(InFlightDiagnostic &&other) noexcept {
    if (this == &other)
      return *this;
    emit_from_destructor();
    sink_ = other.sink_;
    stream_ = std::move(other.stream_);
    active_ = other.active_;
    other.active_ = false;
    return *this;
  }

  ~InFlightDiagnostic() { emit_from_destructor(); }

  template <typename T> InFlightDiagnostic &operator<<(T &&fragment) {
    assert(active_ && "cannot append to an emitted diagnostic");
    if (stream_)
      *stream_ << std::forward<T>(fragment);
    return *this;
  }

  InFlightDiagnostic &operator<<(std::ostream &(*manipulator)(std::ostream &)) {
    assert(active_ && "cannot append to an emitted diagnostic");
    if (stream_)
      *stream_ << manipulator;
    return *this;
  }

  /// Commit the diagnostic and return failure for compact propagation.
  Result emit() {
    if (active_) {
      active_ = false;
      if (stream_)
        sink_.emit(stream_->str());
    }
    return Result::failure();
  }

  operator Result() { return emit(); }

  template <typename T> operator FailureOr<T>() {
    (void)emit();
    return Result::failure();
  }

private:
  void emit_from_destructor() noexcept {
    if (!active_)
      return;
    try {
      (void)emit();
    } catch (...) {
      // A destructor cannot safely surface diagnostic delivery failures.
      active_ = false;
    }
  }

  DiagnosticSink sink_;
  std::optional<std::ostringstream> stream_;
  bool active_ = true;
};

/// A cheap, copyable producer for in-flight diagnostics.
class DiagnosticEmitter {
public:
  DiagnosticEmitter() = default;
  explicit DiagnosticEmitter(DiagnosticSink sink) : sink_(sink) {}

  template <typename Callable>
    requires(!std::same_as<std::remove_cvref_t<Callable>, DiagnosticEmitter> &&
             std::is_object_v<Callable> && !std::is_volatile_v<Callable> &&
             std::invocable<Callable &, std::string_view>)
  explicit DiagnosticEmitter(Callable &callable) : sink_(callable) {}

  [[nodiscard]] InFlightDiagnostic emit() const { return InFlightDiagnostic(sink_); }
  [[nodiscard]] bool ignores_messages() const { return sink_.ignores_messages(); }

private:
  DiagnosticSink sink_;
};

/// Owns the latest diagnostic message and provides an emitter that writes it.
/// This single-threaded helper is pinned because its emitters borrow it.
class StringDiagnostic {
public:
  StringDiagnostic() = default;
  StringDiagnostic(const StringDiagnostic &) = delete;
  StringDiagnostic(StringDiagnostic &&) = delete;
  StringDiagnostic &operator=(const StringDiagnostic &) = delete;
  StringDiagnostic &operator=(StringDiagnostic &&) = delete;

  void operator()(std::string_view message) { message_.assign(message); }

  [[nodiscard]] DiagnosticEmitter emitter() & { return DiagnosticEmitter(*this); }
  DiagnosticEmitter emitter() && = delete;
  [[nodiscard]] const std::string &message() const { return message_; }

private:
  std::string message_;
};

} // namespace util
