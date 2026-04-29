// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// C++17 detection-idiom checks for the ten sampling policy named requirements.
//
// Each is_Xxx_policy<T> trait uses std::void_t to verify that T exposes the
// required non-template method signatures.  Template methods (write, fatal) are
// excluded from the void_t check because their signatures are not testable without
// an instantiation context; they are covered by integration tests instead.
//
// Usage: sampling_service<Policies>'s ctor static_asserts each is_X_policy_v.

#include <chrono>
#include <cstdint>
#include <string_view>
#include <type_traits>
#include <vector>

// Forward-declare the minimum domain types needed for signature checks.
namespace rocprofsys::sampling
{
struct backtrace_record;
struct timer_sample;
struct overflow_sample;
}  // namespace rocprofsys::sampling

// ROCPROFSYS_DEFINE_POLICY_TRAIT(name, expr)
//   Generates `is_<name>_policy<T>` via std::void_t<expr> and the matching
//   `is_<name>_policy_v<T>` constexpr alias. `expr` is evaluated against
//   `std::declval<T&>()` (or `T const&` when read-only).
#define ROCPROFSYS_DEFINE_POLICY_TRAIT(name, ...)                                        \
    template <typename T, typename = void>                                               \
    struct is_##name##_policy : std::false_type                                          \
    {};                                                                                  \
    template <typename T>                                                                \
    struct is_##name##_policy<T, std::void_t<__VA_ARGS__>> : std::true_type              \
    {};                                                                                  \
    template <typename T>                                                                \
    inline constexpr bool is_##name##_policy_v = is_##name##_policy<T>::value

namespace rocprofsys::sampling::detail
{

// ClockPolicy — uint64_t now_ns() const noexcept;
//               std::chrono::steady_clock::time_point now_steady() const noexcept;
ROCPROFSYS_DEFINE_POLICY_TRAIT(
    clock, decltype(static_cast<uint64_t>(std::declval<T const&>().now_ns())),
    decltype(static_cast<std::chrono::steady_clock::time_point>(
        std::declval<T const&>().now_steady())));

// TimerTriggerPolicy — start()/stop() noexcept; bool is_armed() const noexcept.
//   configure() has platform-specific params (clockid_t); not checked here.
ROCPROFSYS_DEFINE_POLICY_TRAIT(
    timer_trigger, decltype(std::declval<T&>().start()),
    decltype(std::declval<T&>().stop()),
    decltype(static_cast<bool>(std::declval<T const&>().is_armed())));

// OverflowTriggerPolicy — start()/stop() noexcept; bool is_open() const noexcept.
//   configure() takes perf_event_attr& (platform type); not checked here.
ROCPROFSYS_DEFINE_POLICY_TRAIT(
    overflow_trigger, decltype(std::declval<T&>().start()),
    decltype(std::declval<T&>().stop()),
    decltype(static_cast<bool>(std::declval<T const&>().is_open())));

// SignalDispatcherPolicy — int apply_sigmask(int, void const*, void*) noexcept.
//   The method is named apply_sigmask so it does NOT collide with the POSIX
//   `sigmask(sig)` 1-arg macro from <signal.h> — no #undef workaround needed.
ROCPROFSYS_DEFINE_POLICY_TRAIT(signal_dispatcher,
                               decltype(static_cast<int>(std::declval<T&>().apply_sigmask(
                                   0, static_cast<void const*>(nullptr),
                                   static_cast<void*>(nullptr)))));

// UnwinderPolicy — auto unwind(void const*) noexcept; static bool valid_pc(uintptr_t).
//   Return type of unwind() is not checked — only method existence.
ROCPROFSYS_DEFINE_POLICY_TRAIT(
    unwinder, decltype(std::declval<T&>().unwind(static_cast<void const*>(nullptr))),
    decltype(static_cast<bool>(T::valid_pc(static_cast<uintptr_t>(0)))));

// EmitterPolicy (Offload) — std::vector<backtrace_record> read(int64_t);
//   std::vector<int64_t> tids() const; void reset()/erase(int64_t) noexcept.
//   write() is a template method; not checkable via void_t at definition time.
ROCPROFSYS_DEFINE_POLICY_TRAIT(emitter, decltype(std::declval<T&>().read(int64_t{})),
                               decltype(std::declval<T const&>().tids()),
                               decltype(std::declval<T&>().reset()),
                               decltype(std::declval<T&>().erase(int64_t{})));

// TraceSinkPolicy — store_timer / store_overflow with vector args.
ROCPROFSYS_DEFINE_POLICY_TRAIT(
    trace_sink,
    decltype(std::declval<T&>().store_timer(
        int64_t{}, std::declval<std::vector<timer_sample> const&>())),
    decltype(std::declval<T&>().store_overflow(
        int64_t{}, std::declval<std::vector<overflow_sample> const&>())));

// PerfettoSinkPolicy — emit_timer / emit_overflow with vector args.
ROCPROFSYS_DEFINE_POLICY_TRAIT(perfetto_sink,
                               decltype(std::declval<T&>().emit_timer(
                                   int64_t{}, static_cast<void const*>(nullptr),
                                   std::declval<std::vector<timer_sample> const&>())),
                               decltype(std::declval<T&>().emit_overflow(
                                   int64_t{}, static_cast<void const*>(nullptr),
                                   std::declval<std::vector<overflow_sample> const&>())));

// ReportWriterPolicy — write_timer_samples / write_overflow_samples / flush.
ROCPROFSYS_DEFINE_POLICY_TRAIT(
    report_writer,
    decltype(std::declval<T&>().write_timer_samples(
        int64_t{}, std::declval<std::vector<timer_sample> const&>())),
    decltype(std::declval<T&>().write_overflow_samples(
        int64_t{}, std::declval<std::vector<overflow_sample> const&>())),
    decltype(std::declval<T&>().flush()));

// FatalErrorPolicy — template <class... Args> [[noreturn]] void fatal(...) noexcept.
// fatal() is a variadic template; not testable via void_t at definition time.
// Verify a single-arg instantiation exists by probing T::template fatal<int>.
template <typename T, typename = void>
struct is_fatal_error_policy : std::false_type
{};

template <typename T>
struct is_fatal_error_policy<
    T, std::void_t<decltype(std::declval<T&>().template fatal<int>(
           static_cast<char const*>(nullptr), 0, std::declval<std::string_view>(),
           std::declval<int const&>()))>> : std::true_type
{};

template <typename T>
inline constexpr bool is_fatal_error_policy_v = is_fatal_error_policy<T>::value;

}  // namespace rocprofsys::sampling::detail

#undef ROCPROFSYS_DEFINE_POLICY_TRAIT
