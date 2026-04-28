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
// Usage: sampling_policies_traits<...> has a static check() that fires
// static_assert for each policy slot.

#include <chrono>
#include <cstdint>
#include <type_traits>
#include <vector>

// Forward-declare the minimum domain types needed for signature checks.
namespace rocprofsys::sampling
{
struct backtrace_record;
struct timer_sample;
struct overflow_sample;
}  // namespace rocprofsys::sampling

namespace rocprofsys::sampling::detail
{

// ── ClockPolicy ──────────────────────────────────────────────────────────────
// Required: uint64_t now_ns() [const] noexcept
//           std::chrono::steady_clock::time_point now_steady() [const] noexcept

template <typename T, typename = void>
struct is_clock_policy : std::false_type
{};

template <typename T>
struct is_clock_policy<
    T, std::void_t<decltype(static_cast<uint64_t>(std::declval<T const&>().now_ns())),
                   decltype(static_cast<std::chrono::steady_clock::time_point>(
                       std::declval<T const&>().now_steady()))>> : std::true_type
{};

template <typename T>
inline constexpr bool is_clock_policy_v = is_clock_policy<T>::value;

// ── TimerTriggerPolicy ───────────────────────────────────────────────────────
// Required: void start() noexcept
//           void stop() noexcept
//           bool is_armed() const noexcept
// Note: configure() has platform-specific params (clockid_t); not checked here.

template <typename T, typename = void>
struct is_timer_trigger_policy : std::false_type
{};

template <typename T>
struct is_timer_trigger_policy<
    T,
    std::void_t<decltype(std::declval<T&>().start()), decltype(std::declval<T&>().stop()),
                decltype(static_cast<bool>(std::declval<T const&>().is_armed()))>>
: std::true_type
{};

template <typename T>
inline constexpr bool is_timer_trigger_policy_v = is_timer_trigger_policy<T>::value;

// ── OverflowTriggerPolicy ────────────────────────────────────────────────────
// Required: void start() noexcept
//           void stop() noexcept
//           bool is_open() const noexcept
// Note: configure() takes perf_event_attr& (platform type); not checked here.

template <typename T, typename = void>
struct is_overflow_trigger_policy : std::false_type
{};

template <typename T>
struct is_overflow_trigger_policy<
    T,
    std::void_t<decltype(std::declval<T&>().start()), decltype(std::declval<T&>().stop()),
                decltype(static_cast<bool>(std::declval<T const&>().is_open()))>>
: std::true_type
{};

template <typename T>
inline constexpr bool is_overflow_trigger_policy_v = is_overflow_trigger_policy<T>::value;

// ── SignalDispatcherPolicy ───────────────────────────────────────────────────
// Required: int sigmask(int, void const*, void*) noexcept
// Note: production uses sigset_t*; mock uses void* — both satisfy the check.
// POSIX defines sigmask(sig) as a 1-arg macro via <signal.h>; undefine it so
// the 3-arg decltype expression below is not mangled by the preprocessor.
#ifdef sigmask
#    undef sigmask
#endif

template <typename T, typename = void>
struct is_signal_dispatcher_policy : std::false_type
{};

template <typename T>
struct is_signal_dispatcher_policy<
    T, std::void_t<decltype(static_cast<int>(std::declval<T&>().sigmask(
           0, static_cast<void const*>(nullptr), static_cast<void*>(nullptr))))>>
: std::true_type
{};

template <typename T>
inline constexpr bool is_signal_dispatcher_policy_v =
    is_signal_dispatcher_policy<T>::value;

// ── UnwinderPolicy ───────────────────────────────────────────────────────────
// Required: auto unwind(void const*) noexcept  (returns vector-like; ctx is void*)
//           static bool valid_pc(uintptr_t) noexcept
// Note: the return type of unwind() is not checked — only method existence.

template <typename T, typename = void>
struct is_unwinder_policy : std::false_type
{};

template <typename T>
struct is_unwinder_policy<
    T, std::void_t<decltype(std::declval<T&>().unwind(static_cast<void const*>(nullptr))),
                   decltype(static_cast<bool>(T::valid_pc(static_cast<uintptr_t>(0))))>>
: std::true_type
{};

template <typename T>
inline constexpr bool is_unwinder_policy_v = is_unwinder_policy<T>::value;

// ── EmitterPolicy (Offload) ───────────────────────────────────────────────────
// Required non-template: std::vector<backtrace_record> read(int64_t)
//                        std::vector<int64_t> tids() const
//                        void reset() noexcept
//                        void erase(int64_t) noexcept
// Note: write() is a template method; not checkable via void_t at definition time.

template <typename T, typename = void>
struct is_emitter_policy : std::false_type
{};

template <typename T>
struct is_emitter_policy<T, std::void_t<decltype(std::declval<T&>().read(int64_t{})),
                                        decltype(std::declval<T const&>().tids()),
                                        decltype(std::declval<T&>().reset()),
                                        decltype(std::declval<T&>().erase(int64_t{}))>>
: std::true_type
{};

template <typename T>
inline constexpr bool is_emitter_policy_v = is_emitter_policy<T>::value;

// ── TraceSinkPolicy ───────────────────────────────────────────────────────────
// Required: void store_timer(int64_t, std::vector<timer_sample> const&)
//           void store_overflow(int64_t, std::vector<overflow_sample> const&)

template <typename T, typename = void>
struct is_trace_sink_policy : std::false_type
{};

template <typename T>
struct is_trace_sink_policy<
    T, std::void_t<decltype(std::declval<T&>().store_timer(
                       int64_t{}, std::declval<std::vector<timer_sample> const&>())),
                   decltype(std::declval<T&>().store_overflow(
                       int64_t{}, std::declval<std::vector<overflow_sample> const&>()))>>
: std::true_type
{};

template <typename T>
inline constexpr bool is_trace_sink_policy_v = is_trace_sink_policy<T>::value;

// ── PerfettoSinkPolicy ────────────────────────────────────────────────────────
// Required: void emit_timer(int64_t, void const*, std::vector<timer_sample> const&)
//           void emit_overflow(int64_t, void const*, std::vector<overflow_sample> const&)

template <typename T, typename = void>
struct is_perfetto_sink_policy : std::false_type
{};

template <typename T>
struct is_perfetto_sink_policy<
    T, std::void_t<decltype(std::declval<T&>().emit_timer(
                       int64_t{}, static_cast<void const*>(nullptr),
                       std::declval<std::vector<timer_sample> const&>())),
                   decltype(std::declval<T&>().emit_overflow(
                       int64_t{}, static_cast<void const*>(nullptr),
                       std::declval<std::vector<overflow_sample> const&>()))>>
: std::true_type
{};

template <typename T>
inline constexpr bool is_perfetto_sink_policy_v = is_perfetto_sink_policy<T>::value;

// ── ReportWriterPolicy ────────────────────────────────────────────────────────
// Required: void write_timer_samples(int64_t, std::vector<timer_sample> const&)
//           void write_overflow_samples(int64_t, std::vector<overflow_sample> const&)
//           void flush()

template <typename T, typename = void>
struct is_report_writer_policy : std::false_type
{};

template <typename T>
struct is_report_writer_policy<
    T, std::void_t<decltype(std::declval<T&>().write_timer_samples(
                       int64_t{}, std::declval<std::vector<timer_sample> const&>())),
                   decltype(std::declval<T&>().write_overflow_samples(
                       int64_t{}, std::declval<std::vector<overflow_sample> const&>())),
                   decltype(std::declval<T&>().flush())>> : std::true_type
{};

template <typename T>
inline constexpr bool is_report_writer_policy_v = is_report_writer_policy<T>::value;

// ── FatalErrorPolicy ──────────────────────────────────────────────────────────
// Required: template <class... Args> [[noreturn]] void fatal(char const*, int,
//               std::string_view fmt, Args const&...)
// fatal() is a variadic template; not testable via void_t at definition time.
// A proxy check: verify the type is not void (weakest possible guard).
// The integration tests cover real conformance.
// NOTE: If you have a better way to detect variadic template methods in C++17,
// extend this trait.

template <typename T>
struct is_fatal_error_policy : std::bool_constant<!std::is_void_v<T>>
{};

template <typename T>
inline constexpr bool is_fatal_error_policy_v = is_fatal_error_policy<T>::value;

}  // namespace rocprofsys::sampling::detail
