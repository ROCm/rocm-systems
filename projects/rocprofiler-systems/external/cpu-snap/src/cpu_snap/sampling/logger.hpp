#pragma once

/**
 * @file logger.hpp
 * @brief Injected log sink: the library never writes to stderr itself.
 *
 * The host passes a log_sink in sampler::config::logger. An empty sink turns
 * logging off at no cost.
 *
 * Clock: log_record::timestamp_ns is read from sampler::config::clock (default
 * CLOCK_MONOTONIC), the clock of sample and module-record timestamps, so log
 * lines line up with the sample timeline. The host may convert it.
 *
 * Threading: the sink is called from the thread that constructs, starts and
 * stops the sampler and from the sampler worker thread. It is NEVER called
 * from a signal handler or from the per-sample path. The sink must be
 * thread-safe and must not call back into the sampler: a record emitted from
 * the worker thread (thread rescan) is delivered while the sampler holds an
 * internal lock, so a sink that reads sampler statistics would deadlock. An
 * exception thrown by the sink is swallowed.
 */

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <type_traits>

#include "cpu_snap/sampling/clock_kind.hpp"
#include "cpu_snap/sampling/time.hpp"

namespace cpu_snap::sampling
{

/** @brief Severity of a log_record. */
enum class log_level : std::uint8_t
{
    debug,
    info,
    warn,
    error,
};

/** @brief One log message. message is valid only for the duration of the call. */
struct log_record
{
    log_level        level;
    /** Nanoseconds of sampler::config::clock at the moment the library emitted the record. */
    std::uint64_t    timestamp_ns;
    std::string_view message;
};

/** @brief Host-supplied receiver of log_record; see the file comment for the contract. */
using log_sink = std::function<void(const log_record&)>;

namespace internal
{

/** @brief Deliver @p message to @p sink stamped with @p clock; no-op when empty, swallows sink exceptions. */
inline void
emit(const log_sink& sink, clock_kind clock, log_level level, std::string_view message) noexcept
{
    if(!sink) return;
    try
    {
        sink(log_record{ level, utils::clock_now_ns(clock), message });
    } catch(...)  // NOLINT(bugprone-empty-catch): a faulty sink must not break sampling
    {
    }
}

/** @brief Like emit(), but builds the message with @p make (returns std::string) only when a sink is set. */
template <typename MessageFn>
    requires std::is_invocable_r_v<std::string, MessageFn&>
inline void
emit(const log_sink& sink, clock_kind clock, log_level level, MessageFn&& make) noexcept
{
    if(!sink) return;
    try
    {
        const std::string message = make();
        sink(log_record{ level, utils::clock_now_ns(clock), message });
    } catch(...)  // NOLINT(bugprone-empty-catch): a faulty sink must not break sampling
    {
    }
}

}  // namespace internal

}  // namespace cpu_snap::sampling
