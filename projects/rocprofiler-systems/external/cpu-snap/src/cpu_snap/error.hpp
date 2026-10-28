#pragma once

#include <cstring>
#include <stdexcept>
#include <string>
#include <system_error>

namespace cpu_snap
{

/**
 * Base of every exception cpu-snap throws for library-originated failures.
 * Thrown only from constructors, start() and setup on the caller's thread;
 * never from the signal handler, the worker loop, destructors, atfork
 * handlers or the per-sample path. Catch it to handle any cpu-snap failure.
 */
class error : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
};

/**
 * Invalid configuration, detected before any resource is acquired. The
 * caller fixes the configuration and retries.
 */
class config_error : public error
{
public:
    using error::error;
};

/**
 * Programming error in how the API is used: wrong call order, or a second
 * active signal backend. The caller fixes the call sequence.
 */
class usage_error : public error
{
public:
    using error::error;
};

namespace detail
{
class coded_error : public error
{
public:
    coded_error(std::error_code code, const std::string& what)
    : error(what)
    , m_code(code)
    {}
    explicit coded_error(const std::string& what)
    : error(what)
    {}

    std::error_code code() const noexcept { return m_code; }

private:
    std::error_code m_code;
};
}  // namespace detail

/**
 * The kernel, privileges or the OS refused an operation (perf_event_open,
 * mmap, epoll, timer_create, sigaction, /proc enumeration). code() carries the
 * errno as a generic_category error_code, or is empty when there is none. The
 * caller relaxes the environment (perf_event_paranoid, CAP_PERFMON) or picks a
 * different backend.
 */
class environment_error : public detail::coded_error
{
public:
    using detail::coded_error::coded_error;

    /** Builds "<what>: <strerror(err)>" with code() set to @p err. */
    [[nodiscard]] static environment_error from_errno(int err, const std::string& what)
    {
        return environment_error(std::error_code(err, std::generic_category()),
                                 what + ": " + std::strerror(err));
    }
};

/**
 * A limit or capacity is exhausted: tid table full, frame-trie node cap,
 * record larger than the chunk, ENOSPC. code() is empty when there is no OS
 * code. The caller raises the relevant limit or reduces the load.
 */
class resource_error : public detail::coded_error
{
public:
    using detail::coded_error::coded_error;
};

}  // namespace cpu_snap
