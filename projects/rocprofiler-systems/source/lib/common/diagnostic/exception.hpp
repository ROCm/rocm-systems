// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/diagnostic/stacktrace.hpp"

#include <stdexcept>
#include <string>

namespace rocprofsys
{
/// Base class for all rocprofsys-thrown exceptions.
///
/// Captures a stacktrace at construction time so the throw site is recorded
/// once, regardless of where the exception is later caught and formatted.
/// Derives from `std::runtime_error` so legacy `catch (std::exception&)` and
/// `catch (std::runtime_error&)` keep working unchanged.
///
/// `what()` returns the message only. Callers wanting the trace ask for
/// `trace()` separately, or rely on
/// `rocprofsys::diagnostic::format_exception(e)` to render both.
class exception : public std::runtime_error
{
public:
    /// @param msg         Human-readable message, returned by `what()`.
    /// @param skip_frames Frames to drop from the top of the captured trace.
    ///                    Defaults to 1 to drop the `exception` ctor itself.
    ///                    Subclasses that add their own ctor frames should
    ///                    pass `skip_frames + 1`.
    explicit exception(std::string msg, int skip_frames = 1);

    ~exception() override;

    exception(const exception&)                = default;
    exception(exception&&) noexcept            = default;
    exception& operator=(const exception&)     = default;
    exception& operator=(exception&&) noexcept = default;

    const char*                   what() const noexcept override;
    const diagnostic::stacktrace& trace() const noexcept;
    const std::string&            message() const noexcept;

private:
    std::string            m_message;
    diagnostic::stacktrace m_trace;
};

/// Defines a new exception type derived from `Base` with the same constructor
/// surface. Empty body; the derivation alone gives `dynamic_cast` and the
/// `catch (Base&)` family the right behavior.
#define ROCPROFSYS_DEFINE_EXCEPTION(Name, Base)                                          \
    class Name : public Base                                                             \
    {                                                                                    \
    public:                                                                              \
        using Base::Base;                                                                \
    }

ROCPROFSYS_DEFINE_EXCEPTION(config_error, exception);

ROCPROFSYS_DEFINE_EXCEPTION(serialization_error, exception);
ROCPROFSYS_DEFINE_EXCEPTION(storage_error, serialization_error);
ROCPROFSYS_DEFINE_EXCEPTION(schema_error, serialization_error);

ROCPROFSYS_DEFINE_EXCEPTION(device_error, exception);
ROCPROFSYS_DEFINE_EXCEPTION(gpu_error, device_error);
ROCPROFSYS_DEFINE_EXCEPTION(nic_error, device_error);
ROCPROFSYS_DEFINE_EXCEPTION(cpu_error, device_error);

ROCPROFSYS_DEFINE_EXCEPTION(sdk_error, exception);
ROCPROFSYS_DEFINE_EXCEPTION(sampling_error, exception);
ROCPROFSYS_DEFINE_EXCEPTION(causal_error, exception);
ROCPROFSYS_DEFINE_EXCEPTION(runtime_error, exception);

/// Out-of-range, mirrors `std::out_of_range` semantics. Kept distinct from
/// `runtime_error` so callers that previously caught `std::out_of_range` (e.g.
/// container bounds) can still target the same category.
class out_of_range : public exception
{
public:
    using exception::exception;
};
}  // namespace rocprofsys
