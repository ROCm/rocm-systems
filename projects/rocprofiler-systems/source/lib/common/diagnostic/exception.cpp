// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/exception.hpp"

#include <utility>

namespace rocprofsys
{
exception::exception(std::string msg, int skip_frames)
: std::runtime_error{ msg }
, m_message{ std::move(msg) }
// Skip exception::exception itself plus stacktrace::capture's own bookkeeping
// frame, then any caller-requested extras (e.g. derived ctor frames).
, m_trace{ diagnostic::stacktrace::capture(skip_frames) }
{}

exception::~exception() = default;

const char*
exception::what() const noexcept
{
    return m_message.c_str();
}

const diagnostic::stacktrace&
exception::trace() const noexcept
{
    return m_trace;
}

const std::string&
exception::message() const noexcept
{
    return m_message;
}
}  // namespace rocprofsys
