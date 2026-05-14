// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/timemory.hpp"

#include <timemory/components/base.hpp>
#include <timemory/macros/language.hpp>
#include <timemory/mpl/concepts.hpp>

#include <chrono>
#include <cstdint>

namespace rocprofsys
{
namespace component
{
struct backtrace_timestamp : comp::empty_base
{
    using value_type = void;

    static std::string label() { return "backtrace_timestamp"; }
    static std::string description() { return "Timestamp for backtrace"; }

    backtrace_timestamp()                               = default;
    ~backtrace_timestamp()                              = default;
    backtrace_timestamp(const backtrace_timestamp&)     = default;
    backtrace_timestamp(backtrace_timestamp&&) noexcept = default;

    backtrace_timestamp& operator=(const backtrace_timestamp&)     = default;
    backtrace_timestamp& operator=(backtrace_timestamp&&) noexcept = default;

    bool operator<(const backtrace_timestamp& rhs) const;

    static void start() {}
    static void stop() {}

    void sample(int = -1);

    auto get_tid() const { return m_tid; }
    auto get_timestamp() const { return m_real; }
    bool is_valid() const;

private:
    std::int64_t  m_tid  = 0;
    std::uint64_t m_real = 0;
};
}  // namespace component
}  // namespace rocprofsys
