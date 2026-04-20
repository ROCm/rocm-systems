// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common.hpp"
#include "core/components/fwd.hpp"
#include "core/defines.hpp"
#include "core/timemory.hpp"
#include "library/thread_data.hpp"

#include <timemory/components/base/declaration.hpp>
#include <timemory/mpl/concepts.hpp>
#include <timemory/unwind/cache.hpp>
#include <timemory/unwind/processed_entry.hpp>
#include <timemory/unwind/stack.hpp>

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace rocprofsys
{
namespace component
{
struct backtrace : comp::empty_base
{
    static constexpr size_t stack_depth = ROCPROFSYS_MAX_UNWIND_DEPTH;

    using data_t            = tim::unwind::stack<stack_depth>;
    using cache_type        = typename data_t::cache_type;
    using entry_type        = tim::unwind::processed_entry;
    using clock_type        = std::chrono::steady_clock;
    using value_type        = void;
    using system_clock      = std::chrono::system_clock;
    using system_time_point = typename system_clock::time_point;

    static std::string label();
    static std::string description();

    backtrace()                     = default;
    ~backtrace()                    = default;
    backtrace(const backtrace&)     = default;
    backtrace(backtrace&&) noexcept = default;

    backtrace& operator=(const backtrace&)     = default;
    backtrace& operator=(backtrace&&) noexcept = default;

    static std::vector<entry_type> filter_and_patch(const std::vector<entry_type>&);

    static void start();
    static void stop();

    void                    sample(int = -1);
    bool                    empty() const;
    size_t                  size() const;
    std::vector<entry_type> get() const;
    data_t                  get_data() const { return m_data; }

private:
    data_t m_data = {};
};
}  // namespace component
}  // namespace rocprofsys
