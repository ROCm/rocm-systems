// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "library/components/backtrace_timestamp.hpp"
#include "core/control/clocks/posix.hpp"
#include "library/thread_info.hpp"
#include <cstdint>

namespace rocprofsys
{
namespace component
{
bool
backtrace_timestamp::operator<(const backtrace_timestamp& rhs) const
{
    return std::tie(m_tid, m_real) < std::tie(rhs.m_tid, rhs.m_real);
}

bool
backtrace_timestamp::is_valid() const
{
    const auto& _info = thread_info::get(m_tid, SequentTID);
    return (_info) ? _info->is_valid_time(m_real) : false;
}

void
backtrace_timestamp::sample(int)
{
    m_tid  = tim::threading::get_id();
    m_real = control::clocks::timeline_ns();
}
}  // namespace component
}  // namespace rocprofsys

TIMEMORY_INITIALIZE_STORAGE(rocprofsys::component::backtrace_timestamp)
