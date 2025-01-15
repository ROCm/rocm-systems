// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "library/components/backtrace_timestamp.hpp"
#include "library/thread_info.hpp"

#include <timemory/components/timing/backends.hpp>

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
    m_real = tim::get_clock_real_now<uint64_t, std::nano>();
}
}  // namespace component
}  // namespace rocprofsys

TIMEMORY_INITIALIZE_STORAGE(rocprofsys::component::backtrace_timestamp)
