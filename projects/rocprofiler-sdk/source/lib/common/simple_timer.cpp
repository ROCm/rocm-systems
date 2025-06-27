// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/simple_timer.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>

namespace rocprofiler
{
namespace common
{
simple_timer::simple_timer(std::string&& label, int log_level)
: m_label{std::move(label)}
, m_os{simple_timer::get_log_stream(log_level)}
{
    start();
}

simple_timer::simple_timer(std::string&& label, defer_start, int log_level)
: m_label{std::move(label)}
, m_os{simple_timer::get_log_stream(log_level)}
{}

simple_timer::~simple_timer()
{
    if(m_quiet)
        return;
    else if(m_end <= m_beg)
        stop();

    report();
}

simple_timer&
simple_timer::start()
{
    m_beg = clock_type::now();
    return *this;
}

simple_timer&
simple_timer::stop()
{
    m_end = clock_type::now();
    return *this;
}

double
simple_timer::get() const
{
    if(m_end <= m_beg) return {};
    return std::chrono::duration_cast<std::chrono::duration<double>>(m_end - m_beg).count();
}

size_t
simple_timer::get_nsec() const
{
    if(m_end <= m_beg) return {};
    return std::chrono::duration_cast<std::chrono::nanoseconds>(m_end - m_beg).count();
}

simple_timer&
simple_timer::report()
{
    if(m_os)
    {
        (*m_os) << fmt::format("{} :: {:12.6f} sec", m_label, get());
    }
    return *this;
}

std::ostream*
simple_timer::get_log_stream(int log_level)
{
    switch(log_level)
    {
        case ROCP_LOG_LEVEL_NONE: return nullptr;
        case ROCP_LOG_LEVEL_WARNING: return &ROCP_WARNING;
        case ROCP_LOG_LEVEL_ERROR: return &ROCP_ERROR;
        case ROCP_LOG_LEVEL_INFO: return &ROCP_INFO;
        case ROCP_LOG_LEVEL_TRACE: return VLOG_IS_ON(log_level) ? &ROCP_INFO : nullptr;
        default: break;
    }

    ROCP_CI_LOG(WARNING) << fmt::format("Unsupported log level for simple timer: {}", log_level);
    return &ROCP_INFO;
}

std::ostream&
operator<<(std::ostream& _os, const simple_timer& _val)
{
    _val.set_quiet(true);
    _os << fmt::format("{} :: {:12.6f} sec", _val.label(), _val.get());

    return _os;
}
}  // namespace common
}  // namespace rocprofiler
