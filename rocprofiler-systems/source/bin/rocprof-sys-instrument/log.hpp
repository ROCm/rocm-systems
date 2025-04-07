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

#pragma once

#include <timemory/log/color.hpp>
#include <timemory/utility/backtrace.hpp>
#include <timemory/utility/join.hpp>

#include <iosfwd>
#include <ostream>
#include <string>
#include <tuple>

#if !defined(JOIN)
#    define JOIN(...) ::timemory::join::join(__VA_ARGS__)
#endif

struct log_entry;

void
print_log_entries(std::ostream& = std::cerr, int64_t _count = 10,
                  const std::function<bool(const log_entry&)>& _cond    = {},
                  const std::function<void()>&                 _prelude = {},
                  const char* _color         = tim::log::color::warning(),
                  bool        _color_entries = true);

struct log_entry
{
    struct source_location
    {
        const char* function = nullptr;
        const char* file     = nullptr;
        int         line     = 0;
    };

    log_entry(std::string _msg);
    log_entry(source_location _loc, std::string _msg);

    std::string as_string(const char* _color = tim::log::color::info(),
                          const char* _src   = tim::log::color::source(),
                          const char* _end   = tim::log::color::end()) const;

    static log_entry& add_log_entry(log_entry&&);

    log_entry& force(bool _v = true)
    {
        m_forced = _v;
        return *this;
    }

    bool forced() const { return m_forced; }

private:
    bool                  m_forced    = false;  // if should always be displayed
    source_location       m_location  = {};
    std::string           m_message   = {};
    tim::unwind::stack<4> m_backtrace = {};

    friend void print_log_entries(std::ostream&, int64_t,
                                  std::function<bool(const log_entry&)>, const char*,
                                  bool);
};

#define ROCPROFSYS_ADD_LOG_ENTRY(...)                                                    \
    log_entry::add_log_entry(                                                            \
        { log_entry::source_location{ __FUNCTION__, __FILE__, __LINE__ },                \
          timemory::join::join(' ', __VA_ARGS__) })

#define ROCPROFSYS_ADD_DETAILED_LOG_ENTRY(DELIM, ...)                                    \
    log_entry::add_log_entry(                                                            \
        { log_entry::source_location{ __FUNCTION__, __FILE__, __LINE__ },                \
          timemory::join::join(DELIM, __VA_ARGS__) })
