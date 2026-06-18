// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
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
print_log_entries(std::ostream& = std::cerr, std::int64_t _count = 10,
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

    friend void print_log_entries(std::ostream&, std::int64_t,
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
