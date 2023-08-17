// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR rhs
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR rhsWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR rhs DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-register/version.h>

#ifndef ROCPROFILER_REGISTER_LOG_COLORS_AVAILABLE
#    define ROCPROFILER_REGISTER_LOG_COLORS_AVAILABLE 1
#endif

#ifndef ROCPROFILER_REGISTER_PROJECT_NAME
#    define ROCPROFILER_REGISTER_PROJECT_NAME "rocprofiler-register"
#endif

#ifndef ROCP_REG_FILE_NAME
#    define ROCP_REG_FILE_NAME                                                           \
        ::std::string{ __FILE__ }                                                        \
            .substr(::std::string_view{ __FILE__ }.find_last_of('/') + 1)                \
            .c_str()
#endif

#include <cstdlib>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocprofiler_register
{
namespace log
{
bool&
monochrome();

inline bool&
monochrome()
{
    static bool _v = []() {
        auto        _val      = false;
        const char* _env_cstr = nullptr;
        _env_cstr             = std::getenv("ROCPROFILER_REGISTER_MONOCHROME");

        if(!_env_cstr) _env_cstr = std::getenv("MONOCHROME");

        if(_env_cstr)
        {
            auto _env = std::string{ _env_cstr };

            // check if numeric
            if(_env.find_first_not_of("0123456789") == std::string::npos)
            {
                return _env.length() > 1 || _env[0] != '0';
            }

            for(auto& itr : _env)
                itr = tolower(itr);

            // check for matches to acceptable forms of false
            for(const auto& itr : { "off", "false", "no", "n", "f" })
            {
                if(_env == itr) return false;
            }

            // check for matches to acceptable forms of true
            for(const auto& itr : { "on", "true", "yes", "y", "t" })
            {
                if(_env == itr) return true;
            }
        }
        return _val;
    }();
    return _v;
}

namespace color
{
static constexpr auto info_value    = "\033[01;34m";
static constexpr auto warning_value = "\033[01;33m";
static constexpr auto fatal_value   = "\033[01;31m";
static constexpr auto source_value  = "\033[01;32m";
static constexpr auto dmesg_value   = "\033[01;37m";
static constexpr auto end_value     = "\033[0m";

inline const char*
info()
{
    return (log::monochrome()) ? "" : info_value;
}

inline const char*
warning()
{
    return (log::monochrome()) ? "" : warning_value;
}

inline const char*
fatal()
{
    return (log::monochrome()) ? "" : fatal_value;
}

inline const char*
source()
{
    return (log::monochrome()) ? "" : source_value;
}

inline const char*
dmesg()
{
    return (log::monochrome()) ? "" : dmesg_value;
}

inline const char*
end()
{
    return (log::monochrome()) ? "" : end_value;
}
}  // namespace color
}  // namespace log
}  // namespace rocprofiler_register

namespace rocprofiler_register
{
namespace log
{
struct base
{
public:
    static base indent(size_t, size_t = 2) { return base{}; }

    template <typename Tp>
    auto operator<<(Tp&&)
    {
        return base{};
    }
};

struct logger : public base
{
    logger() = default;

    explicit logger(bool _exit)
    : m_exit{ _exit }
    { }

    ~logger()
    {
        if(m_done)
        {
            std::cerr << color::end() << "\n";
            if(m_exit) abort();
        }
    }

    logger(logger&& rhs) noexcept
    {
        m_exit     = rhs.m_exit;
        m_done     = rhs.m_done;
        rhs.m_done = false;
    }

    logger& operator=(logger&& rhs) noexcept
    {
        m_exit     = rhs.m_exit;
        m_done     = rhs.m_done;
        rhs.m_done = false;
        return *this;
    }

    logger&& indent(size_t _n, size_t _tab_size = 4)
    {
        for(size_t i = 0; i < _n; i++)
            for(size_t j = 0; j < _tab_size; j++)
                std::cerr << " ";
        return std::move(*this);
    }

    template <typename Tp>
    logger&& operator<<(Tp&& _v)
    {
        std::cerr << std::forward<Tp>(_v);
        return std::move(*this);
    }

private:
    bool m_exit = false;
    bool m_done = true;
};

template <typename Tp>
inline auto&
get_color_hist()
{
    static thread_local std::vector<std::pair<Tp*, const char*>> _v{};
    return _v;
}

template <typename Tp>
inline auto
push_color_hist(Tp* _v, const char* _c)
{
    if(!monochrome()) get_color_hist<Tp>().emplace_back(_v, _c);
    return _c;
}

template <typename Tp>
inline std::string
pop_color_hist(Tp* _v)
{
    if(monochrome()) return std::string{};

    auto& _hist = get_color_hist<Tp>();
    for(auto itr = _hist.rbegin(); itr != _hist.rend(); ++itr)
    {
        Tp* _addr = itr->first;
        if(_addr == _v)
        {
            auto fitr = _hist.begin();
            std::advance(fitr, std::distance(_hist.rbegin(), itr));
            _hist.erase(fitr);
        }
    }
    for(auto itr = _hist.rbegin(); itr != _hist.rend(); ++itr)
    {
        if(itr->first == _v) return itr->second;
    }
    return color::end();
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
info(std::basic_ostream<CharT, Traits>& os)
{
    return (os << push_color_hist(&os, color::info()));
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
warning(std::basic_ostream<CharT, Traits>& os)
{
    return (os << push_color_hist(&os, color::warning()));
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
fatal(std::basic_ostream<CharT, Traits>& os)
{
    return (os << push_color_hist(&os, color::fatal()));
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
source(std::basic_ostream<CharT, Traits>& os)
{
    return (os << push_color_hist(&os, color::source()));
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
end(std::basic_ostream<CharT, Traits>& os)
{
    return (os << push_color_hist(&os, color::end()));
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
pop(std::basic_ostream<CharT, Traits>& os)
{
    return (os << pop_color_hist(&os));
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
reset(std::basic_ostream<CharT, Traits>& os)
{
    pop_color_hist(&os);
    return (os << color::end());
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
flush(std::basic_ostream<CharT, Traits>& os)
{
    return (os << pop << std::flush);
}

template <typename CharT, typename Traits>
std::basic_ostream<CharT, Traits>&
endl(std::basic_ostream<CharT, Traits>& os)
{
    return (os << pop << "\n" << std::flush);
}

template <typename StreamT>
struct stream_base
{
    stream_base(StreamT& _os, const char* _color)
    : m_os{ _os }
    {
        m_os << push_color_hist(&_os, _color);
    }

    ~stream_base() { m_os << pop_color_hist(&m_os); }

    stream_base(const stream_base&) = delete;
    stream_base& operator=(const stream_base&) = delete;

    stream_base(stream_base&& rhs) noexcept = default;
    stream_base& operator=(stream_base&& rhs) noexcept = default;

    template <typename Tp>
    stream_base& operator<<(Tp&& _v)
    {
        m_os << std::forward<Tp>(_v);
        return *this;
    }

    template <typename Tp>
    stream_base& operator<<(Tp& _v)
    {
        m_os << _v;
        return *this;
    }

    stream_base& put(char _c)
    {
        m_os.put(_c);
        return *this;
    }

    stream_base& endl()
    {
        m_os << std::endl;
        return *this;
    }

    template <typename... Args>
    stream_base& write(Args&&... _args)
    {
        m_os.write(std::forward<Args>(_args)...);
        return *this;
    }

    stream_base& flush()
    {
        m_os << std::flush;
        return *this;
    }

    auto tellp() { return m_os.tellp(); }

    template <typename... Args>
    stream_base& seekp(Args&&... _args)
    {
        m_os.seekp(std::forward<Args>(_args)...);
        return *this;
    }

private:
    StreamT& m_os;
};

template <typename StreamT>
stream_base<StreamT>
stream(StreamT& _os, const char* _color)
{
    return stream_base<StreamT>{ _os, _color };
}

template <typename StreamT>
stream_base<StreamT>
info_stream(StreamT& _os)
{
    return stream_base<StreamT>{ _os, color::info() };
}

template <typename StreamT>
stream_base<StreamT>
source_stream(StreamT& _os)
{
    return stream_base<StreamT>{ _os, color::source() };
}

template <typename StreamT>
stream_base<StreamT>
warning_stream(StreamT& _os)
{
    return stream_base<StreamT>{ _os, color::warning() };
}

template <typename StreamT>
stream_base<StreamT>
fatal_stream(StreamT& _os)
{
    return stream_base<StreamT>{ _os, color::fatal() };
}

inline std::string
string(const char* _color, std::string_view _v)
{
    return std::string{ _color } + std::string{ _v } + std::string{ color::end() };
}

inline std::string
string(const char* _color, std::stringstream& _v)
{
    return std::string{ _color } + _v.str() + std::string{ color::end() };
}

inline std::string
string(const char* _color, std::stringstream&& _v)
{
    return std::string{ _color } + _v.str() + std::string{ color::end() };
}

template <typename Tp>
inline auto
info_string(Tp&& _v)
{
    return string(color::info(), std::forward<Tp>(_v));
}

template <typename Tp>
inline auto
source_string(Tp&& _v)
{
    return string(color::source(), std::forward<Tp>(_v));
}

template <typename Tp>
inline auto
warning_string(Tp&& _v)
{
    return string(color::warning(), std::forward<Tp>(_v));
}

template <typename Tp>
inline auto
fatal_string(Tp&& _v)
{
    return string(color::fatal(), std::forward<Tp>(_v));
}
}  // namespace log
}  // namespace rocprofiler_register

#if !defined(ROCPROFILER_REGISTER_LOG)
#    define ROCPROFILER_REGISTER_LOG(COLOR, EXIT_CODE)                                   \
        (::rocprofiler_register::log::logger(EXIT_CODE)                                  \
         << ::rocprofiler_register::log::color::end()                                    \
         << ::rocprofiler_register::log::color::source() << "["                          \
         << ROCPROFILER_REGISTER_PROJECT_NAME << "][" << ROCP_REG_FILE_NAME << ":"       \
         << __LINE__ << "][" << getpid() << "] "                                         \
         << ::rocprofiler_register::log::color::end() << COLOR)
#endif

#if defined(NDEBUG)
#    if !defined(ROCPROFILER_REGISTER_INFO)
#        define ROCPROFILER_REGISTER_INFO (::rocprofiler_register::log::base())
#    endif
#    if !defined(ROCPROFILER_REGISTER_ASSERT)
#        define ROCPROFILER_REGISTER_ASSERT(COND) (::rocprofiler_register::log::base())
#    endif
#else
#    if !defined(ROCPROFILER_REGISTER_INFO)
#        define ROCPROFILER_REGISTER_INFO                                                \
            ROCPROFILER_REGISTER_LOG(::rocprofiler_register::log::color::info(), false)
#    endif
#    if !defined(ROCPROFILER_REGISTER_ASSERT)
#        define ROCPROFILER_REGISTER_ASSERT(COND)                                        \
            (COND) ? ::rocprofiler_register::log::base() : ROCPROFILER_REGISTER_FATAL
#    endif
#endif

#if !defined(ROCPROFILER_REGISTER_WARNING)
#    define ROCPROFILER_REGISTER_WARNING                                                 \
        ROCPROFILER_REGISTER_LOG(::rocprofiler_register::log::color::warning(), false)
#endif

#if !defined(ROCPROFILER_REGISTER_FATAL)
#    define ROCPROFILER_REGISTER_FATAL                                                   \
        ROCPROFILER_REGISTER_LOG(::rocprofiler_register::log::color::fatal(), true)
#endif

#if !defined(ROCPROFILER_REGISTER_PREFER)
#    define ROCPROFILER_REGISTER_PREFER(COND)                                            \
        (COND) ? ::rocprofiler_register::log::base() : ROCPROFILER_REGISTER_WARNING
#endif

#if !defined(ROCPROFILER_REGISTER_REQUIRE)
#    define ROCPROFILER_REGISTER_REQUIRE(COND)                                           \
        (COND) ? ::rocprofiler_register::log::base() : ROCPROFILER_REGISTER_FATAL
#endif
