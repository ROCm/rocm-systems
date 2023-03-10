// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "log.hpp"
#include "fwd.hpp"

#include <cmath>
#include <iomanip>
#include <regex>
#include <vector>

namespace color = tim::log::color;

namespace
{
std::vector<log_entry> log_entries = {};

auto
get_color_regex(std::string _v)
{
    auto _p = _v.find("[");
    if(_p != std::string::npos) _v.insert(_p, "\\");
    return JOIN("", "\\", _v);
}

auto _color_regex = std::regex{ JOIN("", "(", get_color_regex(tim::log::color::info()),
                                     "|", get_color_regex(tim::log::color::source()), "|",
                                     get_color_regex(tim::log::color::warning()), "|",
                                     get_color_regex(tim::log::color::fatal()), "|",
                                     get_color_regex(tim::log::color::end()), ")"),
                                std::regex_constants::optimize };
}  // namespace

log_entry::log_entry(std::string _msg)
: m_message{ std::move(_msg) }
, m_backtrace{ tim::get_unw_stack<4, 1>() }
{
    if(log_ofs) *log_ofs << as_string("", "", "") << "\n";
}

log_entry::log_entry(source_location _loc, std::string _msg)
: m_location{ _loc }
, m_message{ std::move(_msg) }
, m_backtrace{ tim::get_unw_stack<4, 1>() }
{
    if(log_ofs) *log_ofs << as_string("", "", "") << "\n";
}

std::string
log_entry::as_string(const char* _color, const char* _src, const char* _end) const
{
    std::stringstream _ss;
    if(m_location.function && m_location.file)
    {
        _ss << "[" << _src << m_location.file << ":" << m_location.line << _end << "]["
            << _src << m_location.function << _end << "]";
    }

    bool _remove_color = (strlen(_color) + strlen(_src) + strlen(_end) == 0);

    _ss << " " << _color << std::regex_replace(m_message, std::regex{ "\n" }, " ... ")
        << _end;

    return (_remove_color) ? std::regex_replace(_ss.str(), _color_regex, "") : _ss.str();
}

log_entry&
log_entry::add_log_entry(log_entry&& _v)
{
    return log_entries.emplace_back(std::move(_v));
}

void
print_log_entries(std::ostream& _os, int64_t _count,
                  const std::function<bool(const log_entry&)>& _condition,
                  const std::function<void()>& _prelude, const char* _color,
                  bool _color_entries)
{
    size_t i0 = (_count < 0) ? 0 : std::max<int64_t>(log_entries.size() - _count, 0);
    size_t _w = std::log10(log_entries.size()) + 1;

    if(dynamic_cast<std::ofstream*>(&_os) ||
       (&_os != &std::cout && &_os != &std::cerr && &_os != &std::clog))
    {
        _color         = "";
        _color_entries = false;
    }

    const char* _end =
        (strlen(_color) > 0 || _color_entries) ? tim::log::color::end() : "";

    if(_prelude)
    {
        for(size_t i = i0; i < log_entries.size(); ++i)
        {
            if(!_condition || _condition(log_entries.at(i)))
            {
                _prelude();
                break;
            }
        }
    }

    // the requested number of log entries
    auto   _last   = std::string{};
    size_t _last_i = 0;
    size_t _last_n = 0;
    for(size_t i = i0; i < log_entries.size(); ++i)
    {
        auto& itr = log_entries.at(i);

        if(!_condition || _condition(itr))
        {
            auto _msg = ((_color_entries) ? itr.as_string() : itr.as_string("", "", ""));
            if(_msg != _last)
            {
                if(_last_n > 0 && !_last.empty())
                {
                    _os << "[" << _color << std::setw(_w) << _last_i << "/"
                        << log_entries.size() << _end << "] ... repeated " << _last_n
                        << " times ...\n";
                }
                _last_n = 0;
                _last_i = i;
                _last   = _msg;
                _os << "[" << _color << std::setw(_w) << i << "/" << log_entries.size()
                    << _end << "]" << _msg << "\n";
            }
            else
            {
                ++_last_n;
            }
        }
    }

    if(_last_n > 0)
    {
        _os << "[" << _color << std::setw(_w) << _last_i << "/" << log_entries.size()
            << _end << "] ... repeated " << _last_n << " times ...\n";
    }
}
