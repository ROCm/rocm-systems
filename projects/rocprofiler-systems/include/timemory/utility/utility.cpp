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
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TIMEMORY_UTILITY_UTILITY_CPP_
#define TIMEMORY_UTILITY_UTILITY_CPP_ 1

#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"

#if !defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/utility.hpp"
#endif

namespace tim
{
//
std::string
dirname(std::string _fname)
{
#if defined(TIMEMORY_UNIX)
    char* _cfname = realpath(_fname.c_str(), nullptr);
    _fname        = std::string(_cfname);
    free(_cfname);

    while(_fname.find("\\\\") != std::string::npos)
        _fname.replace(_fname.find("\\\\"), 2, "/");
    while(_fname.find('\\') != std::string::npos)
        _fname.replace(_fname.find('\\'), 1, "/");

    return _fname.substr(0, _fname.find_last_of('/'));
#elif defined(TIMEMORY_WINDOWS)
    while(_fname.find('/') != std::string::npos)
        _fname.replace(_fname.find('/'), 1, "\\");

    _fname = _fname.substr(0, _fname.find_last_of('\\'));
    return (_fname.at(_fname.length() - 1) == '\\')
               ? _fname.substr(0, _fname.length() - 1)
               : _fname;
#endif
}

//--------------------------------------------------------------------------------------//

int
makedir(std::string _dir, int umask)
{
#if defined(TIMEMORY_UNIX)
    while(_dir.find("\\\\") != std::string::npos)
        _dir.replace(_dir.find("\\\\"), 2, "/");
    while(_dir.find('\\') != std::string::npos)
        _dir.replace(_dir.find('\\'), 1, "/");

    if(_dir.length() == 0)
        return 0;

    int ret = mkdir(_dir.c_str(), umask);
    if(ret != 0)
    {
        int err = errno;
        if(err != EEXIST)
        {
            std::cerr << "mkdir(" << _dir.c_str() << ", " << umask
                      << ") returned: " << ret << std::endl;
            std::stringstream _sdir;
            _sdir << "/bin/mkdir -p " << _dir;
            return (launch_process(_sdir.str().c_str())) ? 0 : 1;
        }
    }
#elif defined(TIMEMORY_WINDOWS)
    consume_parameters(umask);
    while(_dir.find('/') != std::string::npos)
        _dir.replace(_dir.find('/'), 1, "\\");

    if(_dir.length() == 0)
        return 0;

    int ret = _mkdir(_dir.c_str());
    if(ret != 0)
    {
        int err = errno;
        if(err != EEXIST)
        {
            std::cerr << "_mkdir(" << _dir.c_str() << ") returned: " << ret << std::endl;
            std::stringstream _sdir;
            _sdir << "mkdir " << _dir;
            return (launch_process(_sdir.str().c_str())) ? 0 : 1;
        }
    }
#endif
    return 0;
}

//--------------------------------------------------------------------------------------//

bool
get_bool(const std::string& strbool, bool _default) noexcept
{
    // empty string returns default
    if(strbool.empty())
        return _default;

    // check if numeric
    if(strbool.find_first_not_of("0123456789") == std::string::npos)
    {
        if(strbool.length() > 1 || strbool[0] != '0')
            return true;
        return false;
    }

    // convert to lowercase
    auto _val = std::string{ strbool };
    for(auto& itr : _val)
        itr = tolower(itr);

    // check for matches to acceptable forms of false
    for(const auto& itr : { "off", "false", "no", "n", "f" })
    {
        if(_val == itr)
            return false;
    }

    // check for matches to acceptable forms of true
    for(const auto& itr : { "on", "true", "yes", "y", "t" })
    {
        if(_val == itr)
            return true;
    }

    return _default;
}
//
//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_UNIX)
//
std::string
demangle_backtrace(const char* cstr)
{
    auto _trim = [](std::string& _sub, size_t& _len) {
        size_t _pos = 0;
        while(!_sub.empty() && (_pos = _sub.find_first_of(' ')) == 0)
        {
            _sub = _sub.erase(_pos, 1);
            --_len;
        }
        while(!_sub.empty() && (_pos = _sub.find_last_of(' ')) == _sub.length() - 1)
        {
            _sub = _sub.substr(0, _sub.length() - 1);
            --_len;
        }
        return _sub;
    };

    auto _save = std::string{ cstr };
    auto str   = demangle(std::string(cstr));
    if(str.empty())
        return str;
    auto beg = str.find("(");
    if(beg == std::string::npos)
    {
        beg = str.find("_Z");
        if(beg != std::string::npos)
            beg -= 1;
    }
    auto end = str.find("+", beg);
    if(beg != std::string::npos && end != std::string::npos)
    {
        auto len = end - (beg + 1);
        auto sub = str.substr(beg + 1, len);
        auto dem = demangle(_trim(sub, len));
        str      = str.replace(beg + 1, len, dem);
    }
    else if(beg != std::string::npos)
    {
        auto len = str.length() - (beg + 1);
        auto sub = str.substr(beg + 1, len);
        auto dem = demangle(_trim(sub, len));
        str      = str.replace(beg + 1, len, dem);
    }
    else if(end != std::string::npos)
    {
        auto len = end;
        auto sub = str.substr(beg, len);
        auto dem = demangle(_trim(sub, len));
        str      = str.replace(beg, len, dem);
    }

    if(str.empty())
    {
        TIMEMORY_TESTING_EXCEPTION(__FUNCTION__ << " resulted in empty string. input: "
                                                << _save);
        return _save;
    }

    // cleanup std::_1::basic_<...> types since demangled versions are for diagnostics
    using pair_t = std::pair<std::string, std::string>;
    for(auto&& itr : { pair_t{ demangle<std::string>(), "std::string" },
                       pair_t{ demangle<std::istream>(), "std::istream" },
                       pair_t{ demangle<std::ostream>(), "std::ostream" },
                       pair_t{ demangle<std::stringstream>(), "std::stringstream" },
                       pair_t{ demangle<std::istringstream>(), "std::istringstream" },
                       pair_t{ demangle<std::ostringstream>(), "std::ostringstream" } })
        str = str_transform(str, itr.first, "", [itr](const std::string&) -> std::string {
            return itr.second;
        });
    return str;
}
//
std::string
demangle_backtrace(const std::string& str)
{
    return demangle_backtrace(str.c_str());
}
//
std::string
demangle_unw_backtrace(const char* cstr)
{
    auto _save = std::string{ cstr };
    auto _str  = tim::demangle(cstr);
    if(_str.empty())
        return _str;
    auto _beg = _str.find("_Z");
    auto _end = _str.find(" ", _beg);
    if(_beg != std::string::npos && _end != std::string::npos)
    {
        auto _len = _end - _beg;
        auto _sub = _str.substr(_beg, _len);
        auto _dem = demangle(_sub);
        _str      = _str.replace(_beg, _len, _dem);
    }

    if(_str.empty())
    {
        TIMEMORY_TESTING_EXCEPTION(__FUNCTION__ << " resulted in empty string. input: "
                                                << _save);
        return _save;
    }

    // cleanup std::_1::basic_<...> types since demangled versions are for diagnostics
    using pair_t = std::pair<std::string, std::string>;
    for(auto&& itr : { pair_t{ demangle<std::string>(), "std::string" },
                       pair_t{ demangle<std::istream>(), "std::istream" },
                       pair_t{ demangle<std::ostream>(), "std::ostream" },
                       pair_t{ demangle<std::stringstream>(), "std::stringstream" },
                       pair_t{ demangle<std::istringstream>(), "std::istringstream" },
                       pair_t{ demangle<std::ostringstream>(), "std::ostringstream" } })
        _str =
            str_transform(_str, itr.first, "", [itr](const std::string&) -> std::string {
                return itr.second;
            });

    return _str;
}
//
std::string
demangle_unw_backtrace(const std::string& _str)
{
    return demangle_unw_backtrace(_str.c_str());
}
//
#endif  // defined(TIMEMORY_UNIX)
//
std::vector<std::string>
read_command_line(pid_t _pid)
{
    std::vector<std::string> _cmdline;
#if defined(TIMEMORY_LINUX)
    std::stringstream fcmdline;
    fcmdline << "/proc/" << _pid << "/cmdline";
    std::ifstream ifs(fcmdline.str().c_str());
    if(ifs)
    {
        char        cstr;
        std::string sarg;
        while(!ifs.eof())
        {
            ifs >> cstr;
            if(!ifs.eof())
            {
                if(cstr != '\0')
                {
                    sarg += cstr;
                }
                else
                {
                    _cmdline.push_back(sarg);
                    sarg = "";
                }
            }
        }
        ifs.close();
    }

#else
    consume_parameters(_pid);
#endif
    return _cmdline;
}
//
namespace utility
{
//
path::path(const std::string& _path)
: std::string(osrepr(_path))
{}

path::path(char* _path)
: std::string(osrepr(std::string(_path)))
{}

path::path(const path& rhs)
: std::string(osrepr(rhs))
{}

path::path(const char* _path)
: std::string(osrepr(std::string(const_cast<char*>(_path))))
{}

path&
path::operator=(const std::string& rhs)
{
    std::string::operator=(osrepr(rhs));
    return *this;
}

path&
path::operator=(const path& rhs)
{
    if(this != &rhs)
        std::string::operator=(osrepr(rhs));
    return *this;
}

path&
path::insert(size_type __pos, const std::string& __s)
{
    std::string::operator=(osrepr(std::string::insert(__pos, __s)));
    return *this;
}

path&
path::insert(size_type __pos, const path& __s)
{
    std::string::operator=(osrepr(std::string::insert(__pos, __s)));
    return *this;
}

TIMEMORY_UTILITY_INLINE std::string
                        path::os()
{
#if defined(TIMEMORY_WINDOWS)
    return "\\";
#elif defined(TIMEMORY_UNIX)
    return "/";
#endif
}

TIMEMORY_UTILITY_INLINE std::string
                        path::inverse()
{
#if defined(TIMEMORY_WINDOWS)
    return "/";
#elif defined(TIMEMORY_UNIX)
    return "\\";
#endif
}

// OS-dependent representation
TIMEMORY_UTILITY_INLINE std::string
                        path::osrepr(std::string _path)
{
#if defined(TIMEMORY_WINDOWS)
    while(_path.find('/') != std::string::npos)
        _path.replace(_path.find('/'), 1, "\\");
#elif defined(TIMEMORY_UNIX)
    while(_path.find("\\\\") != std::string::npos)
        _path.replace(_path.find("\\\\"), 2, "/");
    while(_path.find('\\') != std::string::npos)
        _path.replace(_path.find('\\'), 1, "/");
#endif
    return _path;
}
//
}  // namespace utility
}  // namespace tim
//

#endif  // !defined(TIMEMORY_UTILITY_UTILITY_CPP_)
