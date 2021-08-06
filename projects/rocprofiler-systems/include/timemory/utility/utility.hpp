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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//

/** \file utility/utility.hpp
 * \headerfile utility/utility.hpp "timemory/utility/utility.hpp"
 * General utility functions
 *
 */

#pragma once

#include "timemory/api.hpp"
#include "timemory/macros/compiler.hpp"
#include "timemory/macros/language.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/transient_function.hpp"
#include "timemory/utility/types.hpp"

#if defined(TIMEMORY_WINDOWS)
// Without this include on windows launch_process is needed by makedir but is
// missing at link time. With this include on linux there is a problem with
// delimit not being defined before it is used in popen.hpp. I think including
// popen.hpp at the end of this header would be portable and not need the ifdef,
// but that would be too weird, so we may have to live with the ifdef. (At
// least until makedir is implemented using std::filesystem.)
#    include "timemory/utility/popen.hpp"
#endif

#if defined(TIMEMORY_USE_LIBUNWIND)
#    include <libunwind.h>
#endif

// C library
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
// I/O
#include <fstream>
#include <iomanip>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <string>
// general
#include <functional>
#include <limits>
#include <regex>
#include <typeindex>
#include <utility>
// container
#include <vector>
// threading
#include <atomic>
#include <mutex>
#include <thread>

#if defined(TIMEMORY_UNIX)
#    include <cxxabi.h>
#    include <execinfo.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#elif defined(TIMEMORY_WINDOWS)
#    include <direct.h>
using pid_t = int;
#endif

#if !defined(DEFAULT_UMASK)
#    define DEFAULT_UMASK 0777
#endif

//--------------------------------------------------------------------------------------//

namespace tim
{
// alias here for common string type
// there is also a string_view_t alias in macros/language.hpp which is std::string in
// c++14 and std::string_view in c++17 and newer
using string_t = std::string;

/// \typedef std::recursive_mutex mutex_t
/// \brief Recursive mutex is used for convenience since the
/// performance penalty vs. a regular mutex is not really an issue since there are not
/// many locks in general.
using mutex_t = std::recursive_mutex;

/// \typedef std::unique_lock<std::recursive_mutex> auto_lock_t
/// \brief Unique lock type around \ref mutex_t
using auto_lock_t = std::unique_lock<mutex_t>;

//--------------------------------------------------------------------------------------//
// definition in popen.hpp
bool
launch_process(const char* cmd, const std::string& extra = "",
               std::ostream* os = nullptr);

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline bool
isfinite(const Tp& arg)
{
#if defined(TIMEMORY_WINDOWS)
    // Windows seems to be missing std::isfinite
    return (arg == arg && arg != std::numeric_limits<Tp>::infinity() &&
            arg != -std::numeric_limits<Tp>::infinity())
               ? true
               : false;
#else
    return std::isfinite(arg);
#endif
}

//======================================================================================//
//
//  General functions
//
//======================================================================================//

/// \fn mutex_t& type_mutex(uint64_t)
/// \tparam Tp data type for lock
/// \tparam ApiT API for lock
/// \tparam N max size
///
/// \brief A simple way to get a mutex for a class or common behavior, e.g.
/// `type_mutex<decltype(std::cout)>()` provides a mutex for synchronizing output streams.
/// Recommend using in conjunction with auto-lock:
/// `tim::auto_lock_t _lk{ type_mutex<Foo>() }`.
template <typename Tp, typename ApiT = TIMEMORY_API, size_t N = 4>
mutex_t&
type_mutex(uint64_t _n = 0)
{
    static std::array<mutex_t, N> _mutexes{};
    return _mutexes.at(_n % N);
}

//--------------------------------------------------------------------------------------//

inline std::string
demangle(const char* _mangled_name, int* _status = nullptr)
{
#if defined(TIMEMORY_ENABLE_DEMANGLE)
    // return the mangled since there is no buffer
    if(!_mangled_name)
        return std::string{};

    int         _ret = 0;
    std::string _demangled_name{ _mangled_name };
    if(!_status)
        _status = &_ret;

    // PARAMETERS to __cxa_demangle
    //  mangled_name:
    //      A NULL-terminated character string containing the name to be demangled.
    //  buffer:
    //      A region of memory, allocated with malloc, of *length bytes, into which the
    //      demangled name is stored. If output_buffer is not long enough, it is expanded
    //      using realloc. output_buffer may instead be NULL; in that case, the demangled
    //      name is placed in a region of memory allocated with malloc.
    //  _buflen:
    //      If length is non-NULL, the length of the buffer containing the demangled name
    //      is placed in *length.
    //  status:
    //      *status is set to one of the following values
    char* _demang = abi::__cxa_demangle(_mangled_name, nullptr, nullptr, _status);
    switch(*_status)
    {
        //  0 : The demangling operation succeeded.
        // -1 : A memory allocation failiure occurred.
        // -2 : mangled_name is not a valid name under the C++ ABI mangling rules.
        // -3 : One of the arguments is invalid.
        case 0:
        {
            if(_demang)
                _demangled_name = std::string{ _demang };
            break;
        }
        case -1:
        {
            char _msg[1024];
            ::memset(_msg, '\0', 1024 * sizeof(char));
            ::snprintf(_msg, 1024, "memory allocation failure occurred demangling %s",
                       _mangled_name);
            ::perror(_msg);
            break;
        }
        case -2: break;
        case -3:
        {
            char _msg[1024];
            ::memset(_msg, '\0', 1024 * sizeof(char));
            ::snprintf(_msg, 1024, "Invalid argument in: (\"%s\", nullptr, nullptr, %p)",
                       _mangled_name, (void*) _status);
            ::perror(_msg);
            break;
        }
        default: break;
    };

    // free allocated buffer
    ::free(_demang);
    return _demangled_name;
#else
    (void) _status;
    return _mangled_name;
#endif
}

//--------------------------------------------------------------------------------------//

inline std::string
demangle(const std::string& _str, int* _status = nullptr)
{
    return demangle(_str.c_str(), _status);
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline auto
try_demangle()
{
    // static because a type demangle will always be the same
    static auto _val = []() {
        // wrap the type in type_list and then extract ... from tim::type_list<...>
        auto _tmp = ::tim::demangle(typeid(type_list<Tp>).name());
        auto _key = std::string{ "type_list" };
        auto _idx = _tmp.find(_key);
        _idx      = _tmp.find('<', _idx);
        _tmp      = _tmp.substr(_idx + 1);
        _idx      = _tmp.find_last_of('>');
        _tmp      = _tmp.substr(0, _idx);
        // strip trailing whitespaces
        while((_idx = _tmp.find_last_of(' ')) == _tmp.length() - 1)
            _tmp = _tmp.substr(0, _idx);
        return _tmp;
    }();
    return _val;
}

//--------------------------------------------------------------------------------------//

template <typename Tp>
inline auto
demangle()
{
    // static because a type demangle will always be the same
    static auto _val = demangle(typeid(Tp).name());
    return _val;
}

//--------------------------------------------------------------------------------------//

namespace internal
{
template <typename Tp, typename Up = Tp>
inline auto
typeid_hash(int) -> decltype(demangle<Tp>(), size_t{})
{
    return std::type_index(typeid(Tp)).hash_code();
}
//
template <typename Tp, typename Up = Tp>
inline auto
typeid_hash(long)
{
    return 0;
}
}  // namespace internal

template <typename Tp>
inline auto
typeid_hash()
{
    return internal::typeid_hash<Tp>(0);
}

//--------------------------------------------------------------------------------------//

template <typename T>
inline T
from_string(const std::string& str)
{
    std::stringstream ss;
    ss << str;
    T val{};
    ss >> val;
    return val;
}

//--------------------------------------------------------------------------------------//

template <typename T>
inline T
from_string(const char* cstr)
{
    std::stringstream ss;
    ss << cstr;
    T val{};
    ss >> val;
    return val;
}

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE std::string
                        dirname(std::string _fname);

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE int
makedir(std::string _dir, int umask = DEFAULT_UMASK);

//--------------------------------------------------------------------------------------//

TIMEMORY_UTILITY_INLINE bool
get_bool(const std::string& strbool, bool _default = false) noexcept;

//--------------------------------------------------------------------------------------//
//
#if defined(TIMEMORY_UNIX)
//
TIMEMORY_UTILITY_INLINE std::string
                        demangle_backtrace(const char* cstr);
//
TIMEMORY_UTILITY_INLINE std::string
                        demangle_backtrace(const std::string& str);
//
TIMEMORY_UTILITY_INLINE std::string
                        demangle_unw_backtrace(const char* cstr);
//
TIMEMORY_UTILITY_INLINE std::string
                        demangle_unw_backtrace(const std::string& str);
//
template <size_t Depth, size_t Offset = 1>
TIMEMORY_NOINLINE auto
get_backtrace()
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    // destination
    std::array<char[512], Depth> btrace{};
    for(auto& itr : btrace)
        itr[0] = '\0';

    // plus one for this stack-frame
    std::array<void*, Depth + Offset> buffer{};
    buffer.fill(nullptr);

    // size of returned buffer
    auto sz = backtrace(buffer.data(), Depth + Offset);
    // size of relevant data
    auto n = sz - Offset;

    // skip ahead (Offset + 1) stack frames
    char** bsym = backtrace_symbols(buffer.data() + Offset, n);

    // report errors
    if(bsym == nullptr)
    {
        perror("backtrace_symbols");
    }
    else
    {
        for(decltype(n) i = 0; i < n; ++i)
        {
            snprintf(btrace[i], sizeof(btrace[i]), "%s", bsym[i]);
        }
    }
    return btrace;
}
//
template <size_t Depth, size_t Offset = 1>
TIMEMORY_NOINLINE auto
get_unw_backtrace()
{
#    if defined(TIMEMORY_USE_LIBUNWIND)
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    unw_cursor_t  cursor;
    unw_context_t context;

    // destination
    std::array<char[512], Depth> btrace{};
    for(auto& itr : btrace)
        itr[0] = '\0';

    // Initialize cursor to current frame for local unwinding.
    unw_getcontext(&context);
    if(unw_init_local(&cursor, &context) < 0)
    {
        return btrace;
    }

    size_t tot_idx = 0;
    while(unw_step(&cursor) > 0)
    {
        unw_word_t ip;   // stack pointer
        unw_word_t off;  // offset
        auto       _idx = ++tot_idx;
        if(_idx >= Depth + Offset)
            break;
        unw_get_reg(&cursor, UNW_REG_IP, &ip);
        if(ip == 0)
            break;
        char name[496];
        name[0] = '\0';
        if(unw_get_proc_name(&cursor, name, sizeof(name), &off) == 0)
        {
            if(_idx >= Offset)
            {
                auto _lidx = _idx - Offset;
                if(off)
                    snprintf(btrace[_lidx], sizeof(btrace[_lidx]), "%s +0x%lx", name,
                             (long) off);
                else
                    snprintf(btrace[_lidx], sizeof(btrace[_lidx]), "%s", name);
            }
        }
    }
#    else
    std::array<char[512], Depth> btrace{};
    throw std::runtime_error("[timemory]> libunwind not available");
#    endif
    return btrace;
}
//
template <size_t Depth, size_t Offset = 1, typename Func>
TIMEMORY_NOINLINE auto
get_backtrace(Func&& func)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    using type = std::result_of_t<Func(const char[512])>;
    // destination
    std::array<type, Depth> btrace{};

    auto&& _data = ::tim::get_backtrace<Depth, Offset + 1>();
    auto   _n    = _data.size();
    for(decltype(_n) i = 0; i < _n; ++i)
        btrace[i] = func(_data[i]);
    return btrace;
}
//
template <size_t Depth, size_t Offset = 1, typename Func>
TIMEMORY_NOINLINE auto
get_unw_backtrace(Func&& func)
{
    static_assert(Depth > 0, "Error !(Depth > 0)");
    static_assert(Offset >= 0, "Error !(Offset >= 0)");

    using type = std::result_of_t<Func(const char[512])>;
    // destination
    std::array<type, Depth> btrace{};

    auto&& _data = ::tim::get_unw_backtrace<Depth, Offset + 1>();
    auto   _n    = _data.size();
    for(decltype(_n) i = 0; i < _n; ++i)
        btrace[i] = func(_data[i]);
    return btrace;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 1>
TIMEMORY_NOINLINE auto
get_demangled_backtrace()
{
    auto demangle_bt = [](const char cstr[512]) { return demangle_backtrace(cstr); };
    return get_backtrace<Depth, Offset + 1>(demangle_bt);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 1>
TIMEMORY_NOINLINE auto
get_demangled_unw_backtrace()
{
    auto demangle_bt = [](const char cstr[512]) { return demangle_unw_backtrace(cstr); };
    return get_unw_backtrace<Depth, Offset + 1>(demangle_bt);
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 2>
TIMEMORY_NOINLINE std::ostream&
                  print_backtrace(std::ostream& os = std::cerr, std::string _prefix = "",
                                  const std::string& _info = "", const std::string& _indent = "    ")
{
    os << _indent.substr(0, _indent.length() / 2) << "Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << ":\n" << std::flush;
    auto bt = tim::get_backtrace<Depth, Offset>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(strlen(itr) > 0)
            os << _indent << _prefix << itr << "\n";
    }
    os << std::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 2>
TIMEMORY_NOINLINE std::ostream&
                  print_demangled_backtrace(std::ostream& os = std::cerr, std::string _prefix = "",
                                            const std::string& _info   = "",
                                            const std::string& _indent = "    ")
{
    os << _indent.substr(0, _indent.length() / 2) << "Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << ":\n" << std::flush;
    auto bt = tim::get_demangled_backtrace<Depth, Offset>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(itr.length() > 0)
            os << _indent << _prefix << itr << "\n";
    }
    os << std::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 2>
TIMEMORY_NOINLINE std::ostream&
                  print_unw_backtrace(std::ostream& os = std::cerr, std::string _prefix = "",
                                      const std::string& _info = "", const std::string& _indent = "    ")
{
    os << _indent.substr(0, _indent.length() / 2) << "Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << ":\n" << std::flush;
    auto bt = tim::get_unw_backtrace<Depth, Offset>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(strlen(itr) > 0)
            os << _indent << _prefix << itr << "\n";
    }
    os << std::flush;
    return os;
}
//
//--------------------------------------------------------------------------------------//
//
template <size_t Depth, size_t Offset = 3>
TIMEMORY_NOINLINE std::ostream&
                  print_demangled_unw_backtrace(std::ostream& os = std::cerr, std::string _prefix = "",
                                                const std::string& _info   = "",
                                                const std::string& _indent = "    ")
{
    os << _indent.substr(0, _indent.length() / 2) << "Backtrace";
    if(!_info.empty())
        os << " " << _info;
    os << ":\n" << std::flush;
    auto bt = tim::get_demangled_unw_backtrace<Depth, Offset>();
    if(!_prefix.empty() && _prefix.find_last_of(" \t") != _prefix.length() - 1)
        _prefix += " ";
    for(const auto& itr : bt)
    {
        if(itr.length() > 0)
            os << _indent << _prefix << itr << "\n";
    }
    os << std::flush;
    return os;
}
//
#else
//
// define these dummy functions since they are used in operation::decode
//
static inline auto
demangle_backtrace(const char* cstr)
{
    return std::string{ cstr };
}
//
static inline auto
demangle_backtrace(const std::string& str)
{
    return str;
}
//
template <size_t Depth, size_t Offset = 2>
static inline std::ostream&
print_backtrace(std::ostream& os = std::cerr, std::string = {}, std::string = {},
                std::string = {})
{
    os << "[timemory]> Backtrace not supported on this platform\n";
    return os;
}
//
template <size_t Depth, size_t Offset = 3>
static inline std::ostream&
print_demangled_backtrace(std::ostream& os = std::cerr, std::string = {},
                          std::string = {}, std::string = {})
{
    os << "[timemory]> Backtrace not supported on this platform\n";
    return os;
}
//
template <size_t Depth, size_t Offset = 2>
static inline std::ostream&
print_unw_backtrace(std::ostream& os = std::cerr, std::string = {}, std::string = {},
                    std::string = {})
{
    os << "[timemory]> libunwind backtrace not supported on this platform\n";
    return os;
}
//
template <size_t Depth, size_t Offset = 3>
static inline std::ostream&
print_demangled_unw_backtrace(std::ostream& os = std::cerr, std::string = {},
                              std::string = {}, std::string = {})
{
    os << "[timemory]> libunwind backtrace not supported on this platform\n";
    return os;
}
//
#endif
//
//--------------------------------------------------------------------------------------//
//  delimit a string into a set
//
template <typename ContainerT = std::vector<std::string>,
          typename PredicateT = std::function<std::string(const std::string&)>>
inline ContainerT
delimit(const std::string& line, const std::string& delimiters = "\"',;: ",
        PredicateT&& predicate = [](const std::string& s) -> std::string { return s; })
{
    ContainerT _result{};
    size_t     _beginp = 0;  // position that is the beginning of the new string
    size_t     _delimp = 0;  // position of the delimiter in the string
    while(_beginp < line.length() && _delimp < line.length())
    {
        // find the first character (starting at _delimp) that is not a delimiter
        _beginp = line.find_first_not_of(delimiters, _delimp);
        // if no a character after or at _end that is not a delimiter is not found
        // then we are done
        if(_beginp == std::string::npos)
            break;
        // starting at the position of the new string, find the next delimiter
        _delimp = line.find_first_of(delimiters, _beginp);
        std::string _tmp{};
        try
        {
            // starting at the position of the new string, get the characters
            // between this position and the next delimiter
            _tmp = line.substr(_beginp, _delimp - _beginp);
        } catch(std::exception& e)
        {
            // print the exception but don't fail, unless maybe it should?
            fprintf(stderr, "%s\n", e.what());
        }
        // don't add empty strings
        if(!_tmp.empty())
        {
            _result.insert(_result.end(), predicate(_tmp));
        }
    }
    return _result;
}
//
//--------------------------------------------------------------------------------------//
///  \brief apply a string transformation to substring inbetween a common delimiter.
///  e.g.
//
template <typename PredicateT = std::function<std::string(const std::string&)>>
inline std::string
str_transform(const std::string& input, const std::string& _begin,
              const std::string& _end, PredicateT&& predicate)
{
    size_t      _beg_pos = 0;  // position that is the beginning of the new string
    size_t      _end_pos = 0;  // position of the delimiter in the string
    std::string _result  = input;
    while(_beg_pos < _result.length() && _end_pos < _result.length())
    {
        // find the first sequence of characters after the end-position
        _beg_pos = _result.find(_begin, _end_pos);

        // if sequence wasn't found, we are done
        if(_beg_pos == std::string::npos)
            break;

        // starting after the position of the first delimiter, find the end sequence
        if(!_end.empty())
            _end_pos = _result.find(_end, _beg_pos + 1);
        else
            _end_pos = _beg_pos + _begin.length();

        // break if not found
        if(_end_pos == std::string::npos)
            break;

        // length of the substr being operated on
        auto _len = _end_pos - _beg_pos;

        // get the substring between the two delimiters (including first delimiter)
        auto _sub = _result.substr(_beg_pos, _len);

        // apply the transform
        auto _transformed = predicate(_sub);

        // only replace if necessary
        if(_sub != _transformed)
        {
            _result = _result.replace(_beg_pos, _len, _transformed);
            // move end to the end of transformed string
            _end_pos = _beg_pos + _transformed.length();
        }
    }
    return _result;
}

//--------------------------------------------------------------------------------------//
//
TIMEMORY_UTILITY_INLINE std::vector<std::string>
                        read_command_line(pid_t _pid);

//======================================================================================//
//
//  path
//
//======================================================================================//

namespace utility
{
class path : public std::string
{
public:
    using size_type = std::string::size_type;

public:
    TIMEMORY_UTILITY_INLINE path(const std::string& _path);
    TIMEMORY_UTILITY_INLINE path(char* _path);
    TIMEMORY_UTILITY_INLINE path(const path& rhs);
    TIMEMORY_UTILITY_INLINE path(const char* _path);

    TIMEMORY_UTILITY_INLINE path& operator=(const std::string& rhs);
    TIMEMORY_UTILITY_INLINE path& operator=(const path& rhs);
    TIMEMORY_UTILITY_INLINE path& insert(size_type __pos, const std::string& __s);
    TIMEMORY_UTILITY_INLINE path& insert(size_type __pos, const path& __s);

    // OS-dependent representation
    static std::string osrepr(std::string _path);
    static std::string os();
    static std::string inverse();
};
}  // namespace utility

//--------------------------------------------------------------------------------------//

template <typename T>
TIMEMORY_INLINE size_t
                get_hash(T&& obj)
{
    return std::hash<decay_t<T>>()(std::forward<T>(obj));
}

//--------------------------------------------------------------------------------------//

TIMEMORY_INLINE size_t
                get_hash(const string_view_t& str)
{
    return std::hash<string_view_t>{}(str);
}

//--------------------------------------------------------------------------------------//

TIMEMORY_INLINE size_t
                get_hash(const char* cstr)
{
    return std::hash<string_view_t>{}(cstr);
}

//--------------------------------------------------------------------------------------//

template <typename T>
struct hasher
{
    inline size_t operator()(T&& val) const { return get_hash(std::forward<T>(val)); }
    inline size_t operator()(const T& val) const { return get_hash(val); }
};

//--------------------------------------------------------------------------------------//
/*
#if defined(TIMEMORY_UNIX) && \ (defined(TIMEMORY_UTILITY_SOURCE) ||
defined(TIMEMORY_USE_UTILITY_EXTERN))
//
extern template auto
get_backtrace<2, 1>();
extern template auto
get_backtrace<3, 1>();
extern template auto
get_backtrace<4, 1>();
extern template auto
get_backtrace<8, 1>();
extern template auto
get_backtrace<16, 1>();
extern template auto
get_backtrace<32, 1>();
//
extern template auto
get_demangled_backtrace<3, 2>();
extern template auto
get_demangled_backtrace<4, 2>();
extern template auto
get_demangled_backtrace<8, 2>();
extern template auto
get_demangled_backtrace<16, 2>();
extern template auto
get_demangled_backtrace<32, 2>();
//
extern template auto
get_backtrace<3, 2>();
extern template auto
get_backtrace<4, 2>();
extern template auto
get_backtrace<8, 2>();
extern template auto
get_backtrace<16, 2>();
extern template auto
get_backtrace<32, 2>();
//
extern template auto
get_demangled_backtrace<4, 3>();
extern template auto
get_demangled_backtrace<8, 3>();
extern template auto
get_demangled_backtrace<16, 3>();
extern template auto
get_demangled_backtrace<32, 3>();
//
#endif
*/
//--------------------------------------------------------------------------------------//

}  // namespace tim

//--------------------------------------------------------------------------------------//

#if defined(TIMEMORY_UTILITY_HEADER_MODE)
#    include "timemory/utility/utility.cpp"
#endif
