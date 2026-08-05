// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/common/environment.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <sys/auxv.h>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <string_view>

namespace rocprofiler
{
namespace common
{
namespace impl
{
// Safely read environment variable directly from environ array.
// This avoids issues with bash's custom getenv() implementation which
// breaks when setenv() is called before bash initializes its internal tables.
//
// THREAD SAFETY: Reads are NOT safe against concurrent setenv()/putenv()/
// unsetenv() from any thread, because glibc may reallocate the environ
// array itself (not just mutate entries). Prefer reading at init time or
// caching in a function-local static.
std::optional<std::string>
get_env_direct(std::string_view name)
{
    if(name.empty() || !environ) return std::nullopt;

    for(char** env = environ; *env; ++env)
    {
        std::string_view entry{*env};
        if(entry.size() > name.size() && entry.compare(0, name.size(), name) == 0 &&
           entry[name.size()] == '=')
        {
            // copy the value so callers do not retain pointers into environ.
            return std::string{entry.substr(name.size() + 1)};
        }
    }

    return std::nullopt;
}

std::string
get_env(std::string_view env_id, std::string_view _default)
{
    auto env_var = get_env_direct(env_id);
    if(env_var) return *env_var;
    return std::string{_default};
}

std::string
get_env(std::string_view env_id, const char* _default)
{
    return get_env(env_id, std::string_view{_default});
}

bool
get_env(std::string_view env_id, bool _default)
{
    if(env_id.empty()) return _default;
    auto env_var = get_env_direct(env_id);
    if(env_var)
    {
        if(env_var->empty())
        {
            ROCP_FATAL << fmt::format("No boolean value provided for {}", env_id);
        }

        if(env_var->find_first_not_of("0123456789") == std::string_view::npos)
        {
            return static_cast<bool>(std::stoi(*env_var));
        }

        // Convert to lowercase in-place (cast to unsigned char to avoid UB)
        for(size_t i = 0; i < env_var->length(); ++i)
            (*env_var)[i] =
                static_cast<char>(std::tolower(static_cast<unsigned char>((*env_var)[i])));

        for(const auto& itr : {"off", "false", "no", "n", "f", "0"})
            if(*env_var == itr) return false;

        return true;
    }
    return _default;
}

template <typename Tp>
Tp
get_env(std::string_view env_id,
        Tp               _default,
        std::enable_if_t<std::is_integral<Tp>::value || std::is_floating_point<Tp>::value, sfinae>)
{
    static_assert(!std::is_same<Tp, bool>::value, "unexpected! should be using bool overload");
    static_assert(
        sizeof(Tp) <= sizeof(uint64_t),
        "change use of stol/stoul if instantiating for type larger than a 64-bit integer");

    auto env_var = get_env_direct(env_id);
    if(env_var)
    {
        try
        {
            if constexpr(std::is_integral<Tp>::value)
            {
                // use stol/stoul
                if constexpr(std::is_signed<Tp>::value)
                    return static_cast<Tp>(std::stol(*env_var));
                else
                    return static_cast<Tp>(std::stoul(*env_var));
            }
            else if constexpr(std::is_floating_point<Tp>::value)
            {
                return static_cast<Tp>(std::stod(*env_var));
            }
        } catch(std::exception& _e)
        {
            ROCP_ERROR << "[rocprofiler][get_env] Exception thrown converting getenv(\"" << env_id
                       << "\") = " << *env_var << " to " << cxx_demangle(typeid(Tp).name())
                       << " :: " << _e.what() << ". Using default value of " << _default << "\n";
        }
        return _default;
    }
    return _default;
}

// set_env uses standard ::setenv() (no wrapper needed).
// Unlike getenv(), bash does not interpose setenv(), so it works correctly.
int
set_env(std::string_view env_id, bool value, int override)
{
    return ::setenv(env_id.data(), (value) ? "1" : "0", override);
}

template <typename Tp>
int
set_env(std::string_view env_id,
        Tp               value,  // NOLINT(performance-unnecessary-value-param)
        int              override)
{
    auto str_value = std::stringstream{};
    str_value << value;
    return ::setenv(env_id.data(), str_value.str().c_str(), override);
}

#define SPECIALIZE_GET_ENV(TYPE)                                                                   \
    template TYPE get_env<TYPE>(                                                                   \
        std::string_view,                                                                          \
        TYPE,                                                                                      \
        std::enable_if_t<std::is_integral<TYPE>::value || std::is_floating_point<TYPE>::value,     \
                         sfinae>);

#define SPECIALIZE_SET_ENV(TYPE) template int set_env<TYPE>(std::string_view, TYPE, int);

SPECIALIZE_GET_ENV(int8_t)
SPECIALIZE_GET_ENV(int16_t)
SPECIALIZE_GET_ENV(int32_t)
SPECIALIZE_GET_ENV(int64_t)
SPECIALIZE_GET_ENV(uint8_t)
SPECIALIZE_GET_ENV(uint16_t)
SPECIALIZE_GET_ENV(uint32_t)
SPECIALIZE_GET_ENV(uint64_t)
SPECIALIZE_GET_ENV(float)
SPECIALIZE_GET_ENV(double)

SPECIALIZE_SET_ENV(const char*)
SPECIALIZE_SET_ENV(std::string)
SPECIALIZE_SET_ENV(std::string_view)
SPECIALIZE_SET_ENV(float)
SPECIALIZE_SET_ENV(double)
SPECIALIZE_SET_ENV(int8_t)
SPECIALIZE_SET_ENV(int16_t)
SPECIALIZE_SET_ENV(int32_t)
SPECIALIZE_SET_ENV(int64_t)
SPECIALIZE_SET_ENV(uint8_t)
SPECIALIZE_SET_ENV(uint16_t)
SPECIALIZE_SET_ENV(uint32_t)
SPECIALIZE_SET_ENV(uint64_t)
}  // namespace impl

bool
is_at_secure()
{
    // AT_SECURE is set by the kernel when the program was executed in a way that
    // requires "secure execution" (setuid/setgid, file capabilities, etc.).
    // Cache the value since it cannot change during the lifetime of the process.
    static const bool _v = (::getauxval(AT_SECURE) != 0);
    return _v;
}

env_store::env_store(std::initializer_list<env_config>&& _container)
{
    for(const auto& itr : _container)
    {
        m_original.emplace_back(env_config{itr.env_name, get_env(itr.env_name, ""), 1});
        m_modified.emplace_back(env_config{itr.env_name, itr.env_value, itr.overwrite});
    }
}

env_store::~env_store() { pop(); }

bool
env_store::push()
{
    // not that push ignored bc already pushed
    if(m_pushed) return false;

    for(const auto& itr : m_modified)
        itr();

    m_pushed = true;
    return true;
}

bool
env_store::pop(bool unset_if_empty)
{
    if(!m_pushed) return false;

    for(const auto& itr : m_original)
    {
        auto _current = get_env(itr.env_name, "");
        if(!unset_if_empty && itr.env_value.empty())
            continue;
        else if(_current == itr.env_value)
            continue;
        else if(_current != itr.env_value)
        {
            ROCP_INFO << fmt::format("[rocprofiler][env][pop] {}=\"{}\" => {}=\"{}\"",
                                     itr.env_name,
                                     _current,
                                     itr.env_name,
                                     itr.env_value);
        }
        itr();
    }

    m_pushed = false;
    return true;
}
}  // namespace common
}  // namespace rocprofiler
