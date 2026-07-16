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

#pragma once

#include "lib/common/logging.hpp"

#include <unistd.h>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

namespace rocprofiler
{
namespace common
{
namespace impl
{
struct sfinae
{};

std::optional<std::string> get_env_direct(std::string_view);

std::string get_env(std::string_view, std::string_view);

std::string
get_env(std::string_view, const char*);

bool
get_env(std::string_view, bool);

template <typename Tp>
Tp get_env(std::string_view,
           Tp,
           std::enable_if_t<std::is_integral<Tp>::value || std::is_floating_point<Tp>::value,
                            sfinae> = {});

int
set_env(std::string_view, bool, int override = 0);

template <typename Tp>
int
set_env(std::string_view, Tp, int override = 0);
}  // namespace impl

// Get environment variable value, distinguishing "not set" from "set to empty".
//
// This is the lowest-level API for reading environment variables when you need
// to distinguish between:
//   - Variable not set:         returns std::nullopt
//   - Variable set to "":       returns std::optional("")
//   - Variable set to "value":  returns std::optional("value")
//
// Usage:
//   auto val = get_env_optional("MY_VAR");
//   if(!val) {
//       // Not set
//   } else if(val->empty()) {
//       // Set to empty string
//   } else {
//       // Set with value: *val
//   }
//
// To check presence: if(get_env_optional("MY_VAR").has_value()) { ... }
// For most cases, use get_env(name, default) instead.
inline std::optional<std::string>
get_env_optional(std::string_view env_id)
{
    return impl::get_env_direct(env_id);
}

template <typename Tp>
inline auto
get_env(std::string_view env_id, Tp&& _default)
{
    if constexpr(std::is_enum<Tp>::value)
    {
        using Up = std::underlying_type_t<Tp>;
        // cast to underlying type -> get_env -> cast to enum type
        return static_cast<Tp>(impl::get_env(env_id, static_cast<Up>(_default)));
    }
    else
    {
        return impl::get_env(env_id, std::forward<Tp>(_default));
    }
}

template <typename Tp>
inline auto
set_env(std::string_view env_id, Tp&& value, int override = 0)
{
    return impl::set_env(env_id, std::forward<Tp>(value), override);
}

// Returns true if the process is running in a "secure" execution context, i.e.
// the AT_SECURE auxiliary-vector entry is set (setuid/setgid binaries, files
// with capabilities, etc.). In such contexts, environment variables that are
// controllable by an unprivileged user must NOT be honored to load code (e.g.
// dlopen of tool libraries), otherwise a local attacker could inject a library
// into a privileged process. The result is cached on first call.
bool
is_at_secure();

struct env_config
{
    std::string env_name  = {};
    std::string env_value = {};
    int         overwrite = 0;  // -1=only if set, 0=no overwrite, 1=overwrite

    auto operator()(bool _verbose = false) const
    {
        if(env_name.empty()) return -1;
        // overwrite < 0: only modify if variable already exists
        if(overwrite < 0 && !get_env_optional(env_name).has_value())
            return 0;
        else if(_verbose)
        {
            ROCP_INFO << "[rocprofiler][set_env] setenv(\"" << env_name << "\", \"" << env_value
                      << "\", " << overwrite << ")\n";
        }
        auto _ow = (overwrite < 0) ? 1 : overwrite;
        return (env_value.empty() && _ow > 0) ? unsetenv(env_name.c_str())
                                              : setenv(env_name.c_str(), env_value.c_str(), _ow);
    }
};

struct env_store
{
    template <template <typename, typename...> class ContainerT, typename... TailT>
    explicit env_store(ContainerT<env_config, TailT...>&& _container);
    explicit env_store(std::initializer_list<env_config>&& _container);

    ~env_store();
    env_store(const env_store&)     = default;
    env_store(env_store&&) noexcept = default;
    env_store& operator=(const env_store&) = default;
    env_store& operator=(env_store&&) noexcept = default;

    bool push();
    bool pop(bool unset_if_empty = true);
    bool is_pushed() const { return m_pushed; }

private:
    bool                    m_pushed   = false;
    std::vector<env_config> m_original = {};
    std::vector<env_config> m_modified = {};
};

template <template <typename, typename...> class ContainerT, typename... TailT>
env_store::env_store(ContainerT<env_config, TailT...>&& _container)
{
    for(const auto& itr : _container)
    {
        m_original.emplace_back(env_config{itr.env_name, get_env(itr.env_name, ""), 1});
        m_modified.emplace_back(env_config{itr.env_name, itr.env_value, itr.overwrite});
    }
}
}  // namespace common
}  // namespace rocprofiler
