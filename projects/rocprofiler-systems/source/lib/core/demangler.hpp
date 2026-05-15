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

#include <cstdlib>
#include <cxxabi.h>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <string_view>

namespace rocprofsys
{
namespace utility
{

struct cxa_demangle_wrapper_impl
{
    static char* demangle(const char* _mangled_name, char* _output_buffer,
                          size_t* _length, int* _status)
    {
        return abi::__cxa_demangle(_mangled_name, _output_buffer, _length, _status);
    }
};

template <typename DemanglerTp = cxa_demangle_wrapper_impl>
struct demangler
{
    template <typename Tp>
    std::string demangle()
    {
        return demangle(typeid(Tp).name());
    }

    std::string demangle(std::string_view _mangled_name)
    {
        if(_mangled_name.empty()) return {};

        const auto result = try_get_from_cache(_mangled_name);
        if(result._found)
        {
            return result._cache_it->second;
        }

        // Possible improvement: Limit the cache size to avoid memory bloat
        return demangle_and_cache(_mangled_name);
    }

private:
    std::shared_mutex                               m_mutex;
    std::map<std::string, std::string, std::less<>> m_cache;
    using cache_iterator = typename decltype(m_cache)::iterator;

    struct cache_result
    {
        bool           _found;
        cache_iterator _cache_it;
    };

    static std::string demangle_impl(const char* _mangled_name)
    {
        int                                         _status = 0;
        std::unique_ptr<char, decltype(&std::free)> _demangled(
            DemanglerTp::demangle(_mangled_name, nullptr, nullptr, &_status), &std::free);

        if(_status != 0 || !_demangled) return std::string{ _mangled_name };

        return std::string{ _demangled.get() };
    }

    cache_result try_get_from_cache(std::string_view _mangled_name)
    {
        std::shared_lock<std::shared_mutex> _read_lock{ m_mutex };

        auto _it = m_cache.find(_mangled_name);
        if(_it != m_cache.end())
        {
            return { true, _it };
        }

        return { false, m_cache.end() };
    }

    std::string demangle_and_cache(std::string_view _mangled_name)
    {
        std::unique_lock<std::shared_mutex> _write_lock{ m_mutex };

        auto _it = m_cache.find(_mangled_name);
        if(_it != m_cache.end())
        {
            return _it->second;
        }

        auto _result = demangle_impl(_mangled_name.data());
        m_cache.emplace(_mangled_name, _result);
        return _result;
    }
};

inline demangler<>&
get_demangler()
{
    static demangler g_demangler;
    return g_demangler;
}

template <typename Tp>
inline std::string
demangle()
{
    return get_demangler().demangle<Tp>();
}

inline std::string
demangle(std::string_view name)
{
    return get_demangler().demangle(name);
}

}  // namespace utility
}  // namespace rocprofsys
