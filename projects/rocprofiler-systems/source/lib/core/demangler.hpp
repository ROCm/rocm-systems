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

        std::lock_guard<std::mutex> _lk{ m_mutex };

        auto _it = m_cache.find(_mangled_name);
        if(_it != m_cache.end()) return _it->second;

        auto _result = demangle_impl(_mangled_name.data());
        m_cache.emplace(_mangled_name, _result);
        return _result;
    }

private:
    std::mutex                                      m_mutex;
    std::map<std::string, std::string, std::less<>> m_cache;

    static std::string demangle_impl(const char* _mangled_name)
    {
        int                                         _status = 0;
        std::unique_ptr<char, decltype(&std::free)> _demangled(
            DemanglerTp::demangle(_mangled_name, nullptr, nullptr, &_status), &std::free);

        if(_status != 0 || !_demangled) return std::string{ _mangled_name };

        return std::string{ _demangled.get() };
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
