// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/defines.hpp"

#include <functional>
#include <utility>

namespace rocprofiler
{
namespace common
{
struct scope_destructor
{
    /// \fn scope_destructor(FuncT&& _fini, InitT&& _init)
    /// \tparam FuncT "std::function<void()> or void (*)()"
    /// \tparam InitT "std::function<void()> or void (*)()"
    /// \param _fini Function to execute when object is destroyed
    /// \param _init Function to execute when object is created (optional)
    ///
    /// \brief Provides a utility to perform an operation when exiting a scope.
    template <typename FuncT, typename InitT = void (*)()>
    scope_destructor(
        FuncT&& _fini,
        InitT&& _init = []() {});

    ~scope_destructor() { m_functor(); }

    // delete copy operations
    scope_destructor(const scope_destructor&) = delete;
    scope_destructor& operator=(const scope_destructor&) = delete;

    // allow move operations
    scope_destructor(scope_destructor&& rhs) noexcept;
    scope_destructor& operator=(scope_destructor&& rhs) noexcept;

private:
    std::function<void()> m_functor = []() {};
};

template <typename FuncT, typename InitT>
scope_destructor::scope_destructor(FuncT&& _fini, InitT&& _init)
: m_functor{std::forward<FuncT>(_fini)}
{
    _init();
}

inline scope_destructor::scope_destructor(scope_destructor&& rhs) noexcept
: m_functor{std::move(rhs.m_functor)}
{
    rhs.m_functor = []() {};
}

inline scope_destructor&
scope_destructor::operator=(scope_destructor&& rhs) noexcept
{
    if(this != &rhs)
    {
        m_functor     = std::move(rhs.m_functor);
        rhs.m_functor = []() {};
    }
    return *this;
}
}  // namespace common
}  // namespace rocprofiler
