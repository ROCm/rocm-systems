// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <cstddef>
#include <functional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace rocprofiler
{
namespace common
{
namespace mpl
{
namespace impl
{
template <typename... Tp>
struct type_list
{
    static constexpr auto size() { return sizeof...(Tp); }
};

template <typename InTuple, typename OutTuple>
struct reverse;

template <template <typename...> class InTuple,
          typename InT,
          typename... InTail,
          template <typename...>
          class OutTuple,
          typename... OutTail>
struct reverse<InTuple<InT, InTail...>, OutTuple<OutTail...>>
: reverse<InTuple<InTail...>, OutTuple<InT, OutTail...>>
{};

template <template <typename...> class InTuple,
          template <typename...>
          class OutTuple,
          typename... OutTail>
struct reverse<InTuple<>, OutTuple<OutTail...>>
{
    using type = OutTuple<OutTail...>;
};

template <template <typename...> class InTuple, typename... InTail>
struct reverse<InTuple<InTail...>, void> : reverse<InTuple<InTail...>, InTuple<>>
{};

template <typename T>
struct function_traits;

template <typename T>
struct function_traits<T&> : function_traits<T>
{};

template <typename R, typename... Args>
struct function_traits<std::function<R(Args...)>>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...)>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R(Args...)>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

// member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...)>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = type_list<C&, Args...>;
};

// const member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = true;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = type_list<C&, Args...>;
};

// member object pointer
template <typename C, typename R>
struct function_traits<R(C::*)>
{
    static constexpr bool is_memfun = true;
    static constexpr bool is_const  = false;
    static const size_t   nargs     = 0;
    using result_type               = R;
    using args_type                 = type_list<>;
    using call_type                 = type_list<C&>;
};

#if __cplusplus >= 201703L

template <typename R, typename... Args>
struct function_traits<std::function<R(Args...) noexcept>>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R (*)(Args...) noexcept>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

template <typename R, typename... Args>
struct function_traits<R(Args...) noexcept>
{
    static constexpr bool   is_memfun = false;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = args_type;
};

// member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) noexcept>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = false;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = type_list<C&, Args...>;
};

// const member function pointer
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const noexcept>
{
    static constexpr bool   is_memfun = true;
    static constexpr bool   is_const  = true;
    static constexpr size_t nargs     = sizeof...(Args);
    using result_type                 = R;
    using args_type                   = type_list<Args...>;
    using call_type                   = type_list<C&, Args...>;
};

#endif
}  // namespace impl
}  // namespace mpl
}  // namespace common
}  // namespace rocprofiler
