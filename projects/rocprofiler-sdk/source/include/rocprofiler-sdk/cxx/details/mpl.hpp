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
//

#pragma once

#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace sdk
{
namespace mpl
{
template <typename Tp>
struct string_support
{
    using type        = Tp;
    using return_type = void;

    static constexpr auto value = false;
    static constexpr void default_value() {}
};

template <>
struct string_support<const char*>
{
    using type        = const char*;
    using return_type = type;

    static constexpr auto value = true;
    static constexpr type default_value() { return nullptr; }

    type operator()(const char* val) const { return val; }
};

template <>
struct string_support<std::string_view>
{
    using type        = std::string_view;
    using return_type = type;

    static constexpr auto value = true;
    static constexpr type default_value() { return type{}; }

    type operator()(const char* val) const { return type{val}; }
};

template <>
struct string_support<std::string>
{
    using type        = std::string;
    using return_type = type;

    static constexpr auto value = true;
    static type           default_value() { return type{}; }

    type operator()(const char* val) const { return type{val}; }
};
}  // namespace mpl
}  // namespace sdk
}  // namespace rocprofiler
