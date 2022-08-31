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

#pragma once

#include "library/defines.hpp"

#include <timemory/mpl/concepts.hpp>

#include <memory>
#include <optional>

namespace omnitrace
{
namespace concepts = ::tim::concepts;  // NOLINT

template <typename Tp>
struct thread_deleter;

// unique ptr type for omnitrace
template <typename Tp>
using unique_ptr_t = std::unique_ptr<Tp, thread_deleter<Tp>>;
}  // namespace omnitrace

namespace tim
{
namespace concepts
{
template <typename Tp>
struct is_unique_pointer : std::false_type
{};

template <typename Tp>
struct is_unique_pointer<::omnitrace::unique_ptr_t<Tp>> : std::true_type
{};

template <typename Tp>
struct is_unique_pointer<std::unique_ptr<Tp>> : std::true_type
{};

template <typename Tp>
struct is_optional : std::false_type
{};

template <typename Tp>
struct is_optional<std::optional<Tp>> : std::true_type
{};
}  // namespace concepts
}  // namespace tim
