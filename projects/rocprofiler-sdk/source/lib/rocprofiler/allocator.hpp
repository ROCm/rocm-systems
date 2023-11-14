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

#include <memory>
#include "lib/common/defines.hpp"
#include "lib/common/memory/deleter.hpp"
#include "lib/common/memory/stateless_allocator.hpp"

namespace rocprofiler
{
namespace allocator
{
// declare this trivial type for common::memory::deleter specialization
struct static_data;
}  // namespace allocator

namespace common
{
namespace memory
{
template <>
struct deleter<allocator::static_data>
{
    // specialize the deleter call operator to invoke registration::finalize
    void operator()() const;
};
}  // namespace memory
}  // namespace common

namespace allocator
{
// use this allocator for static data which only gets deleted at the end of the application
template <typename Tp>
using static_data_allocator =
    common::memory::stateless_allocator<Tp, 64, common::memory::deleter<static_data>>;

// use this for unique_ptr
template <typename Tp>
struct static_data_deleter
{
    void operator()(Tp* ptr) const
    {
        common::memory::deleter<static_data>{}();
        delete ptr;
    }
};

template <typename Tp>
using unique_static_ptr_t = std::unique_ptr<Tp, static_data_deleter<Tp>>;

template <typename Tp, typename... Args>
decltype(auto)
make_unique_static(Args&&... args)
{
    return unique_static_ptr_t<Tp>{new Tp{std::forward<Args>(args)...}};
}
}  // namespace allocator
}  // namespace rocprofiler
