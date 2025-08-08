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

#include "core/defines.hpp"

#include <timemory/backends/threading.hpp>
#include <timemory/mpl/type_traits.hpp>
#include <timemory/operations/types.hpp>
#include <timemory/utility/macros.hpp>
#include <timemory/utility/type_list.hpp>

namespace rocprofsys
{
namespace component
{
namespace
{
template <typename... Tp>
struct ensure_storage
{
    ROCPROFSYS_DEFAULT_OBJECT(ensure_storage)

    void operator()() const { ROCPROFSYS_FOLD_EXPRESSION((*this)(tim::type_list<Tp>{})); }

private:
    template <typename Up, std::enable_if_t<tim::trait::is_available<Up>::value, int> = 0>
    void operator()(tim::type_list<Up>) const
    {
        using namespace tim;
        static thread_local auto _storage = operation::get_storage<Up>{}();
        static thread_local auto _tid     = threading::get_id();
        static thread_local auto _dtor =
            scope::destructor{ []() { operation::set_storage<Up>{}(nullptr, _tid); } };

        tim::operation::set_storage<Up>{}(_storage, _tid);
        if(_tid == 0 && !_storage) tim::trait::runtime_enabled<Up>::set(false);
    }

    template <typename Up,
              std::enable_if_t<!tim::trait::is_available<Up>::value, long> = 0>
    void operator()(tim::type_list<Up>) const
    {
        tim::trait::runtime_enabled<Up>::set(false);
    }
};
}  // namespace
}  // namespace component
}  // namespace rocprofsys
