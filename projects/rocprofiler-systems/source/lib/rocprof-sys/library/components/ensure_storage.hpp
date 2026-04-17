// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
