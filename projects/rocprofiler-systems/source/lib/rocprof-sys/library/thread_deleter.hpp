// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/concepts.hpp"

namespace rocprofsys
{
template <>
struct thread_deleter<void>
{
    void operator()() const;
};

extern template struct thread_deleter<void>;

template <typename Tp>
struct thread_deleter
{
    void operator()(Tp* ptr) const
    {
        constexpr bool k_delete_pointer = !never_delete_unique_ptr<Tp>::value;

        thread_deleter<void>{}();
        if constexpr(k_delete_pointer)
        {
            delete ptr;
        }

        (void) ptr;
    }
};
}  // namespace rocprofsys
