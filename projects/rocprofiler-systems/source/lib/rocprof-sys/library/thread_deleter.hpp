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
        constexpr bool delete_pointer =
            (use_placement_new_when_generating_unique_ptr<Tp>::value == false);

        thread_deleter<void>{}();
        if constexpr(delete_pointer) delete ptr;

        (void) ptr;
    }
};
}  // namespace rocprofsys
